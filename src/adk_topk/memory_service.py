"""ADK memory service backed by TopK: one isolated partition per app and user."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from topk_sdk import AsyncClient
from topk_sdk.error import CollectionAlreadyExistsError, CollectionNotFoundError, PartitionNotFoundError
from topk_sdk.query import LogicalExpr, field, fn, match, select, should
from topk_sdk.schema import semantic_index, text
from typing_extensions import override

from adk_topk._common import (
    METADATA_FIELD,
    METADATA_JSON_FIELD,
    filterable,
    make_client,
    partition_exists,
    partition_name,
    read,
    translate_filter,
)

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

SearchMode = Literal["hybrid", "semantic", "keyword"]

TEXT_FIELD = "text"
_RETURN_FIELDS = (TEXT_FIELD, "author", "session_id", "timestamp", "kind", METADATA_JSON_FIELD)
_BATCH = 200


class TopKMemoryService(BaseMemoryService):
    """Long-term memory for ADK agents, stored in `TopK <https://topk.io>`_.

    Every ``(app_name, user_id)`` pair gets its own TopK partition, so a search
    can only ever see that user's memories. TopK embeds the text on the server
    (``semantic_index()``), so you don't need an embedding model or key.

    ``search_memory`` runs hybrid search by default: semantic similarity and BM25
    ranked together in one query, so a paraphrase ("what can't I eat?") and an
    exact string ("order 88412") both find the right memory.

    Works with ADK's built-in ``load_memory`` and ``preload_memory`` tools, and
    with :class:`adk_topk.TopKMemoryToolset` for explicit remember/recall/forget.

    Example:
        .. code-block:: python

            from adk_topk import TopKMemoryService
            from google.adk.runners import Runner

            memory_service = TopKMemoryService(collection_name="adk_memory")
            runner = Runner(agent=agent, app_name="my_app", session_service=sessions,
                            memory_service=memory_service)
    """

    def __init__(
        self,
        collection_name: str = "adk_memory",
        *,
        client: AsyncClient | None = None,
        api_key: str | None = None,
        region: str | None = None,
        search_mode: SearchMode = "hybrid",
        top_k: int = 10,
        hybrid_alpha: float = 0.5,
        create_collection: bool = True,
    ) -> None:
        """Create the service.

        Args:
            collection_name: TopK collection that holds every user's memories. Created if missing.
            client: An existing ``topk_sdk.AsyncClient``. Overrides ``api_key`` and ``region``.
            api_key: TopK API key. Defaults to ``TOPK_API_KEY``. Create one at https://console.topk.io.
            region: TopK region, e.g. ``"aws-us-east-1-elastica"``. Defaults to ``TOPK_REGION``.
            search_mode: How ``search_memory`` ranks: ``"hybrid"`` (semantic + BM25),
                ``"semantic"`` or ``"keyword"`` (BM25 only).
            top_k: Memories returned per search.
            hybrid_alpha: Weight of semantic similarity in hybrid search; BM25 gets ``1 - hybrid_alpha``.
            create_collection: Create the collection on first write if it doesn't exist.
        """
        if search_mode not in ("hybrid", "semantic", "keyword"):
            msg = f"Unknown search_mode {search_mode!r}; use hybrid, semantic or keyword"
            raise ValueError(msg)
        if not 0.0 <= hybrid_alpha <= 1.0:
            msg = f"hybrid_alpha must be between 0 and 1, got {hybrid_alpha}"
            raise ValueError(msg)
        self.collection_name = collection_name
        self.search_mode: SearchMode = search_mode
        self.top_k = top_k
        self.hybrid_alpha = hybrid_alpha
        self.create_collection = create_collection
        self._client = client or make_client(api_key, region)
        self._collection_ready = not create_collection
        # Last write LSN per partition, so reads always see this service's own writes.
        self._lsn: dict[str, str] = {}
        self._known_partitions: set[str] = set()

    @property
    def client(self) -> AsyncClient:
        """The underlying ``topk_sdk.AsyncClient``."""
        return self._client

    @staticmethod
    def partition_for(app_name: str, user_id: str) -> str:
        """The TopK partition that holds this app's memories for this user."""
        return partition_name(app_name, user_id)

    # ------------------------------------------------------------------ writes

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        await self._add_events(session.app_name, session.user_id, session.events, session.id, None)

    @override
    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        await self._add_events(app_name, user_id, events, session_id, custom_metadata)

    @override
    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        docs = []
        for memory in memories:
            body = _text(memory.content)
            if not body:
                continue
            meta = {**(custom_metadata or {}), **memory.custom_metadata}
            docs.append(
                _doc(
                    memory.id or f"mem-{uuid.uuid4().hex}",
                    body,
                    kind="memory",
                    author=memory.author,
                    timestamp=_parse_ts(memory.timestamp),
                    metadata=meta,
                )
            )
        await self._upsert(self.partition_for(app_name, user_id), docs)

    async def remember(
        self,
        *,
        app_name: str,
        user_id: str,
        text: str,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Save one text memory and return its id."""
        memory_id = f"mem-{uuid.uuid4().hex[:16]}"
        await self.add_memory(
            app_name=app_name,
            user_id=user_id,
            memories=[
                MemoryEntry(
                    id=memory_id,
                    content=types.Content(role="user", parts=[types.Part(text=text)]),
                    author=author,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    custom_metadata=dict(metadata or {}),
                )
            ],
        )
        return memory_id

    @override
    async def delete_memory(self, *, app_name: str, user_id: str, memory_id: str | None = None) -> None:
        if not memory_id:
            return
        # Leave a tombstone instead of deleting, so add_session_to_memory doesn't
        # bring the event back the next time the same session is saved.
        await self._upsert(self.partition_for(app_name, user_id), [{"_id": memory_id, "deleted": True}])

    async def clear(self, *, app_name: str, user_id: str) -> None:
        """Delete every memory for this user, including tombstones."""
        partition = self.partition_for(app_name, user_id)
        collection = self._client.collection(self.collection_name, partition=partition)
        # Delete by filter rather than delete_partition(): a deleted partition stays
        # readable, and rejects writes, for about a minute.
        try:
            self._lsn[partition] = await collection.delete(field("_id").is_not_null())
        except (CollectionNotFoundError, PartitionNotFoundError):
            pass

    # ------------------------------------------------------------------ reads

    @override
    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        return await self.search(app_name=app_name, user_id=user_id, query=query)

    async def search(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        top_k: int | None = None,
        search_mode: SearchMode | None = None,
        filters: dict[str, Any] | LogicalExpr | None = None,
    ) -> SearchMemoryResponse:
        """Like ``search_memory``, with a per-call result count, mode and metadata filter."""
        if not query or not query.strip():
            return SearchMemoryResponse()
        mode = search_mode or self.search_mode
        scores: dict[str, Any] = {}
        if mode == "keyword":
            scores["kw"] = fn.bm25_score()
            text_filter = match(query, field=TEXT_FIELD)
            rank = field("kw")
        elif mode == "semantic":
            scores["dense"] = fn.semantic_similarity(TEXT_FIELD, query)
            text_filter = None
            rank = field("dense")
        elif mode == "hybrid":
            scores["dense"] = fn.semantic_similarity(TEXT_FIELD, query)
            scores["kw"] = fn.bm25_score()
            # should() scores keyword matches without dropping memories that have none.
            text_filter = should(query, field=TEXT_FIELD)
            rank = field("dense") * self.hybrid_alpha + field("kw") * (1.0 - self.hybrid_alpha)
        else:
            msg = f"Unknown search_mode {mode!r}; use hybrid, semantic or keyword"
            raise ValueError(msg)

        q = select(*_RETURN_FIELDS, **scores).filter(field("deleted").is_null())
        if text_filter is not None:
            q = q.filter(text_filter)
        meta_filter = translate_filter(filters)
        if meta_filter is not None:
            q = q.filter(meta_filter)
        q = q.sort(rank, asc=False).limit(max(1, top_k or self.top_k))

        rows = await self._query(self.partition_for(app_name, user_id), q)
        return SearchMemoryResponse(memories=[_to_entry(r) for r in rows])

    # ------------------------------------------------------------------ internals

    async def _add_events(
        self,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None,
        custom_metadata: Mapping[str, object] | None,
    ) -> None:
        docs = []
        for event in events:
            if getattr(event, "partial", False):
                continue
            body = _text(event.content)
            if not body:
                continue
            docs.append(
                _doc(
                    event.id,
                    body,
                    kind="event",
                    author=event.author,
                    timestamp=event.timestamp,
                    session_id=session_id,
                    metadata=dict(custom_metadata or {}),
                )
            )
        if not docs:
            return
        partition = self.partition_for(app_name, user_id)
        # A session is saved again and again as it grows; only write new events and
        # never overwrite a tombstone left by delete_memory.
        existing = await self._existing_ids(partition, [d["_id"] for d in docs])
        await self._upsert(partition, [d for d in docs if d["_id"] not in existing])

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        try:
            await self._client.collections().create(
                self.collection_name, schema={TEXT_FIELD: text().index(semantic_index())}
            )
        except CollectionAlreadyExistsError:
            pass
        self._collection_ready = True

    async def _upsert(self, partition: str, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        await self._ensure_collection()
        collection = self._client.collection(self.collection_name, partition=partition)
        for start in range(0, len(docs), _BATCH):
            self._lsn[partition] = await collection.upsert(docs[start : start + _BATCH])
        self._known_partitions.add(partition)

    async def _exists(self, partition: str) -> bool:
        if partition not in self._known_partitions:
            if not await partition_exists(self._client, self.collection_name, partition):
                return False
            self._known_partitions.add(partition)
        return True

    async def _existing_ids(self, partition: str, ids: list[str]) -> set[str]:
        if not await self._exists(partition):
            return set()
        collection = self._client.collection(self.collection_name, partition=partition)
        found: set[str] = set()
        for start in range(0, len(ids), _BATCH):
            batch = ids[start : start + _BATCH]
            got = await read(
                lambda b=batch: collection.get(b, fields=["deleted"], **self._read(partition)),
                {},
                written=partition in self._lsn,
            )
            found.update(got)
        return found

    async def _query(self, partition: str, q: Any) -> list[dict[str, Any]]:
        if not await self._exists(partition):
            return []
        collection = self._client.collection(self.collection_name, partition=partition)
        return await read(lambda: collection.query(q, **self._read(partition)), [], written=partition in self._lsn)

    def _read(self, partition: str) -> dict[str, Any]:
        lsn = self._lsn.get(partition)
        return {"lsn": lsn} if lsn else {}


def _text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "\n".join(p.text for p in content.parts if p.text and not getattr(p, "thought", False)).strip()


def _doc(
    doc_id: str,
    body: str,
    *,
    kind: str,
    author: str | None,
    timestamp: float | None,
    metadata: Mapping[str, Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "_id": doc_id,
        TEXT_FIELD: body,
        "kind": kind,
        METADATA_FIELD: filterable(dict(metadata)),
        METADATA_JSON_FIELD: json.dumps(dict(metadata), default=str),
    }
    if author:
        doc["author"] = author
    if timestamp is not None:
        doc["timestamp"] = float(timestamp)
    if session_id:
        doc["session_id"] = session_id
    return doc


def _parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _to_entry(row: dict[str, Any]) -> MemoryEntry:
    raw = row.get(METADATA_JSON_FIELD)
    metadata = json.loads(raw) if raw else {}
    if row.get("session_id"):
        metadata.setdefault("session_id", row["session_id"])
    ts = row.get("timestamp")
    return MemoryEntry(
        id=row["_id"],
        content=types.Content(role="user", parts=[types.Part(text=row.get(TEXT_FIELD) or "")]),
        author=row.get("author"),
        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds") if ts else None,
        custom_metadata=metadata,
    )
