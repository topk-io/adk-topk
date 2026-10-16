"""ADK toolsets: search over a TopK collection, and per-user remember/recall/forget."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset, ToolPredicate
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from topk_sdk import AsyncClient
from topk_sdk.error import CollectionAlreadyExistsError, CollectionNotFoundError, PartitionNotFoundError
from topk_sdk.query import field, fn, match, select, should
from topk_sdk.schema import semantic_index, text

from adk_topk._common import (
    FILTER_OPERATORS,
    METADATA_FIELD,
    METADATA_JSON_FIELD,
    filterable,
    make_client,
    partition_exists,
    read,
    translate_filter,
)
from adk_topk.memory_service import TopKMemoryService

TEXT_FIELD = "text"
_MAX_TOP_K = 20
_BATCH = 200


class _DescribedTool(FunctionTool):
    """A FunctionTool whose declaration uses ``self.description``, so it can carry extra context."""

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        declaration = super()._get_declaration()
        if declaration is not None:
            declaration.description = self.description
        return declaration


class TopKSearchToolset(BaseToolset):
    """Keyword, semantic and hybrid search over a TopK collection, as ADK tools.

    Gives the agent three tools and tells it when to use each one:

    * ``topk_keyword_search``: BM25, for exact strings (error codes, IDs, SKUs, names)
    * ``topk_semantic_search``: by meaning, for paraphrased or conceptual questions
    * ``topk_hybrid_search``: both, ranked together in one query; the default choice

    Every tool takes an optional metadata ``filters`` object. TopK embeds text on
    the server, so no embedding model is needed.

    Example:
        .. code-block:: python

            from adk_topk import TopKSearchToolset
            from google.adk.agents import Agent

            docs = TopKSearchToolset(
                collection_name="support_docs",
                corpus_description="product docs and incident runbooks",
                metadata_fields={"source": "'runbook' or 'docs'", "year": "integer"},
            )
            await docs.add_texts(["Error E-4021 means the card was declined."], metadatas=[{"source": "runbook"}])
            root_agent = Agent(model="gemini-flash-latest", name="support", tools=[docs])
    """

    def __init__(
        self,
        collection_name: str,
        *,
        client: AsyncClient | None = None,
        api_key: str | None = None,
        region: str | None = None,
        partition: str | None = None,
        corpus_description: str | None = None,
        metadata_fields: dict[str, str] | None = None,
        default_top_k: int = 5,
        max_chars_per_result: int = 1500,
        create_collection: bool = True,
        tool_filter: ToolPredicate | list[str] | None = None,
        tool_name_prefix: str | None = None,
    ) -> None:
        """Create the toolset.

        Args:
            collection_name: TopK collection to search. Created on first ``add_texts`` if missing.
            client: An existing ``topk_sdk.AsyncClient``. Overrides ``api_key`` and ``region``.
            api_key: TopK API key. Defaults to ``TOPK_API_KEY``.
            region: TopK region. Defaults to ``TOPK_REGION``.
            partition: Search and write only this partition (for example one per tenant).
            corpus_description: What the collection contains. Added to every tool description.
            metadata_fields: Filterable metadata fields and what they hold, added to every tool
                description, e.g. ``{"source": "'runbook' or 'docs'"}``.
            default_top_k: Results per search when the agent doesn't ask for a number.
            max_chars_per_result: Truncate each result's text to keep tool output small.
            create_collection: Create the collection on first write if it doesn't exist.
            tool_filter: ADK tool filter (names or predicate).
            tool_name_prefix: ADK tool name prefix.
        """
        super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)
        self.collection_name = collection_name
        self.partition = partition
        self.corpus_description = corpus_description
        self.metadata_fields = metadata_fields or {}
        self.default_top_k = default_top_k
        self.max_chars_per_result = max_chars_per_result
        self._client = client or make_client(api_key, region)
        self._collection_ready = not create_collection
        self._lsn: str | None = None
        self._partition_known = False
        self._tools = [
            self._tool(self.topk_keyword_search),
            self._tool(self.topk_semantic_search),
            self._tool(self.topk_hybrid_search),
        ]

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        return [t for t in self._tools if self._is_tool_selected(t, readonly_context)]

    # ------------------------------------------------------------------ indexing

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Upsert texts into the collection. Documents with an existing id are replaced."""
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            msg = "metadatas must have the same length as texts"
            raise ValueError(msg)
        if ids is not None and len(ids) != len(texts):
            msg = "ids must have the same length as texts"
            raise ValueError(msg)
        ids = [i or str(uuid.uuid4()) for i in (ids or [None] * len(texts))]
        metas = list(metadatas or [{} for _ in texts])
        docs = [
            {
                "_id": doc_id,
                TEXT_FIELD: body,
                METADATA_FIELD: filterable(meta or {}),
                METADATA_JSON_FIELD: json.dumps(meta or {}, default=str),
            }
            for doc_id, body, meta in zip(ids, texts, metas, strict=True)
        ]
        await self._ensure_collection()
        collection = self._collection()
        for start in range(0, len(docs), _BATCH):
            self._lsn = await collection.upsert(docs[start : start + _BATCH])
        self._partition_known = True
        return ids

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete documents by id. Missing ids are ignored."""
        if not ids:
            return
        try:
            self._lsn = await self._collection().delete(list(ids))
        except (CollectionNotFoundError, PartitionNotFoundError):
            pass

    # ------------------------------------------------------------------ tools

    async def topk_keyword_search(
        self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Exact keyword search (BM25).

        Use this when the user mentions a specific string that must appear in the
        document: an error code like E-4021, an ID, a SKU, a ticket number, a function
        or product name, or a quoted phrase. Only documents containing the query terms
        are returned, so an empty result means the term appears nowhere.

        Args:
            query: The exact terms to look for, e.g. "E-4021" or "rotate_api_key".
            top_k: How many results to return (default 5, max 20).
            filters: Optional metadata filter, e.g. {"source": "runbook"} or {"year": {"$gte": 2024}}.

        Returns:
            The matching documents with id, score, metadata and text.
        """
        return await self._search(query, top_k, filters, "keyword", 0.0)

    async def topk_semantic_search(
        self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Semantic search by meaning.

        Use this for conceptual or paraphrased questions where the documents may use
        different words than the user, e.g. "why are customers churning?" or "how do I
        make deploys safer?". Don't use it to look up exact codes or IDs.

        Args:
            query: A natural-language question or description.
            top_k: How many results to return (default 5, max 20).
            filters: Optional metadata filter, e.g. {"source": "runbook"} or {"year": {"$gte": 2024}}.

        Returns:
            The best matching documents with id, score, metadata and text.
        """
        return await self._search(query, top_k, filters, "semantic", 1.0)

    async def topk_hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        keyword_weight: float = 0.5,
    ) -> dict[str, Any]:
        """Hybrid search: meaning and exact keywords ranked together in one query.

        The best default when a question mixes both, e.g. "why does checkout fail with
        E-4021?", or when you're unsure which search fits.

        Args:
            query: The question, including any exact terms.
            top_k: How many results to return (default 5, max 20).
            filters: Optional metadata filter, e.g. {"source": "runbook"} or {"year": {"$gte": 2024}}.
            keyword_weight: 0 to 1. Raise it when exact terms matter more than meaning (default 0.5).

        Returns:
            The best matching documents with id, score, metadata and text.
        """
        alpha = 1.0 - max(0.0, min(1.0, keyword_weight))
        return await self._search(query, top_k, filters, "hybrid", alpha)

    # ------------------------------------------------------------------ internals

    def _tool(self, func: Any) -> FunctionTool:
        tool = _DescribedTool(func)
        extra = [f"Filter operators: {FILTER_OPERATORS}. Keys are ANDed; use $or for alternatives."]
        if self.corpus_description:
            extra.append(f"The collection contains: {self.corpus_description}.")
        if self.metadata_fields:
            fields = "; ".join(f"{k}: {v}" for k, v in self.metadata_fields.items())
            extra.append(f"Metadata fields you can filter on: {fields}.")
        tool.description = f"{tool.description}\n\n" + "\n".join(extra)
        return tool

    async def _search(
        self, query: str, top_k: int | None, filters: dict[str, Any] | None, mode: str, alpha: float
    ) -> dict[str, Any]:
        if not query or not query.strip():
            return {"status": "error", "error": "query is empty"}
        try:
            meta_filter = translate_filter(filters)
        except ValueError as e:
            return {"status": "error", "error": str(e)}

        scores: dict[str, Any] = {}
        text_filter = None
        if mode == "keyword":
            scores["kw"] = fn.bm25_score()
            text_filter = match(query, field=TEXT_FIELD)
            rank = field("kw")
        else:
            scores["dense"] = fn.semantic_similarity(TEXT_FIELD, query)
            rank = field("dense")
            if mode == "hybrid":
                scores["kw"] = fn.bm25_score()
                text_filter = should(query, field=TEXT_FIELD)
                rank = field("dense") * alpha + field("kw") * (1.0 - alpha)

        q = select(TEXT_FIELD, METADATA_JSON_FIELD, **scores)
        if text_filter is not None:
            q = q.filter(text_filter)
        if meta_filter is not None:
            q = q.filter(meta_filter)
        q = q.sort(rank, asc=False).limit(max(1, min(top_k or self.default_top_k, _MAX_TOP_K)))

        rows = await self._query(q)
        results = []
        for r in rows:
            body = r.get(TEXT_FIELD) or ""
            if len(body) > self.max_chars_per_result:
                body = body[: self.max_chars_per_result] + "…"
            raw = r.get(METADATA_JSON_FIELD)
            dense, kw = r.get("dense") or 0.0, r.get("kw") or 0.0
            score = kw if mode == "keyword" else dense if mode == "semantic" else dense * alpha + kw * (1 - alpha)
            results.append(
                {
                    "id": r["_id"],
                    "score": round(float(score), 4),
                    "metadata": json.loads(raw) if raw else {},
                    "text": body,
                }
            )
        return {"status": "success", "mode": mode, "results": results}

    async def _query(self, q: Any) -> list[dict[str, Any]]:
        if self.partition and not self._partition_known:
            if not await partition_exists(self._client, self.collection_name, self.partition):
                return []
            self._partition_known = True
        read_kwargs = {"lsn": self._lsn} if self._lsn else {}
        return await read(lambda: self._collection().query(q, **read_kwargs), [], written=self._lsn is not None)

    def _collection(self) -> Any:
        return self._client.collection(self.collection_name, partition=self.partition)

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


class TopKMemoryToolset(BaseToolset):
    """``topk_remember`` / ``topk_recall`` / ``topk_forget`` tools for the current user.

    The agent decides what's worth keeping. Memories go to the current user's
    partition in a :class:`TopKMemoryService`, the same place ``load_memory`` and
    ``preload_memory`` read from, so explicit memories and saved sessions are
    searched together. One user's agent can never read another user's memories.

    Example:
        .. code-block:: python

            memory_service = TopKMemoryService(collection_name="adk_memory")
            root_agent = Agent(..., tools=[TopKMemoryToolset(memory_service)])
            runner = Runner(..., memory_service=memory_service)
    """

    def __init__(
        self,
        memory_service: TopKMemoryService,
        *,
        default_top_k: int = 5,
        tool_filter: ToolPredicate | list[str] | None = None,
        tool_name_prefix: str | None = None,
    ) -> None:
        super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)
        self.memory_service = memory_service
        self.default_top_k = default_top_k
        self._tools = [FunctionTool(self.topk_remember), FunctionTool(self.topk_recall), FunctionTool(self.topk_forget)]

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        return [t for t in self._tools if self._is_tool_selected(t, readonly_context)]

    async def topk_remember(
        self, memory: str, tool_context: ToolContext, category: str | None = None
    ) -> dict[str, Any]:
        """Save a fact about the user for future conversations.

        Use it when the user shares a lasting preference, constraint, personal detail,
        decision or goal (e.g. "I'm allergic to peanuts", "we deploy on Fridays", "my
        order number is 88412"). Write the memory as a short, self-contained sentence in
        the third person. Don't save small talk or one-off requests.

        Args:
            memory: The fact to remember, e.g. "The user is allergic to peanuts."
            category: Optional label such as "preference", "health", "work" or "account".

        Returns:
            The id of the saved memory.
        """
        memory_id = await self.memory_service.remember(
            app_name=tool_context.session.app_name,
            user_id=tool_context.user_id,
            text=memory,
            author="memory",
            metadata={"category": category} if category else None,
        )
        return {"status": "success", "memory_id": memory_id}

    async def topk_recall(self, query: str, tool_context: ToolContext, top_k: int | None = None) -> dict[str, Any]:
        """Look up what you know about the user from earlier conversations.

        Use it before answering anything that could depend on the user's preferences,
        history or personal details, and whenever the user asks what you remember.
        Matches both by meaning and by exact terms (names, numbers, codes).

        Args:
            query: What you need to know, e.g. "food allergies" or "order 88412".
            top_k: How many memories to return (default 5, max 20).

        Returns:
            Matching memories with id, text, when it was saved and its category.
        """
        response = await self.memory_service.search(
            app_name=tool_context.session.app_name,
            user_id=tool_context.user_id,
            query=query,
            top_k=max(1, min(top_k or self.default_top_k, _MAX_TOP_K)),
        )
        memories = [
            {
                "id": m.id,
                "text": "\n".join(p.text for p in (m.content.parts or []) if p.text),
                "author": m.author,
                "timestamp": m.timestamp,
                **({"category": m.custom_metadata["category"]} if "category" in m.custom_metadata else {}),
            }
            for m in response.memories
        ]
        return {"status": "success", "memories": memories}

    async def topk_forget(self, memory_id: str, tool_context: ToolContext) -> dict[str, Any]:
        """Delete a memory that is wrong or that the user asked you to forget.

        Args:
            memory_id: The id returned by topk_recall.

        Returns:
            Confirmation that the memory was deleted.
        """
        await self.memory_service.delete_memory(
            app_name=tool_context.session.app_name, user_id=tool_context.user_id, memory_id=memory_id
        )
        return {"status": "success", "deleted": memory_id}
