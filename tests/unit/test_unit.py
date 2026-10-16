"""Offline tests: no network, no API keys."""

import json

import pytest
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
from topk_sdk.error import PartitionNotFoundError

from adk_topk import TopKMemoryService, TopKMemoryToolset, TopKSearchToolset
from adk_topk._common import make_client, partition_name, translate_filter


class FakePartitions:
    def __init__(self, names):
        self._names = iter(names)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return type("P", (), {"name": next(self._names)})
        except StopIteration:
            raise StopAsyncIteration from None


class FakeCollection:
    def __init__(self, store, partition):
        self.store, self.partition = store, partition

    async def upsert(self, docs):
        self.store.setdefault(self.partition, {}).update({d["_id"]: d for d in docs})
        return "1"

    async def get(self, ids, fields=None, **_):
        if self.partition not in self.store:
            raise PartitionNotFoundError("missing")
        return {i: self.store[self.partition][i] for i in ids if i in self.store[self.partition]}

    async def query(self, q, **_):
        return []

    def list_partitions(self, prefix=None):
        return FakePartitions([p for p in self.store if p and p.startswith(prefix or "")])


class FakeClient:
    def __init__(self):
        self.store = {}
        self.created = []

    def collection(self, name, partition=None):
        return FakeCollection(self.store, partition)

    def collections(self):
        client = self

        class Collections:
            async def create(self, name, schema):
                client.created.append((name, schema))

        return Collections()


def _event(text=None, partial=False, call=False):
    part = types.Part(function_call=types.FunctionCall(name="f", args={})) if call else types.Part(text=text)
    return Event(author="user", invocation_id="i", partial=partial, content=types.Content(role="user", parts=[part]))


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("app", "user-1"), "app--user-1"),
        (("my_app", "U_42"), "my_app--U_42"),
    ],
)
def test_partition_name_readable(parts, expected):
    assert partition_name(*parts) == expected


def test_partition_name_unsafe_ids_are_valid_and_distinct():
    names = {
        partition_name("app.v2", "user@example.com"),
        partition_name("app", "v2--user@example.com"),
        partition_name("app--v2", "user"),
        partition_name("app", "v2--user"),
        partition_name("app", "x" * 300),
        partition_name("app", "x" * 301),
        partition_name("-", "-"),
    }
    assert len(names) == 7
    for name in names:
        assert len(name) <= 128 and not name.startswith("-")
        assert all(c.isalnum() or c in "_-" for c in name)


def test_translate_filter():
    assert translate_filter(None) is None and translate_filter({}) is None
    assert translate_filter({"source": "runbook", "year": {"$gte": 2024, "$lt": 2026}}) is not None
    assert translate_filter({"$or": [{"a": 1}, {"$not": {"b": {"$in": [1, 2]}}}]}) is not None
    with pytest.raises(ValueError, match=r"\$between"):
        translate_filter({"year": {"$between": [1, 2]}})
    with pytest.raises(ValueError, match="non-empty list"):
        translate_filter({"$or": []})
    with pytest.raises(ValueError, match="must be an object"):
        translate_filter("source = runbook")


def test_make_client_requires_key_and_region(monkeypatch):
    monkeypatch.delenv("TOPK_API_KEY", raising=False)
    monkeypatch.delenv("TOPK_REGION", raising=False)
    with pytest.raises(ValueError, match="TOPK_API_KEY"):
        make_client(None, None)
    with pytest.raises(ValueError, match="TOPK_REGION"):
        make_client("key", None)


def test_memory_service_validates_args():
    with pytest.raises(ValueError, match="search_mode"):
        TopKMemoryService(client=FakeClient(), search_mode="vector")
    with pytest.raises(ValueError, match="hybrid_alpha"):
        TopKMemoryService(client=FakeClient(), hybrid_alpha=2)


async def test_add_session_skips_partial_tool_and_known_events():
    client = FakeClient()
    ms = TopKMemoryService("mem", client=client)
    keep, partial, call, empty = _event("keep me"), _event("par", partial=True), _event(call=True), _event("  ")
    session = Session(app_name="app", user_id="u", id="s", events=[keep, partial, call, empty])
    await ms.add_session_to_memory(session)
    stored = client.store["app--u"]
    assert list(stored) == [keep.id]
    assert stored[keep.id]["session_id"] == "s" and stored[keep.id]["kind"] == "event"
    assert client.created[0][0] == "mem"

    # A tombstone is never overwritten by saving the session again.
    await ms.delete_memory(app_name="app", user_id="u", memory_id=keep.id)
    await ms.add_session_to_memory(session)
    assert stored[keep.id] == {"_id": keep.id, "deleted": True}


async def test_search_on_unknown_user_returns_empty_without_reading():
    ms = TopKMemoryService("mem", client=FakeClient())
    assert (await ms.search_memory(app_name="app", user_id="nobody", query="hi")).memories == []
    assert (await ms.search_memory(app_name="app", user_id="nobody", query="  ")).memories == []


async def test_search_tools_declarations_carry_context():
    ts = TopKSearchToolset(
        "docs", client=FakeClient(), corpus_description="support runbooks", metadata_fields={"source": "'runbook'"}
    )
    tools = {t.name: t for t in await ts.get_tools()}
    assert set(tools) == {"topk_keyword_search", "topk_semantic_search", "topk_hybrid_search"}
    decl = tools["topk_keyword_search"]._get_declaration()
    assert decl.description.startswith("Exact keyword search (BM25).")
    assert "support runbooks" in decl.description and "source: 'runbook'" in decl.description
    assert "$contains" in decl.description

    filtered = TopKSearchToolset("docs", client=FakeClient(), tool_filter=["topk_hybrid_search"])
    assert [t.name for t in await filtered.get_tools()] == ["topk_hybrid_search"]


async def test_search_tool_input_errors_are_returned_not_raised():
    ts = TopKSearchToolset("docs", client=FakeClient())
    assert (await ts.topk_hybrid_search(""))["status"] == "error"
    bad = await ts.topk_semantic_search("x", filters={"a": {"$regex": "y"}})
    assert bad["status"] == "error" and "$regex" in bad["error"]


async def test_memory_tools_hide_tool_context_from_model():
    tools = await TopKMemoryToolset(TopKMemoryService(client=FakeClient())).get_tools()
    for tool in tools:
        schema = tool._get_declaration().model_dump(mode="json", exclude_none=True)
        assert "tool_context" not in json.dumps(schema)
