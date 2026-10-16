"""Live tests for TopKMemoryService. Need TOPK_API_KEY and TOPK_REGION."""

from google.adk.events.event import Event
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types

from adk_topk import TopKMemoryService

APP = "test_app"


def _event(author: str, text: str) -> Event:
    role = "user" if author == "user" else "model"
    return Event(author=author, invocation_id="inv", content=types.Content(role=role, parts=[types.Part(text=text)]))


def _session(user_id: str, session_id: str, texts: list[tuple[str, str]]) -> Session:
    return Session(app_name=APP, user_id=user_id, id=session_id, events=[_event(a, t) for a, t in texts])


def _texts(response) -> list[str]:
    return [m.content.parts[0].text for m in response.memories]


async def test_session_roundtrip_hybrid_keyword_semantic(client, collection_name):
    ms = TopKMemoryService(collection_name, client=client)
    session = _session(
        "alice",
        "s1",
        [
            ("user", "I'm allergic to peanuts, so keep them out of any recipe."),
            ("chef", "Got it, no peanuts."),
            ("user", "My order number is 88412 and it hasn't arrived."),
            ("user", "I prefer dark mode in every app."),
        ],
    )
    await ms.add_session_to_memory(session)

    hybrid = await ms.search_memory(app_name=APP, user_id="alice", query="what foods can't I eat?")
    assert "peanuts" in _texts(hybrid)[0]
    entry = hybrid.memories[0]
    assert entry.author == "user"
    assert entry.timestamp and entry.timestamp.startswith("20")
    assert entry.custom_metadata["session_id"] == "s1"
    assert entry.id == session.events[0].id

    keyword = await ms.search(app_name=APP, user_id="alice", query="88412", search_mode="keyword")
    assert len(keyword.memories) == 1 and "88412" in _texts(keyword)[0]

    semantic = await ms.search(
        app_name=APP, user_id="alice", query="theme for the user interface", search_mode="semantic"
    )
    assert "dark mode" in _texts(semantic)[0]

    limited = await ms.search(app_name=APP, user_id="alice", query="peanuts", top_k=2)
    assert len(limited.memories) == 2


async def test_users_and_apps_are_isolated(client, collection_name):
    ms = TopKMemoryService(collection_name, client=client)
    await ms.add_session_to_memory(_session("alice", "s1", [("user", "My locker code is 4417.")]))
    await ms.add_session_to_memory(_session("bob", "s2", [("user", "I drive a red bicycle.")]))

    assert _texts(await ms.search_memory(app_name=APP, user_id="bob", query="locker code 4417")) == [
        "I drive a red bicycle."
    ]
    # A user that never wrote anything has no partition yet: empty, not an error.
    assert (await ms.search_memory(app_name=APP, user_id="carol", query="locker")).memories == []
    assert (await ms.search_memory(app_name="other_app", user_id="alice", query="locker")).memories == []
    # Unsafe ids still map to valid, distinct partitions.
    weird = "user@example.com/with spaces"
    await ms.add_session_to_memory(Session(app_name="app.v2", user_id=weird, id="s", events=[_event("user", "hello")]))
    assert _texts(await ms.search_memory(app_name="app.v2", user_id=weird, query="hello")) == ["hello"]


async def test_resave_is_idempotent_and_delete_sticks(client, collection_name):
    ms = TopKMemoryService(collection_name, client=client)
    session = _session("alice", "s1", [("user", "I like hiking in the Alps."), ("user", "My cat is called Miso.")])
    await ms.add_session_to_memory(session)
    session.events.append(_event("user", "I also like sailing."))
    await ms.add_session_to_memory(session)  # saved again as the session grows

    everything = await ms.search(app_name=APP, user_id="alice", query="hobbies cat hiking sailing", top_k=10)
    assert sorted(_texts(everything)) == sorted(e.content.parts[0].text for e in session.events)

    cat_id = session.events[1].id
    await ms.delete_memory(app_name=APP, user_id="alice", memory_id=cat_id)
    await ms.add_session_to_memory(session)  # must not bring the deleted event back
    after = await ms.search(app_name=APP, user_id="alice", query="cat Miso", top_k=10)
    assert cat_id not in [m.id for m in after.memories]
    assert len(after.memories) == 2


async def test_add_memory_remember_filters_and_clear(client, collection_name):
    ms = TopKMemoryService(collection_name, client=client)
    await ms.add_memory(
        app_name=APP,
        user_id="alice",
        memories=[
            MemoryEntry(
                id="m1",
                content=types.Content(parts=[types.Part(text="The user works night shifts.")]),
                custom_metadata={"category": "work", "tags": ["schedule"]},
            )
        ],
    )
    mem_id = await ms.remember(
        app_name=APP, user_id="alice", text="The user is vegetarian.", metadata={"category": "food"}
    )
    assert mem_id.startswith("mem-")

    food = await ms.search(app_name=APP, user_id="alice", query="user", filters={"category": "food"})
    assert _texts(food) == ["The user is vegetarian."]
    work = await ms.search(app_name=APP, user_id="alice", query="user", filters={"tags": {"$contains": "schedule"}})
    assert [m.id for m in work.memories] == ["m1"]
    assert work.memories[0].custom_metadata == {"category": "work", "tags": ["schedule"]}

    await ms.clear(app_name=APP, user_id="alice")
    assert (await ms.search_memory(app_name=APP, user_id="alice", query="vegetarian")).memories == []


async def test_long_events_and_non_text_events(client, collection_name):
    ms = TopKMemoryService(collection_name, client=client)
    long_text = ("The quarterly report covers revenue growth in every region. " * 200) + "Codename ZEPHYR-9."
    assert len(long_text) > 10_000
    tool_call = Event(
        author="agent",
        invocation_id="inv",
        content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="f", args={}))]),
    )
    await ms.add_session_to_memory(_session("alice", "s1", [("user", long_text)]))
    await ms.add_events_to_memory(app_name=APP, user_id="alice", events=[tool_call], session_id="s1")
    hits = await ms.search(app_name=APP, user_id="alice", query="ZEPHYR-9", search_mode="keyword")
    assert len(hits.memories) == 1 and hits.memories[0].content.parts[0].text == long_text
    semantic = await ms.search(app_name=APP, user_id="alice", query="regional revenue growth", search_mode="semantic")
    assert len(semantic.memories) == 1
