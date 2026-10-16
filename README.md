# adk-topk

[TopK](https://topk.io) search tools and long-term memory for
[ADK](https://adk.dev) (Agent Development Kit) agents.

TopK is a retrieval engine where keyword search, vector search, filters and
embedding inference run in one query. It embeds text on the server, so you
don't need an embedding model or key, and its hybrid search ranks semantic
similarity and BM25 together in one query over a single index.

| Component | What it does |
|---|---|
| `TopKMemoryService` | ADK `BaseMemoryService`. One isolated TopK partition per app and user, hybrid search, works with `load_memory` and `preload_memory`. |
| `TopKMemoryToolset` | `topk_remember`, `topk_recall`, `topk_forget` for the current user. |
| `TopKSearchToolset` | `topk_keyword_search` (BM25), `topk_semantic_search`, `topk_hybrid_search` over a collection, with metadata filters. |

## Install

```bash
pip install adk-topk
export TOPK_API_KEY="your-api-key"          # https://console.topk.io
export TOPK_REGION="aws-us-east-1-elastica" # https://docs.topk.io/regions
```

## Memory

```python
from adk_topk import TopKMemoryService, TopKMemoryToolset
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import preload_memory

memory_service = TopKMemoryService(collection_name="adk_memory")


async def save_session_to_memory(callback_context):
    await callback_context.add_session_to_memory()


root_agent = Agent(
    model="gemini-flash-latest",
    name="assistant",
    instruction="Save lasting facts about the user with topk_remember; check topk_recall before answering.",
    tools=[TopKMemoryToolset(memory_service), preload_memory],
    after_agent_callback=save_session_to_memory,
)
runner = Runner(
    agent=root_agent,
    app_name="my_app",
    session_service=InMemorySessionService(),
    memory_service=memory_service,
)
```

- Every `(app_name, user_id)` pair gets its own TopK partition. Ids that aren't
  valid partition names are mapped to a readable name plus a hash.
- `add_session_to_memory` can be called after every turn: it only writes events
  it hasn't stored yet.
- `delete_memory` leaves a tombstone, so saving the session again doesn't bring
  a deleted event back. `clear(app_name=..., user_id=...)` removes everything
  for a user.
- `search(app_name=..., user_id=..., query=..., top_k=..., search_mode=..., filters=...)`
  is `search_memory` with per-call options. `search_mode` is `"hybrid"`
  (default), `"semantic"` or `"keyword"`.

## Search

```python
import asyncio

from adk_topk import TopKSearchToolset
from google.adk.agents import Agent

docs = TopKSearchToolset(
    collection_name="support_docs",
    corpus_description="product docs and incident runbooks",
    metadata_fields={"source": "'runbook' or 'docs'", "year": "integer"},
)
asyncio.run(
    docs.add_texts(
        ["Error E-4021 means the card was declined by the issuing bank."],
        metadatas=[{"source": "runbook", "year": 2025}],
    )
)

root_agent = Agent(model="gemini-flash-latest", name="support", instruction="...", tools=[docs])
```

Each tool's description tells the model when to use it: keyword search for
exact codes, IDs and names, semantic search for paraphrased questions, hybrid
search when a question mixes both. `corpus_description` and `metadata_fields`
are appended to every tool description. Filters use MongoDB-style operators:
`$eq $ne $gt $gte $lt $lte $in $nin $contains $starts_with $exists`, plus
`$and $or $not`. Pass `partition=` to scope the toolset to one tenant.

## Storage layout

Each document has `text` (with `semantic_index()`, which also enables BM25),
`metadata` (a struct used for filters) and `metadata_json` (the original
metadata, returned in results). Memory documents also store `author`,
`session_id`, `timestamp` and `kind` (`"event"` or `"memory"`).

## Tests

```bash
uv venv && uv pip install -e . pytest pytest-asyncio
pytest tests/unit                       # offline
TOPK_API_KEY=... TOPK_REGION=... pytest # adds live tests; collections are deleted afterwards
```

The live tests include an end-to-end run through ADK's `Runner` with a scripted
model, so they don't need an LLM key. Tested with google-adk 1.20.0 and 2.12.0.

## License

MIT
