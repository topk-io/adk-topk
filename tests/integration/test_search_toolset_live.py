"""Live tests for TopKSearchToolset. Need TOPK_API_KEY and TOPK_REGION."""

from adk_topk import TopKSearchToolset

DOCS = [
    ("Error E-4021 means the payment card was declined by the issuing bank.", {"source": "runbook", "year": 2025}),
    (
        "If checkout keeps failing, ask the customer to try another card or contact their bank.",
        {"source": "runbook", "year": 2023},
    ),
    (
        "Deploy safety: roll out behind feature flags and watch error rates before widening.",
        {"source": "docs", "year": 2025},
    ),
    (
        "The rotate_api_key function issues a new key and revokes the old one after 24 hours.",
        {"source": "docs", "year": 2024},
    ),
]


async def _toolset(client, collection_name, **kwargs) -> TopKSearchToolset:
    ts = TopKSearchToolset(collection_name, client=client, **kwargs)
    await ts.add_texts([t for t, _ in DOCS], metadatas=[m for _, m in DOCS], ids=[f"d{i}" for i in range(len(DOCS))])
    return ts


async def test_modes(client, collection_name):
    ts = await _toolset(client, collection_name)

    kw = await ts.topk_keyword_search("rotate_api_key")
    assert kw["status"] == "success" and [r["id"] for r in kw["results"]] == ["d3"]
    assert (await ts.topk_keyword_search("ZZ-9999"))["results"] == []

    sem = await ts.topk_semantic_search("how do I release changes without breaking production?")
    assert sem["results"][0]["id"] == "d2"
    assert sem["results"][0]["metadata"] == {"source": "docs", "year": 2025}

    hyb = await ts.topk_hybrid_search("why does checkout fail with E-4021?", top_k=2)
    # Both checkout documents, ranked together: one by the exact code, one by meaning.
    assert sorted(r["id"] for r in hyb["results"]) == ["d0", "d1"]
    kw_heavy = await ts.topk_hybrid_search("E-4021", keyword_weight=1.0)
    assert kw_heavy["results"][0]["id"] == "d0"


async def test_filters_partitions_and_errors(client, collection_name):
    ts = await _toolset(client, collection_name)
    runbooks = await ts.topk_semantic_search("card declined", filters={"source": "runbook", "year": {"$gte": 2024}})
    assert [r["id"] for r in runbooks["results"]] == ["d0"]
    either = await ts.topk_semantic_search("anything", top_k=10, filters={"$or": [{"year": 2023}, {"year": 2024}]})
    assert sorted(r["id"] for r in either["results"]) == ["d1", "d3"]
    bad = await ts.topk_semantic_search("x", filters={"year": {"$between": [1, 2]}})
    assert bad["status"] == "error" and "$between" in bad["error"]

    tenant = TopKSearchToolset(collection_name, client=client, partition="tenant-b")
    assert (await tenant.topk_hybrid_search("E-4021"))["results"] == []  # partition doesn't exist yet
    await tenant.add_texts(["Tenant B only: error E-4021 is a test code."], ids=["b1"])
    assert [r["id"] for r in (await tenant.topk_keyword_search("E-4021"))["results"]] == ["b1"]

    missing = TopKSearchToolset(collection_name + "_missing", client=client)
    assert (await missing.topk_hybrid_search("anything"))["results"] == []

    await ts.delete(["d0"])
    assert "d0" not in [r["id"] for r in (await ts.topk_keyword_search("E-4021"))["results"]]


async def test_result_truncation(client, collection_name):
    ts = TopKSearchToolset(collection_name, client=client, max_chars_per_result=40)
    await ts.add_texts(["alpha " * 100])
    res = await ts.topk_keyword_search("alpha")
    assert len(res["results"][0]["text"]) == 41 and res["results"][0]["text"].endswith("…")
