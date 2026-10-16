import os
import uuid

import pytest
import pytest_asyncio
from topk_sdk import AsyncClient

pytestmark = pytest.mark.integration


def pytest_collection_modifyitems(items):
    if os.environ.get("TOPK_API_KEY") and os.environ.get("TOPK_REGION"):
        return
    skip = pytest.mark.skip(reason="needs TOPK_API_KEY and TOPK_REGION")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture
def client():
    return AsyncClient(api_key=os.environ["TOPK_API_KEY"], region=os.environ["TOPK_REGION"])


@pytest_asyncio.fixture
async def collection_name(client):
    name = f"adk_test_{uuid.uuid4().hex[:10]}"
    yield name
    try:
        await client.collections().delete(name)
    except Exception:
        pass
