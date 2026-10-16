"""Shared helpers: client setup, partition names, metadata filters."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from topk_sdk import AsyncClient
from topk_sdk.error import CollectionNotFoundError, PartitionNotFoundError, QueryLsnTimeoutError
from topk_sdk.query import LogicalExpr, field, not_
from topk_sdk.query import all as topk_all
from topk_sdk.query import any as topk_any

REGIONS = ("aws-us-east-1-elastica", "aws-eu-central-1-monstera", "gcp-us-east4-aloe")
METADATA_FIELD = "metadata"
METADATA_JSON_FIELD = "metadata_json"

# TopK partition names: letters, digits, "_" and "-", not starting with "-", at most 128 chars.
_PARTITION_OK = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,127}$")
_PARTITION_BAD_CHARS = re.compile(r"[^A-Za-z0-9_-]")


T = TypeVar("T")

# After a read misses a partition, the same client keeps reporting it missing for a few
# seconds even once it has been written. Retry for up to ~12s when we know it exists.
_NOT_FOUND_BACKOFF = (0.25, 0.5, 1.0, 2.0, 4.0, 4.0)


async def read(call: Callable[[], Awaitable[T]], empty: T, *, written: bool) -> T:
    """Run a read. A missing collection or partition reads as ``empty``.

    ``written`` says this process has written to the partition, so a not-found error
    is the client's stale view and worth retrying.
    """
    delays = iter(_NOT_FOUND_BACKOFF if written else ())
    lsn_retried = False
    while True:
        try:
            return await call()
        except QueryLsnTimeoutError:
            # The write is durable but not indexed yet; one retry is usually enough.
            if lsn_retried:
                raise
            lsn_retried = True
        except (CollectionNotFoundError, PartitionNotFoundError):
            delay = next(delays, None)
            if delay is None:
                return empty
            await asyncio.sleep(delay)


async def partition_exists(client: AsyncClient, collection: str, partition: str) -> bool:
    """Check for a partition without reading it.

    A read of a missing partition makes the client report that partition missing for
    a few seconds after it's created, so a new user's first save would be invisible
    to the next search. Listing partitions doesn't have that effect.
    """
    try:
        async for found in client.collection(collection).list_partitions(prefix=partition):
            if found.name == partition:
                return True
    except CollectionNotFoundError:
        pass
    return False


def make_client(api_key: str | None, region: str | None) -> AsyncClient:
    api_key = api_key or os.environ.get("TOPK_API_KEY")
    region = region or os.environ.get("TOPK_REGION")
    if not api_key:
        msg = "Set TOPK_API_KEY or pass api_key=... (create a key at https://console.topk.io)"
        raise ValueError(msg)
    if not region:
        msg = f"Set TOPK_REGION or pass region=... (one of {', '.join(REGIONS)}; see https://docs.topk.io/regions)"
        raise ValueError(msg)
    return AsyncClient(api_key=api_key, region=region)


def partition_name(*parts: str) -> str:
    """Join ids into a valid TopK partition name.

    Readable when the ids are already safe (``"my_app--user-42"``). Otherwise the
    unsafe characters are replaced and a hash of the original ids is appended, so
    two different users never share a partition.
    """
    joined = "--".join(parts)
    if _PARTITION_OK.match(joined) and not any("--" in p for p in parts):
        return joined
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]
    readable = _PARTITION_BAD_CHARS.sub("_", joined).lstrip("-")[:100]
    return f"{readable}-{digest}" if readable else f"p-{digest}"


def filterable(value: Any) -> Any:
    """Keep the metadata values TopK can store and filter on; everything survives in metadata_json."""
    if isinstance(value, dict):
        out = {str(k): filterable(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v is not None}
    if isinstance(value, bool | str | int | float):
        return value
    if isinstance(value, list | tuple):
        items = [v for v in value if isinstance(v, str | int | float) and not isinstance(v, bool)]
        if items and len(items) == len(value) and len({type(v) for v in items}) == 1:
            return list(items)
    return None


_COMPARISONS = {
    "$eq": lambda f, v: f.eq(v),
    "$ne": lambda f, v: f.ne(v),
    "$gt": lambda f, v: f.gt(v),
    "$gte": lambda f, v: f.gte(v),
    "$lt": lambda f, v: f.lt(v),
    "$lte": lambda f, v: f.lte(v),
    "$in": lambda f, v: f.in_(list(v)),
    "$nin": lambda f, v: not_(f.in_(list(v))),
    "$contains": lambda f, v: f.contains(v),
    "$starts_with": lambda f, v: f.starts_with(v),
    "$exists": lambda f, v: f.is_not_null() if v else f.is_null(),
}
FILTER_OPERATORS = ", ".join(_COMPARISONS)


def translate_filter(filters: dict[str, Any] | LogicalExpr | None) -> LogicalExpr | None:
    """Translate a MongoDB-style metadata filter into a TopK expression.

    ``{"source": "runbook", "year": {"$gte": 2024}}`` ANDs its keys. A plain value
    means equality. ``$and``, ``$or`` and ``$not`` combine filters, and nested
    metadata is addressed with dots (``{"author.name": "Ada"}``). A native TopK
    expression is passed through unchanged.
    """
    if filters is None:
        return None
    if isinstance(filters, LogicalExpr):
        return filters
    if not isinstance(filters, dict):
        msg = f"filters must be an object like {{'source': 'runbook'}}, got {type(filters).__name__}"
        raise ValueError(msg)
    if not filters:
        return None
    clauses = [_translate_key(k, v) for k, v in filters.items()]
    return clauses[0] if len(clauses) == 1 else topk_all(clauses)


def _translate_key(key: str, value: Any) -> LogicalExpr:
    if key in ("$and", "$or"):
        if not isinstance(value, list) or not value:
            msg = f"{key} expects a non-empty list of filters"
            raise ValueError(msg)
        exprs = [_require(translate_filter(v)) for v in value]
        return topk_all(exprs) if key == "$and" else topk_any(exprs)
    if key == "$not":
        return not_(_require(translate_filter(value)))
    if key.startswith("$"):
        msg = f"Unsupported top-level filter operator {key!r}; use $and, $or or $not"
        raise ValueError(msg)

    f = field(f"{METADATA_FIELD}.{key}")
    if isinstance(value, dict) and value and all(k.startswith("$") for k in value):
        ops = []
        for op, operand in value.items():
            if op not in _COMPARISONS:
                msg = f"Unknown filter operator {op!r} on {key!r}; use one of {FILTER_OPERATORS}"
                raise ValueError(msg)
            ops.append(_COMPARISONS[op](f, operand))
        return ops[0] if len(ops) == 1 else topk_all(ops)
    return f.eq(value)


def _require(expr: LogicalExpr | None) -> LogicalExpr:
    if expr is None:
        msg = "Empty filter inside a logical operator"
        raise ValueError(msg)
    return expr
