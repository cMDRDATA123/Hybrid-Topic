#!/usr/bin/env python3
"""Refresh account-visible model IDs in the replaceable official catalog.

This command only reads provider model-list endpoints. It never invents
context, output or tokenizer values; those fields remain in the JSON until an
official capability page has been checked and the entry is updated explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "hybrid_topic/model_catalog.json"
KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def merge_inventory(catalog: dict, provider: str, model_ids: list[str], *, refreshed_at: str) -> dict:
    """Merge a provider's returned IDs while preserving verified capabilities."""
    providers = catalog.setdefault("providers", {})
    if provider not in providers or not isinstance(providers[provider], dict):
        raise ValueError(f"provider is missing from catalog: {provider}")
    provider_entry = providers[provider]
    models = provider_entry.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"catalog models must be an object for {provider}")
    visible = {model_id.strip() for model_id in model_ids if isinstance(model_id, str) and model_id.strip()}
    if not visible:
        raise ValueError(f"provider returned no model IDs: {provider}")
    for model_id in visible:
        entry = models.setdefault(model_id, {"capability_status": "unverified", "modality": "unknown"})
        if not isinstance(entry, dict):
            entry = {}
            models[model_id] = entry
        entry["account_visible"] = True
        entry["last_seen_in_inventory"] = refreshed_at
    for model_id, entry in models.items():
        if isinstance(entry, dict) and model_id not in visible:
            entry["account_visible"] = False
    provider_entry["last_inventory_refresh"] = refreshed_at
    return catalog


def _request_headers(provider: str, api_key: str) -> dict[str, str]:
    if provider == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {api_key}"}


def fetch_inventory_ids(
    provider: str,
    endpoint: str,
    *,
    api_key: str,
    opener: Callable[..., object] = urlopen,
) -> list[str]:
    ids = set()
    seen_cursors = set()
    url = endpoint
    while True:
        request = Request(url, headers={**_request_headers(provider, api_key), "Accept": "application/json"})
        try:
            with opener(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not refresh {provider} model inventory from {endpoint}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected model-list response for {provider}")
        data = payload.get("data", payload.get("models"))
        has_more = payload.get("has_more", False)
        if not isinstance(data, list) or not isinstance(has_more, bool):
            raise RuntimeError(f"unexpected model-list response for {provider}")
        for item in data:
            model_id = item if isinstance(item, str) else (
                item.get("id", item.get("name")) if isinstance(item, Mapping) else None)
            if not isinstance(model_id, str) or not model_id.strip():
                raise RuntimeError(f"invalid model ID in {provider} inventory")
            ids.add(model_id.strip())
        if not has_more:
            return sorted(ids)
        cursor = payload.get("last_id")
        if not isinstance(cursor, str) or not cursor.strip() or cursor in seen_cursors:
            raise RuntimeError(f"missing or repeated pagination cursor for {provider}")
        seen_cursors.add(cursor)
        parts = urlsplit(endpoint)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["after_id"] = cursor
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--provider", choices=sorted(KEY_ENV), required=True)
    parser.add_argument("--api-key", help="provider key; defaults to the provider environment variable")
    parser.add_argument("--dry-run", action="store_true", help="fetch and report IDs without writing the catalog")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    provider_entry = catalog.get("providers", {}).get(args.provider, {})
    endpoint = provider_entry.get("runtime_inventory_endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise SystemExit(f"no runtime inventory endpoint configured for {args.provider}")
    api_key = args.api_key or os.environ.get(KEY_ENV[args.provider])
    if not api_key:
        raise SystemExit(f"set {KEY_ENV[args.provider]} or pass --api-key")
    refreshed_at = datetime.now(timezone.utc).isoformat()
    ids = fetch_inventory_ids(args.provider, endpoint, api_key=api_key)
    print(f"{args.provider}: {len(ids)} account-visible model IDs")
    if args.dry_run:
        return 0
    _write_atomic(args.catalog, merge_inventory(catalog, args.provider, ids, refreshed_at=refreshed_at))
    print(f"updated {args.catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
