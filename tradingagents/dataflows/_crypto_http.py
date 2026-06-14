"""Shared HTTP helpers for the crypto dataflows.

Crypto reference APIs (CoinGecko, alternative.me) are aggressively rate-limited,
so JSON responses are cached to ``data_cache_dir/crypto`` with a short TTL. All
helpers raise on failure — callers catch and degrade to a placeholder string so
one dead source never sinks a whole analysis.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import requests

from .config import get_config

_UA = {"User-Agent": "CortexAIO-TradingAgents/1.0 (+crypto-analysis)"}


def _cache_path(key: str) -> str:
    cache_dir = os.path.join(get_config()["data_cache_dir"], "crypto")
    os.makedirs(cache_dir, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)
    return os.path.join(cache_dir, f"{safe}.json")


def get_json(
    url: str,
    params: Optional[dict] = None,
    *,
    cache_key: Optional[str] = None,
    ttl_seconds: int = 6 * 3600,
    timeout: int = 15,
) -> Any:
    """GET *url* and parse JSON, optionally serving/refreshing a TTL cache.

    Raises on transport/HTTP/JSON errors. A fresh cache hit (younger than
    ``ttl_seconds``) is returned without a network call.
    """
    if cache_key:
        path = _cache_path(cache_key)
        try:
            if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception:
            pass  # corrupt/locked cache — fall through to network

    resp = requests.get(url, params=params, headers=_UA, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if cache_key:
        try:
            with open(_cache_path(cache_key), "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception:
            pass  # cache write is best-effort
    return data


def fmt_usd(value: Optional[float]) -> str:
    """Compact USD formatting: 1.23T / 45.6B / 789.0M / 1,234.56."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:.2f}T"
    if a >= 1e9:
        return f"${v / 1e9:.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:.2f}M"
    if a >= 1e3:
        return f"${v:,.0f}"
    return f"${v:,.2f}"


def fmt_num(value: Optional[float]) -> str:
    """Compact count formatting: 1.23T / 45.6B / 789.0M / 1,234."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    a = abs(v)
    if a >= 1e12:
        return f"{v / 1e12:.2f}T"
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    return f"{v:,.0f}"
