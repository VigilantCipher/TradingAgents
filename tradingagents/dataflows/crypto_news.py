"""Crypto news headlines from keyless RSS feeds (+ optional CryptoPanic token).

CoinDesk / CoinTelegraph / Bitcoin Magazine RSS need no key. If a free
``CRYPTOPANIC_TOKEN`` is set, its currency-filtered feed is added for
asset-specific coverage. Always returns a string (placeholder on total failure)
so the news analyst never sees an exception.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Annotated, Optional

import requests

from ._crypto_http import get_json
from .crypto_symbols import crypto_base

_HEADERS = {"User-Agent": "CortexAIO-TradingAgents/1.0 (+crypto-news)"}
_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/feed",
}
_MAX_ITEMS = 25


def _parse_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except Exception:
        return None


def _fetch_feed(name: str, url: str) -> list[dict]:
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "source": name,
            "title": title,
            "link": (item.findtext("link") or "").strip(),
            "published": _parse_date(item.findtext("pubDate")),
            "summary": (item.findtext("description") or "").strip()[:300],
        })
    return items


def _fetch_cryptopanic(base: str) -> list[dict]:
    token = os.getenv("CRYPTOPANIC_TOKEN")
    if not token:
        return []
    data = get_json(
        "https://cryptopanic.com/api/v1/posts/",
        params={"auth_token": token, "currencies": base, "public": "true"},
        cache_key=f"cryptopanic_{base}", ttl_seconds=1800,
    )
    out = []
    for p in data.get("results", [])[:_MAX_ITEMS]:
        out.append({
            "source": "CryptoPanic",
            "title": (p.get("title") or "").strip(),
            "link": p.get("url", ""),
            "published": _parse_date(p.get("published_at")),
            "summary": "",
        })
    return out


def get_crypto_news(
    query: Annotated[str, "search focus, e.g. BTC-USD or 'bitcoin etf'"],
    start_date: Annotated[str, "start date yyyy-mm-dd"] = None,
    end_date: Annotated[str, "end date yyyy-mm-dd"] = None,
) -> str:
    """Recent crypto headlines, filtered to *query* terms and date window."""
    base = crypto_base(query) if query else ""
    terms = {t.lower() for t in (base, query or "") if t}
    terms.discard("")
    # Common name aliases so a BTC query still catches "bitcoin" headlines.
    alias = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "DOGE": "dogecoin"}
    if base in alias:
        terms.add(alias[base])

    items: list[dict] = []
    for name, url in _FEEDS.items():
        try:
            items.extend(_fetch_feed(name, url))
        except Exception:
            continue
    try:
        items.extend(_fetch_cryptopanic(base))
    except Exception:
        pass

    if not items:
        return f"No crypto news available right now for '{query}' (RSS sources unreachable)."

    start_dt = _safe_dt(start_date)
    end_dt = _safe_dt(end_date)

    def _in_window(dt: Optional[datetime]) -> bool:
        if dt is None:
            return True
        if start_dt and dt < start_dt:
            return False
        if end_dt and dt > end_dt:
            return False
        return True

    # Asset-specific matches first; fall back to general crypto-market headlines
    # so the analyst always has macro context even when nothing names the coin.
    matched = [
        it for it in items
        if _in_window(it["published"]) and (not terms or any(t in (it["title"] + " " + it["summary"]).lower() for t in terms))
    ]
    pool = matched if matched else [it for it in items if _in_window(it["published"])]
    pool.sort(key=lambda it: it["published"] or datetime.min, reverse=True)
    pool = pool[:_MAX_ITEMS]

    scope = base or query or "crypto"
    header = (
        f"# Crypto News — focus: {scope}\n"
        f"# Sources: {', '.join(_FEEDS)}"
        + (" + CryptoPanic" if os.getenv("CRYPTOPANIC_TOKEN") else "")
        + (f"\n# Note: no headline specifically named {scope}; showing recent crypto-market news."
           if not matched else "")
        + "\n"
    )
    lines = []
    for it in pool:
        when = it["published"].strftime("%Y-%m-%d") if it["published"] else "n/a"
        lines.append(f"- [{when}] ({it['source']}) {it['title']}")
    return header + "\n" + "\n".join(lines)


def _safe_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None
