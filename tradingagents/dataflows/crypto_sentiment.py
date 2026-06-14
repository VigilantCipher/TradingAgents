"""Crypto sentiment sources: the Fear & Greed index + crypto-subreddit chatter.

Both keyless. ``alternative.me`` publishes the canonical Crypto Fear & Greed
index; Reddit's public JSON gives community discussion. Reused by the
crypto-sentiment analyst, which pre-fetches these into its prompt (mirroring the
equity sentiment analyst's no-tool-call pattern).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ._crypto_http import get_json
from .crypto_symbols import crypto_base
from .reddit import fetch_reddit_posts

# General crypto subreddits + the highest-signal per-asset community.
_GENERAL_SUBS = ("CryptoCurrency", "CryptoMarkets")
_ASSET_SUBS = {
    "BTC": ("Bitcoin",), "ETH": ("ethereum", "ethtrader"), "SOL": ("solana",),
    "XRP": ("Ripple",), "DOGE": ("dogecoin",), "ADA": ("cardano",),
    "AVAX": ("Avax",), "LINK": ("Chainlink",), "MATIC": ("0xPolygon",),
    "DOT": ("Polkadot",), "ATOM": ("cosmosnetwork",),
}


def fetch_fear_greed(limit: int = 3) -> str:
    """Crypto Fear & Greed index (current + recent), from alternative.me."""
    try:
        data = get_json(
            "https://api.alternative.me/fng/", params={"limit": str(limit)},
            cache_key="fng", ttl_seconds=3600,
        ).get("data", [])
    except Exception as exc:
        return f"<Fear & Greed index unavailable: {exc}>"
    if not data:
        return "<Fear & Greed index returned no data>"

    def _row(d: dict) -> str:
        ts = d.get("timestamp")
        when = (
            datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
            if ts else "?"
        )
        return f"{when}: {d.get('value', '?')}/100 ({d.get('value_classification', '?')})"

    current = data[0]
    lines = [f"Current: {current.get('value', '?')}/100 — {current.get('value_classification', '?')}"]
    if len(data) > 1:
        lines.append("Recent: " + "  |  ".join(_row(d) for d in data[:limit]))
    return "\n".join(lines)


def fetch_crypto_reddit(symbol: str, limit_per_sub: int = 5) -> str:
    """Recent Reddit discussion for a crypto asset across general + asset subs."""
    base = crypto_base(symbol)
    subs = _ASSET_SUBS.get(base, ()) + _GENERAL_SUBS
    # Search by the base ticker (e.g. "BTC") — "BTC-USD" matches nothing on Reddit.
    return fetch_reddit_posts(base, subreddits=subs, limit_per_sub=limit_per_sub)
