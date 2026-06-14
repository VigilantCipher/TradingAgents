"""Energy supply/inventory data from the EIA (key-gated).

EIA v2 needs a free API key (``EIA_API_KEY``). When set and the instrument is an
energy future, this reports the latest weekly US inventories (crude / gasoline /
distillate stocks, or natural-gas working storage) with the week-over-week
change. Without a key, or for non-energy futures, it degrades to a clear note —
positioning/macro still carry the analysis.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated, Optional

from ._crypto_http import fmt_num, get_json
from .commodity_symbols import category, commodity_name, to_yfinance

# Energy =F symbol -> (label, EIA v2 route, series id, units). Brent uses US
# crude stocks as a proxy (no direct EIA Brent inventory series).
_EIA_SERIES = {
    "CL=F": ("US crude oil ending stocks", "petroleum/stoc/wstk", "WCESTUS1", "thousand bbl"),
    "BZ=F": ("US crude oil ending stocks (proxy)", "petroleum/stoc/wstk", "WCESTUS1", "thousand bbl"),
    "RB=F": ("US total gasoline stocks", "petroleum/stoc/wstk", "WGTSTUS1", "thousand bbl"),
    "HO=F": ("US distillate (heating oil) stocks", "petroleum/stoc/wstk", "WDISTUS1", "thousand bbl"),
    "NG=F": ("US working natural gas in storage (Lower 48)", "natural-gas/stor/wkly", "NW2_EPG0_SWO_R48_BCF", "Bcf"),
}


def _eia_latest(route: str, series_id: str, key: str):
    """Return the two most recent {period, value} points for an EIA series."""
    data = get_json(
        f"https://api.eia.gov/v2/{route}/data/",
        params={
            "api_key": key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": series_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": "5",
        },
        cache_key=f"eia_{series_id}", ttl_seconds=12 * 3600,
    )
    return (data.get("response", {}) or {}).get("data", [])


def get_commodity_supply(
    ticker: Annotated[str, "futures ticker, e.g. CL=F"],
    curr_date: Annotated[str, "current date (unused; EIA publishes weekly)"] = None,
) -> str:
    """Latest EIA energy inventories for energy futures (key-gated)."""
    name = commodity_name(ticker) or ticker
    sym = to_yfinance(ticker)
    cat = category(ticker)
    header = f"# Supply / Inventories — {name}\n# Source: EIA (US Energy Information Administration)\n"

    if cat != "energy":
        return header + (
            f"\nEIA inventory data is energy-specific and does not apply to {name} ({cat}). "
            f"Assess supply via COT positioning and the production/seasonal cycle."
        )

    key = os.getenv("EIA_API_KEY")
    if not key:
        return header + (
            "\nEIA_API_KEY not set — energy inventory data unavailable. "
            "Set a free key in services/TradingAgents/.env to enable crude/gasoline/distillate "
            "stocks and natural-gas storage. Using COT positioning + macro for now."
        )

    spec = _EIA_SERIES.get(sym)
    if not spec:
        return header + f"\nNo EIA inventory series mapped for {name}."

    label, route, series_id, units = spec
    try:
        pts = _eia_latest(route, series_id, key)
    except Exception as exc:
        return header + f"\nEIA request failed: {exc}"
    if not pts:
        return header + f"\nEIA returned no data for {label}."

    try:
        latest_v = float(pts[0]["value"])
        latest_p = pts[0]["period"]
        prev_v = float(pts[1]["value"]) if len(pts) > 1 else None
    except (KeyError, TypeError, ValueError) as exc:
        return header + f"\nEIA data parse error: {exc}"

    wow = ""
    if prev_v is not None:
        delta = latest_v - prev_v
        wow = f" (WoW {delta:+,.0f} {units}, {'build' if delta > 0 else 'draw'})"

    guide = (
        "\n\nReading guide: an inventory BUILD (rising stocks) is typically bearish for the "
        "commodity (ample supply), a DRAW (falling stocks) bullish (tightening supply) — relative "
        "to seasonal norms and expectations."
    )
    return header + f"\n- {label}: {fmt_num(latest_v)} {units} as of {latest_p}{wow}" + guide
