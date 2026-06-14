"""Macro drivers for commodities/futures — US Dollar Index + Treasury yields.

The dominant cross-asset drivers for the futures complex are the dollar (DXY) and
real/nominal yields: precious metals trade inverse to both, energy/base metals
key off the dollar and the growth cycle, and rate futures key off yields
directly. Sourced from yfinance (reachable), framed by the instrument's category.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from .commodity_symbols import category, commodity_name

_DXY = "DX-Y.NYB"
_TNX = "^TNX"  # CBOE 10-Year Treasury yield index (value is the yield in %)

_CATEGORY_FRAME = {
    "precious_metal": (
        "Precious metals trade INVERSE to the dollar and to real yields: a rising DXY / rising "
        "yields are a headwind, falling DXY / falling yields a tailwind (lower opportunity cost "
        "of holding non-yielding metal)."
    ),
    "base_metal": (
        "Base metals (Dr. Copper) are pro-cyclical growth barometers: a weaker dollar and "
        "easing yields support demand; rising real yields / strong dollar weigh on it."
    ),
    "energy": (
        "Energy keys off the dollar (inverse) and global growth/risk appetite, with supply "
        "shocks (OPEC+, geopolitics) layered on top."
    ),
    "agriculture": (
        "Agricultural futures are driven mainly by weather/supply and the dollar (a weaker "
        "dollar lifts USD-denominated export demand)."
    ),
    "financial": (
        "For financial futures the dollar and yields ARE core: index futures track risk "
        "appetite (rising yields/strong dollar can pressure equities), rate futures move "
        "inverse to yields."
    ),
}


def _series(symbol: str):
    """Return (last, chg_5d_pct, chg_20d_pct) for *symbol*, or (None, None, None)."""
    try:
        import yfinance as yf

        from .stockstats_utils import yf_retry

        hist = yf_retry(lambda: yf.Ticker(symbol).history(period="2mo"))
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None, None, None
        last = float(closes.iloc[-1])

        def chg(n):
            if len(closes) > n:
                prev = float(closes.iloc[-1 - n])
                return (last - prev) / prev * 100 if prev else None
            return None

        return last, chg(5), chg(20)
    except Exception:
        return None, None, None


def _fmt_chg(c: Optional[float]) -> str:
    return f"{c:+.2f}%" if c is not None else "n/a"


def get_commodity_macro(
    ticker: Annotated[str, "futures ticker, e.g. GC=F"],
    curr_date: Annotated[str, "current date (unused; yfinance returns latest)"] = None,
) -> str:
    """Dollar + yields snapshot, framed by the instrument's category."""
    name = commodity_name(ticker) or ticker
    cat = category(ticker) or "commodity"

    dxy_last, dxy5, dxy20 = _series(_DXY)
    tnx_last, tnx5, tnx20 = _series(_TNX)

    lines = [
        f"US Dollar Index (DXY): {dxy_last:.2f} (5d {_fmt_chg(dxy5)}, 20d {_fmt_chg(dxy20)})"
        if dxy_last is not None else "US Dollar Index (DXY): unavailable",
        f"10Y Treasury yield: {tnx_last:.2f}% (5d {_fmt_chg(tnx5)}, 20d {_fmt_chg(tnx20)})"
        if tnx_last is not None else "10Y Treasury yield: unavailable",
    ]
    frame = _CATEGORY_FRAME.get(cat, "")

    header = (
        f"# Macro Drivers — {name} ({cat})\n"
        f"# Source: yfinance (DXY, 10Y yield) | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    body = "\n".join(f"- {l}" for l in lines)
    return f"{header}\n{body}" + (f"\n\nHow this maps to {name}: {frame}" if frame else "")
