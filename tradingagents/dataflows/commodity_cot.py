"""Commodity/futures positioning from the CFTC Commitments of Traders report.

Keyless CFTC Socrata legacy futures-only dataset (6dca-aqww) — covers the entire
futures complex uniformly (metals, energy, ag, and financial index/rate futures)
with non-commercial (large speculators) vs commercial (hedgers/producers)
positioning. This is the commodity analog of crypto funding/OI/long-short: it is
the crowd-vs-smart-money read for futures.

COT is weekly (released Friday, as-of the prior Tuesday), so the report date is
always stated — positioning is necessarily a few days stale.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from ._crypto_http import fmt_num, get_json
from .commodity_symbols import commodity_name, cot_search

_CFTC = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"


def _f(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_commodity_cot(
    ticker: Annotated[str, "futures ticker, e.g. GC=F"],
    curr_date: Annotated[str, "current date (unused; CFTC publishes weekly)"] = None,
) -> str:
    """Latest CFTC COT positioning for the instrument's main contract."""
    name = commodity_name(ticker) or ticker
    search = cot_search(ticker)
    header = f"# COT Positioning — {name}\n# Source: CFTC Commitments of Traders (legacy, futures-only)\n"

    if not search:
        return header + f"\nNo COT mapping for {ticker}; rely on macro and technicals."

    try:
        rows = get_json(
            _CFTC,
            params={
                "$where": f"market_and_exchange_names like '%{search}%'",
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": "15",
            },
            cache_key=f"cot_{search}", ttl_seconds=6 * 3600,
        )
    except Exception as exc:
        return header + f"\nCOT data unavailable: {exc}"

    if not rows:
        return header + f"\nNo COT report found matching '{search}'."

    # Pick the main contract: highest open interest at the most recent report date
    # (so "GOLD" beats "MICRO GOLD", "E-MINI S&P 500" beats its micro, etc.).
    latest_date = max(r.get("report_date_as_yyyy_mm_dd", "") for r in rows)
    same_date = [r for r in rows if r.get("report_date_as_yyyy_mm_dd") == latest_date]
    r = max(same_date, key=lambda x: _f(x.get("open_interest_all")) or 0)

    nc_long = _f(r.get("noncomm_positions_long_all")) or 0
    nc_short = _f(r.get("noncomm_positions_short_all")) or 0
    c_long = _f(r.get("comm_positions_long_all")) or 0
    c_short = _f(r.get("comm_positions_short_all")) or 0
    oi = _f(r.get("open_interest_all"))
    nc_net = nc_long - nc_short
    c_net = c_long - c_short
    d_nc_net = (_f(r.get("change_in_noncomm_long_all")) or 0) - (_f(r.get("change_in_noncomm_short_all")) or 0)
    d_oi = _f(r.get("change_in_open_interest_all"))
    pct_nc_long = _f(r.get("pct_of_oi_noncomm_long_all"))
    pct_nc_short = _f(r.get("pct_of_oi_noncomm_short_all"))

    report_date = latest_date[:10] if latest_date else "?"
    market = r.get("market_and_exchange_names", search)
    nc_bias = "net LONG (bullish speculative positioning)" if nc_net > 0 else (
        "net SHORT (bearish speculative positioning)" if nc_net < 0 else "flat")
    c_bias = "net short (hedged/producers selling)" if c_net < 0 else "net long"

    lines = [
        f"Report date: {report_date} (weekly, as-of Tuesday) | Market: {market}",
        f"Open interest: {fmt_num(oi)} contracts" + (f" (WoW {d_oi:+,.0f})" if d_oi is not None else ""),
        f"Large speculators (non-commercial): net {fmt_num(nc_net)} contracts — {nc_bias}",
        f"  long {fmt_num(nc_long)} / short {fmt_num(nc_short)}"
        + (f" | {pct_nc_long:.0f}% / {pct_nc_short:.0f}% of OI" if pct_nc_long is not None else "")
        + (f" | WoW net {d_nc_net:+,.0f}" if d_nc_net else ""),
        f"Commercials (hedgers/producers): net {fmt_num(c_net)} contracts — {c_bias}",
        f"  long {fmt_num(c_long)} / short {fmt_num(c_short)}",
    ]
    guide = (
        "\n\nReading guide: large speculators are trend-followers/crowd — extreme net-long can mark "
        "crowded positioning (reversal/squeeze risk); extreme net-short can precede short-covering "
        "rallies. Commercials (hedgers) are typically the contrarian 'smart money' on the opposite "
        "side. Watch the WoW change for momentum in positioning."
    )
    return header + "\n" + "\n".join(f"- {l}" if not l.startswith("  ") else l for l in lines) + guide
