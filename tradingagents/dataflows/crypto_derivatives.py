"""Crypto derivatives positioning — OKX perpetuals, CoinGecko aggregate fallback.

OKX (keyless, reachable from the containers) is the primary: it gives the three
crypto-native positioning signals — funding rate, open interest, and the
long/short *account* ratio. If OKX has no perp for the asset (or is briefly
unavailable) we fall back to CoinGecko's cross-venue ``/derivatives`` aggregate,
which still carries funding + OI. Binance fapi is geo-blocked (451) and unused.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from ._crypto_http import fmt_num, fmt_usd, get_json
from .crypto_symbols import crypto_base, to_okx_swap

logger = logging.getLogger(__name__)

_OKX = "https://www.okx.com/api/v5"
_COINGECKO = "https://api.coingecko.com/api/v3"
# OKX perps fund every 8h → 3 intervals/day.
_FUNDING_PER_YEAR = 3 * 365

_GUIDE = (
    "\n\nReading guide: persistently positive funding + high/rising OI = leveraged longs "
    "crowding (squeeze/correction risk); negative funding + rising OI = bearish crowding. "
    "OI rising with price = trend conviction; OI falling = position unwind. A long/short "
    "account ratio >1 means more accounts are net long (retail crowding)."
)


def _f(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _okx_derivatives(base: str, inst: str) -> Optional[str]:
    """Funding + OI + long/short account ratio from OKX perps, or None if no perp."""
    lines: list[str] = []

    last = None
    tick = get_json(f"{_OKX}/market/ticker", params={"instId": inst},
                    cache_key=f"okx_tick_{inst}", ttl_seconds=300).get("data", [])
    if tick:
        last = _f(tick[0].get("last"))

    fr = get_json(f"{_OKX}/public/funding-rate", params={"instId": inst},
                  cache_key=f"okx_funding_{inst}", ttl_seconds=300).get("data", [])
    if fr:
        rate = _f(fr[0].get("fundingRate"))
        if rate is not None:
            annual = rate * _FUNDING_PER_YEAR * 100
            bias = ("longs pay shorts (bullish crowding)" if rate > 0
                    else "shorts pay longs (bearish crowding)" if rate < 0 else "balanced")
            lines.append(
                f"Funding rate (8h): {rate * 100:+.4f}% → ~{annual:+.1f}%/yr — {bias}"
            )

    oi = get_json(f"{_OKX}/public/open-interest", params={"instType": "SWAP", "instId": inst},
                  cache_key=f"okx_oi_{inst}", ttl_seconds=300).get("data", [])
    if oi:
        oi_ccy = _f(oi[0].get("oiCcy"))  # OI in base coin
        if oi_ccy is not None:
            usd = f" (~{fmt_usd(oi_ccy * last)})" if last else ""
            lines.append(f"Open interest: {fmt_num(oi_ccy)} {base}{usd}")

    try:
        lsr = get_json(f"{_OKX}/rubik/stat/contracts/long-short-account-ratio",
                       params={"ccy": base, "period": "1D"},
                       cache_key=f"okx_lsr_{base}", ttl_seconds=900).get("data", [])
        if lsr and len(lsr[0]) > 1 and (ratio := _f(lsr[0][1])) is not None:
            skew = ("more accounts long" if ratio > 1
                    else "more accounts short" if ratio < 1 else "balanced")
            lines.append(f"Long/short account ratio (1D): {ratio:.2f} — {skew}")
    except Exception:
        pass  # rubik stats are best-effort and not available for every asset

    if not lines:
        return None
    return "\n".join(f"- {l}" for l in lines)


def _coingecko_derivatives(base: str) -> Optional[str]:
    """Cross-venue funding + OI aggregate from CoinGecko, or None if no perp."""
    rows = get_json(f"{_COINGECKO}/derivatives", params={"include_tickers": "unexpired"},
                    cache_key="cg_derivatives", ttl_seconds=600)
    perps = [
        r for r in rows
        if r.get("index_id") == base and r.get("contract_type") == "perpetual" and not r.get("expired_at")
    ]
    if not perps:
        return None

    total_oi = sum(oi for r in perps if (oi := _f(r.get("open_interest"))))
    total_vol = sum(v for r in perps if (v := _f(r.get("volume_24h"))))
    weighted_num, weight = 0.0, 0.0
    for r in perps:
        f_r, oi = _f(r.get("funding_rate")), _f(r.get("open_interest"))
        if f_r is not None and oi:
            weighted_num += f_r * oi
            weight += oi
    funding = (weighted_num / weight) if weight else None

    lines = [f"Venues with a {base} perp: {len(perps)}"]
    if total_oi:
        lines.append(f"Aggregate open interest: {fmt_usd(total_oi)}")
    if total_vol:
        lines.append(f"Aggregate 24h volume: {fmt_usd(total_vol)}")
    if funding is not None:
        annual = funding * _FUNDING_PER_YEAR
        bias = ("longs pay shorts (bullish crowding)" if funding > 0
                else "shorts pay longs (bearish crowding)" if funding < 0 else "balanced")
        lines.append(f"OI-weighted funding rate: {funding:+.4f}%/interval (~{annual:+.1f}%/yr) — {bias}")
    return "\n".join(f"- {l}" for l in lines)


def get_crypto_derivatives(
    ticker: Annotated[str, "crypto ticker, e.g. BTC-USD"],
    curr_date: Annotated[str, "current date (unused; venues return live state)"] = None,
) -> str:
    """Funding rate, open interest, and positioning — OKX primary, CoinGecko fallback."""
    base = crypto_base(ticker)
    header = (
        f"# Crypto Derivatives Positioning — {base}\n"
        f"# Retrieved: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
    )

    try:
        body = _okx_derivatives(base, to_okx_swap(ticker))
        if body:
            return f"{header}# Source: OKX perpetual swaps\n\n{body}{_GUIDE}"
    except Exception as exc:
        logger.warning("OKX derivatives failed for %s (%s); trying CoinGecko", base, exc)

    try:
        body = _coingecko_derivatives(base)
        if body:
            return f"{header}# Source: CoinGecko aggregated perpetuals (OKX unavailable)\n\n{body}{_GUIDE}"
    except Exception as exc:
        return header + f"\nDerivatives data unavailable: {exc}"

    return header + (
        f"\nNo listed perpetual market for {base} on tracked venues. "
        f"Rely on technicals and tokenomics for this asset."
    )
