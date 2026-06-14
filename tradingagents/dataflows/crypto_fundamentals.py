"""Crypto "fundamentals" = tokenomics + market structure, from CoinGecko (keyless).

Replaces the equity fundamentals (PE / debt-to-equity / margins) that do not
exist for crypto with the metrics that actually drive a token's value: market
cap & FDV, circulating/total/max supply and implied inflation, rank, ATH/ATL
drawdowns, multi-horizon returns, and BTC/ETH dominance.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from ._crypto_http import fmt_num, fmt_usd, get_json
from .crypto_symbols import crypto_base, to_coingecko_id

_COINGECKO = "https://api.coingecko.com/api/v3"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def get_crypto_fundamentals(
    ticker: Annotated[str, "crypto ticker, e.g. BTC-USD"],
    curr_date: Annotated[str, "current date (unused; CoinGecko returns latest)"] = None,
) -> str:
    """Tokenomics & market-structure overview for a crypto asset from CoinGecko."""
    base = crypto_base(ticker)
    coin_id = to_coingecko_id(ticker)
    if not coin_id:
        return (
            f"No tokenomics data: '{base}' is not a recognised CoinGecko asset. "
            f"Proceed using technicals and derivatives only."
        )

    try:
        coin = get_json(
            f"{_COINGECKO}/coins/{coin_id}",
            params={
                "localization": "false", "tickers": "false", "market_data": "true",
                "community_data": "false", "developer_data": "false", "sparkline": "false",
            },
            cache_key=f"cg_coin_{coin_id}",
            ttl_seconds=3 * 3600,
        )
    except Exception as exc:
        return f"Error retrieving CoinGecko tokenomics for {base}: {exc}"

    md = coin.get("market_data", {}) or {}

    def _usd(field: str) -> Optional[float]:
        v = md.get(field)
        return v.get("usd") if isinstance(v, dict) else v

    price = _usd("current_price")
    mcap = _usd("market_cap")
    fdv = _usd("fully_diluted_valuation")
    vol = _usd("total_volume")
    circ = md.get("circulating_supply")
    total = md.get("total_supply")
    max_supply = md.get("max_supply")
    ath = _usd("ath")
    ath_chg = (md.get("ath_change_percentage") or {}).get("usd") if isinstance(md.get("ath_change_percentage"), dict) else md.get("ath_change_percentage")
    atl = _usd("atl")

    circ_pct = f"{(circ / max_supply * 100):.1f}% of max" if (circ and max_supply) else (
        f"{(circ / total * 100):.1f}% of total" if (circ and total) else "n/a")

    # BTC/ETH dominance from the global snapshot (best-effort).
    dominance_line = ""
    try:
        glob = get_json(f"{_COINGECKO}/global", cache_key="cg_global", ttl_seconds=3 * 3600)
        mcp = (glob.get("data", {}) or {}).get("market_cap_percentage", {}) or {}
        btc_dom, eth_dom = mcp.get("btc"), mcp.get("eth")
        if btc_dom is not None:
            dominance_line = f"BTC dominance: {btc_dom:.1f}% | ETH dominance: {eth_dom:.1f}%"
    except Exception:
        pass

    name = coin.get("name", base)
    rank = coin.get("market_cap_rank")
    categories = [c for c in (coin.get("categories") or []) if c][:6]

    lines = [
        f"# Crypto Fundamentals (Tokenomics) — {name} ({base})",
        f"# Source: CoinGecko | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Price: {fmt_usd(price)}  |  Market-cap rank: #{rank if rank else 'n/a'}",
        f"Market cap: {fmt_usd(mcap)}  |  Fully-diluted valuation (FDV): {fmt_usd(fdv)}",
        f"24h volume: {fmt_usd(vol)}  |  Volume/MCap: "
        + (f"{(vol / mcap):.3f}" if (vol and mcap) else "n/a"),
        "",
        "## Supply",
        f"Circulating: {fmt_num(circ)}  ({circ_pct})",
        f"Total: {fmt_num(total)}  |  Max: {fmt_num(max_supply) if max_supply else 'uncapped'}",
        "",
        "## Returns & extremes",
        f"24h: {_pct(md.get('price_change_percentage_24h'))}  |  "
        f"7d: {_pct(md.get('price_change_percentage_7d'))}  |  "
        f"30d: {_pct(md.get('price_change_percentage_30d'))}  |  "
        f"1y: {_pct(md.get('price_change_percentage_1y'))}",
        f"All-time high: {fmt_usd(ath)} ({_pct(ath_chg)} from ATH)  |  All-time low: {fmt_usd(atl)}",
    ]
    if dominance_line:
        lines += ["", "## Market context", dominance_line]
    if categories:
        lines += ["", f"Categories: {', '.join(categories)}"]

    lines += [
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Price | {fmt_usd(price)} |",
        f"| Market cap (rank) | {fmt_usd(mcap)} (#{rank if rank else 'n/a'}) |",
        f"| FDV | {fmt_usd(fdv)} |",
        f"| Circulating supply | {fmt_num(circ)} ({circ_pct}) |",
        f"| Max supply | {fmt_num(max_supply) if max_supply else 'uncapped'} |",
        f"| 24h volume | {fmt_usd(vol)} |",
        f"| 7d / 30d return | {_pct(md.get('price_change_percentage_7d'))} / {_pct(md.get('price_change_percentage_30d'))} |",
        f"| From ATH | {_pct(ath_chg)} |",
    ]
    return "\n".join(lines)
