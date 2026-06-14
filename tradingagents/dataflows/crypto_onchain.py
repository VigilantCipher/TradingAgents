"""Basic keyless on-chain / network metrics for crypto.

No paid analytics key (Glassnode/CryptoQuant) is available, so coverage is
intentionally BTC/ETH-centric plus DefiLlama TVL for supported L1s, and degrades
to a clear "limited coverage" note for everything else. The aim is honest
network-health context (security spend, usage, fees, staking, TVL), not a full
on-chain suite.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from ._crypto_http import fmt_num, fmt_usd, get_json
from .crypto_symbols import crypto_base


def _btc_onchain() -> list[str]:
    out: list[str] = []
    try:
        s = get_json("https://api.blockchain.info/stats", params={"format": "json"},
                     cache_key="bc_stats", ttl_seconds=3600)
        hash_ehs = (float(s["hash_rate"]) / 1e9) if s.get("hash_rate") else None  # GH/s → EH/s
        fees_sat = s.get("total_fees_btc")
        fees_btc = (fees_sat / 1e8) if (fees_sat and fees_sat > 0) else None  # satoshis → BTC
        out += [
            f"Hash rate: {hash_ehs:.1f} EH/s" if hash_ehs else "Hash rate: n/a",
            f"24h transactions: {fmt_num(s.get('n_tx'))}",
            f"24h on-chain volume (est.): {fmt_usd(s.get('estimated_transaction_volume_usd'))}",
            f"Miner revenue (24h): {fmt_usd(s.get('miners_revenue_usd'))}  |  "
            f"Total fees (24h): {fmt_num(fees_btc)} BTC",
            f"Difficulty: {fmt_num(s.get('difficulty'))}  |  Avg block time: "
            f"{s.get('minutes_between_blocks', 'n/a')} min",
        ]
    except Exception as exc:
        out.append(f"Network stats unavailable (blockchain.info): {exc}")
    try:
        fees = get_json("https://mempool.space/api/v1/fees/recommended",
                        cache_key="mempool_fees", ttl_seconds=900)
        out.append(
            f"Mempool fees (sat/vB): fast {fees.get('fastestFee', 'n/a')} | "
            f"30-min {fees.get('halfHourFee', 'n/a')} | hour {fees.get('hourFee', 'n/a')}"
        )
    except Exception:
        pass
    return out


def _eth_onchain() -> list[str]:
    out: list[str] = []
    try:
        epoch = get_json("https://beaconcha.in/api/v1/epoch/latest",
                         cache_key="beacon_epoch", ttl_seconds=3600).get("data", {})
        validators = epoch.get("validatorscount")
        staked = epoch.get("totalvalidatorbalance")  # in Gwei
        staked_eth = (staked / 1e9) if staked else None
        out.append(
            f"Validators: {fmt_num(validators)}  |  Total staked: "
            f"{fmt_num(staked_eth)} ETH" if staked_eth else f"Validators: {fmt_num(validators)}"
        )
        if epoch.get("finalitydelay") is not None:
            out.append(f"Finality delay: {epoch.get('finalitydelay')} epochs")
    except Exception as exc:
        out.append(f"Staking stats unavailable (beaconcha.in): {exc}")
    tvl = _chain_tvl("ETH")
    if tvl:
        out.append(tvl)
    return out


def _chain_tvl(base: str) -> Optional[str]:
    """DefiLlama total value locked for the L1 whose token symbol is *base*."""
    try:
        chains = get_json("https://api.llama.fi/v2/chains", cache_key="llama_chains", ttl_seconds=6 * 3600)
        for c in chains:
            if (c.get("tokenSymbol") or "").upper() == base:
                return f"DeFi TVL ({c.get('name', base)}): {fmt_usd(c.get('tvl'))}"
    except Exception:
        return None
    return None


def get_crypto_onchain(
    ticker: Annotated[str, "crypto ticker, e.g. BTC-USD"],
    curr_date: Annotated[str, "current date (unused; sources return latest)"] = None,
) -> str:
    """Keyless network-health metrics: BTC/ETH specifics + DefiLlama TVL for L1s."""
    base = crypto_base(ticker)
    header = (
        f"# On-chain / Network Health — {base}\n"
        f"# Keyless sources (no MVRV/NVT/exchange-flow analytics without a paid key) | "
        f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    if base == "BTC":
        lines = _btc_onchain()
    elif base == "ETH":
        lines = _eth_onchain()
    else:
        lines = []
        tvl = _chain_tvl(base)
        if tvl:
            lines.append(tvl)
        if not lines:
            return header + (
                f"\nLimited on-chain coverage for {base} without a paid analytics key. "
                f"No BTC/ETH-specific feed and no DefiLlama L1 match. "
                f"Lean on tokenomics, derivatives, and technicals for this asset."
            )

    if not lines:
        return header + f"\nNo on-chain metrics could be retrieved for {base} right now."
    return header + "\n" + "\n".join(f"- {l}" for l in lines)
