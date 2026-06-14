"""Canonical crypto symbol helpers for TradingAgents — detection + normalization.

One source of truth so the server (asset-class routing), the crypto data vendor,
the analyst selection in graph setup, and the benchmark resolver all agree on
what is crypto and how to spell the pair for each upstream provider.

Mirrors services/cortex-mlsignal/app/loaders/_symbols.py (same ``_CRYPTO_BASES``
set and ``crypto_base``/``is_crypto`` semantics) so the two services classify
identically, and adds the per-provider spellings TradingAgents needs:

  to_yfinance("btc")     -> "BTC-USD"     (Yahoo crypto OHLCV / fallback)
  to_binance("btc")      -> "BTCUSDT"     (Binance spot/futures REST)
  to_okx("btc")          -> "BTC-USDT"    (OKX spot)
  to_okx_swap("btc")     -> "BTC-USDT-SWAP"(OKX perpetual / funding + OI)
  to_coingecko_id("btc") -> "bitcoin"     (CoinGecko market + tokenomics)
"""
from __future__ import annotations

from typing import Optional

# Kept in sync with cortex-mlsignal/app/loaders/_symbols.py::_CRYPTO_BASES.
_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "AVAX", "BNB", "DOT",
    "MATIC", "POL", "LINK", "TRX", "ATOM", "UNI", "XLM", "BCH", "NEAR", "APT",
    "ARB", "OP", "SUI", "TIA", "INJ", "PEPE", "WIF", "SHIB", "FIL", "AAVE",
}

_QUOTES = ("USDT", "USDC", "USD", "PERP")

# CoinGecko coin IDs for the bases we know. Bases absent here fall back to the
# lowercased base, which is correct for many coins but not all — callers should
# treat a CoinGecko miss as "unknown alt" rather than an error.
_COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "ADA": "cardano", "DOGE": "dogecoin", "LTC": "litecoin",
    "AVAX": "avalanche-2", "BNB": "binancecoin", "DOT": "polkadot",
    "MATIC": "matic-network", "POL": "polygon-ecosystem-token",
    "LINK": "chainlink", "TRX": "tron", "ATOM": "cosmos", "UNI": "uniswap",
    "XLM": "stellar", "BCH": "bitcoin-cash", "NEAR": "near", "APT": "aptos",
    "ARB": "arbitrum", "OP": "optimism", "SUI": "sui", "TIA": "celestia",
    "INJ": "injective-protocol", "PEPE": "pepe", "WIF": "dogwifcoin",
    "SHIB": "shiba-inu", "FIL": "filecoin", "AAVE": "aave",
}


def crypto_base(symbol: str) -> str:
    """Strip exchange/quote decoration to the base asset.

    BTC-USD / BTCUSD / ETHUSDT / BTC-USDT-SWAP / BTC -> BTC / ETH.
    """
    s = symbol.upper().split("-")[0].replace("_", "")
    for q in _QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def is_crypto(symbol: Optional[str]) -> bool:
    """True for crypto tickers in any common form: BTC-USD, BTCUSD, ETHUSDT, BTC."""
    return bool(symbol) and crypto_base(symbol) in _CRYPTO_BASES


def to_yfinance(symbol: str) -> str:
    """Normalize to a Yahoo Finance crypto symbol: BTC/BTCUSDT/BTC-USD -> BTC-USD."""
    return crypto_base(symbol) + "-USD"


def to_binance(symbol: str) -> str:
    """Normalize to a Binance pair: BTC/BTC-USD/BTC-USDT -> BTCUSDT."""
    return crypto_base(symbol) + "USDT"


def to_okx(symbol: str) -> str:
    """Normalize to an OKX spot instId: BTC/BTCUSDT/BTC-USD -> BTC-USDT."""
    return crypto_base(symbol) + "-USDT"


def to_okx_swap(symbol: str) -> str:
    """Normalize to an OKX perpetual instId: BTC -> BTC-USDT-SWAP (funding + OI)."""
    return crypto_base(symbol) + "-USDT-SWAP"


def to_coingecko_id(symbol: str) -> Optional[str]:
    """Return the CoinGecko coin id for *symbol*, or None if the base is unknown.

    A None result means we cannot confidently resolve the asset on CoinGecko —
    callers use that to surface "unknown crypto asset" rather than guessing.
    """
    return _COINGECKO_IDS.get(crypto_base(symbol))


def coingecko_name(symbol: str) -> Optional[str]:
    """Best-effort human name from the known base (title-cased coin id).

    For identity validation/echo only — the authoritative name comes from a live
    CoinGecko lookup when available. Returns None for unknown bases.
    """
    cid = to_coingecko_id(symbol)
    if not cid:
        return None
    # "bitcoin" -> "Bitcoin", "avalanche-2" -> "Avalanche", "matic-network" -> "Matic Network"
    cleaned = cid.rsplit("-", 1)[0] if cid.rsplit("-", 1)[-1].isdigit() else cid
    return cleaned.replace("-", " ").title()
