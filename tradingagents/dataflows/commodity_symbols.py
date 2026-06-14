"""Canonical commodity/futures symbol helpers — detection + normalization.

Single source of truth (like crypto_symbols.py) for the third asset class:
the futures complex — precious & base metals, energy, agriculture, and
financial index/rate futures. Maps human names → Yahoo continuous-future
symbols (``=F``), classifies category, and provides the CFTC Commitments-of-
Traders market search string.

Homophone guard: bare lowercase/word inputs ("gold", "oil", "natural gas") map
to the future, but an UPPERCASE equity-style ticker ("GOLD" = Barrick Gold) is
NOT hijacked — it falls through to the equity path. The agent resolves names to
``=F`` symbols and confirms before submitting, so this is belt-and-suspenders.
"""
from __future__ import annotations

from typing import Optional

# Yahoo =F symbol -> metadata. ``cot`` is a substring of the CFTC
# ``market_and_exchange_names`` field; the COT fetcher picks the highest-OI
# market matching it at the latest report date (so "GOLD" selects the full
# contract over "MICRO GOLD", "E-MINI S&P 500" over micros, etc.).
_FUTURES: dict[str, dict] = {
    # Precious metals
    "GC=F": {"name": "Gold", "category": "precious_metal", "cot": "GOLD"},
    "SI=F": {"name": "Silver", "category": "precious_metal", "cot": "SILVER"},
    "PL=F": {"name": "Platinum", "category": "precious_metal", "cot": "PLATINUM"},
    "PA=F": {"name": "Palladium", "category": "precious_metal", "cot": "PALLADIUM"},
    # Base metals
    "HG=F": {"name": "Copper", "category": "base_metal", "cot": "COPPER"},
    # Energy
    "CL=F": {"name": "Crude Oil (WTI)", "category": "energy", "cot": "CRUDE OIL, LIGHT SWEET"},
    "BZ=F": {"name": "Brent Crude Oil", "category": "energy", "cot": "BRENT"},
    "NG=F": {"name": "Natural Gas", "category": "energy", "cot": "NATURAL GAS"},
    "RB=F": {"name": "RBOB Gasoline", "category": "energy", "cot": "GASOLINE RBOB"},
    "HO=F": {"name": "Heating Oil", "category": "energy", "cot": "#2 HEATING OIL"},
    # Agriculture
    "ZC=F": {"name": "Corn", "category": "agriculture", "cot": "CORN"},
    "ZW=F": {"name": "Wheat (SRW)", "category": "agriculture", "cot": "WHEAT-SRW"},
    "ZS=F": {"name": "Soybeans", "category": "agriculture", "cot": "SOYBEANS"},
    "ZL=F": {"name": "Soybean Oil", "category": "agriculture", "cot": "SOYBEAN OIL"},
    "ZM=F": {"name": "Soybean Meal", "category": "agriculture", "cot": "SOYBEAN MEAL"},
    "SB=F": {"name": "Sugar No.11", "category": "agriculture", "cot": "SUGAR NO. 11"},
    "KC=F": {"name": "Coffee", "category": "agriculture", "cot": "COFFEE C"},
    "CT=F": {"name": "Cotton", "category": "agriculture", "cot": "COTTON NO. 2"},
    "CC=F": {"name": "Cocoa", "category": "agriculture", "cot": "COCOA"},
    "LE=F": {"name": "Live Cattle", "category": "agriculture", "cot": "LIVE CATTLE"},
    "HE=F": {"name": "Lean Hogs", "category": "agriculture", "cot": "LEAN HOGS"},
    # Financial — equity index
    "ES=F": {"name": "E-mini S&P 500", "category": "financial", "cot": "E-MINI S&P 500"},
    "NQ=F": {"name": "E-mini Nasdaq-100", "category": "financial", "cot": "NASDAQ-100"},
    "YM=F": {"name": "E-mini Dow", "category": "financial", "cot": "DJIA"},
    "RTY=F": {"name": "E-mini Russell 2000", "category": "financial", "cot": "RUSSELL 2000"},
    # Financial — rates
    "ZN=F": {"name": "10-Year T-Note", "category": "financial", "cot": "10-YEAR U.S. TREASURY"},
    "ZB=F": {"name": "30-Year T-Bond", "category": "financial", "cot": "TREASURY BONDS"},
    "ZF=F": {"name": "5-Year T-Note", "category": "financial", "cot": "5-YEAR U.S. TREASURY"},
    "ZT=F": {"name": "2-Year T-Note", "category": "financial", "cot": "2-YEAR U.S. TREASURY"},
}

# Human aliases / synonyms -> =F symbol.
_ALIASES: dict[str, str] = {
    "gold": "GC=F", "xau": "GC=F", "xauusd": "GC=F",
    "silver": "SI=F", "xag": "SI=F", "xagusd": "SI=F",
    "platinum": "PL=F", "palladium": "PA=F",
    "copper": "HG=F",
    "oil": "CL=F", "crude": "CL=F", "crude oil": "CL=F", "wti": "CL=F",
    "brent": "BZ=F",
    "natural gas": "NG=F", "natgas": "NG=F", "nat gas": "NG=F",
    "gasoline": "RB=F", "rbob": "RB=F", "heating oil": "HO=F",
    "corn": "ZC=F", "wheat": "ZW=F",
    "soybean": "ZS=F", "soybeans": "ZS=F", "soy": "ZS=F",
    "soybean oil": "ZL=F", "soybean meal": "ZM=F",
    "sugar": "SB=F", "coffee": "KC=F", "cotton": "CT=F", "cocoa": "CC=F",
    "live cattle": "LE=F", "cattle": "LE=F", "lean hogs": "HE=F", "hogs": "HE=F",
    "sp500": "ES=F", "s&p": "ES=F", "s&p 500": "ES=F", "spx": "ES=F", "e-mini": "ES=F", "emini": "ES=F",
    "nasdaq": "NQ=F", "nasdaq 100": "NQ=F", "nasdaq-100": "NQ=F", "ndx": "NQ=F",
    "dow": "YM=F", "djia": "YM=F",
    "russell": "RTY=F", "russell 2000": "RTY=F",
    "10 year": "ZN=F", "10-year": "ZN=F", "10y": "ZN=F", "ten year": "ZN=F", "t-note": "ZN=F",
    "30 year": "ZB=F", "30-year": "ZB=F", "30y": "ZB=F", "t-bond": "ZB=F",
    "5 year": "ZF=F", "5y": "ZF=F", "2 year": "ZT=F", "2y": "ZT=F",
}


def _looks_like_equity_ticker(s: str) -> bool:
    """True for an UPPERCASE 1-5 letter token (e.g. GOLD=Barrick) we must not hijack."""
    return s.isalpha() and s.isupper() and 1 <= len(s) <= 5


def _resolve(symbol: Optional[str]) -> Optional[str]:
    """Return the canonical ``=F`` symbol for a commodity/futures input, else None."""
    if not symbol:
        return None
    s = symbol.strip()
    if s.upper().endswith("=F"):
        return s.upper()
    if _looks_like_equity_ticker(s):
        return None  # e.g. "GOLD" (Barrick), "OIL" — leave to the equity path
    low = s.lower()
    if low in _ALIASES:
        return _ALIASES[low]
    root = s.upper() + "=F"
    if root in _FUTURES:
        return root
    return None


def is_commodity(symbol: Optional[str]) -> bool:
    """True for a recognised futures-complex instrument (=F symbol or name alias)."""
    return _resolve(symbol) is not None


def to_yfinance(symbol: str) -> str:
    """Normalize a commodity/futures input to its Yahoo ``=F`` symbol."""
    return _resolve(symbol) or symbol


def commodity_name(symbol: str) -> Optional[str]:
    """Human name for the instrument (e.g. 'Gold', 'Crude Oil (WTI)'), or None."""
    meta = _FUTURES.get(to_yfinance(symbol))
    return meta["name"] if meta else None


def category(symbol: str) -> Optional[str]:
    """precious_metal | base_metal | energy | agriculture | financial, or None."""
    meta = _FUTURES.get(to_yfinance(symbol))
    return meta["category"] if meta else None


def cot_search(symbol: str) -> Optional[str]:
    """CFTC market_and_exchange_names search substring for the instrument, or None."""
    meta = _FUTURES.get(to_yfinance(symbol))
    return meta["cot"] if meta else None
