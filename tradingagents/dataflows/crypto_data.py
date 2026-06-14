"""Crypto OHLCV for TradingAgents — OKX 1D candles, with yfinance/CoinGecko fallbacks.

Source priority (all verified reachable from the containers, keyless):
  1. OKX ``/market/history-candles`` — crypto-native 24/7 daily bars, no weekend
     gaps (so the stockstats indicator window has no spurious "non-trading day"
     holes). Binance REST is geo-blocked (451), so OKX is the exchange source.
  2. yfinance ``BASE-USD`` — full daily OHLCV fallback (retries on 429).
  3. CoinGecko ``market_chart`` — daily close+volume only (Open/High/Low set to
     Close, so range indicators degrade) as a last resort.

All paths return a ``Date/Open/High/Low/Close/Volume`` DataFrame so the existing
stockstats path (``load_ohlcv``) and CSV formatting (``get_YFin_data_online``)
work unchanged. Imported lazily by those callers to avoid an import cycle.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

from ._crypto_http import get_json
from .crypto_symbols import crypto_base, to_coingecko_id, to_okx, to_yfinance

logger = logging.getLogger(__name__)

_OKX = "https://www.okx.com/api/v5"
_OKX_MAX_PER_CALL = 100
_DEFAULT_DAILY_BARS = 400  # enough for a 200-period SMA plus context
# OKX Cloudflare blocks Python's bare urllib UA; requests' default UA works, but
# we send an explicit one so behaviour does not depend on the client default.
_HEADERS = {"User-Agent": "CortexAIO-TradingAgents/1.0 (+crypto-ohlcv)"}
_COINGECKO = "https://api.coingecko.com/api/v3"
_OHLCV_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _okx_daily(symbol: str, bars: int = _DEFAULT_DAILY_BARS) -> pd.DataFrame:
    """Up to *bars* daily OKX candles, oldest-first.

    OKX ``history-candles`` returns ≤100 rows newest-first; we page backwards via
    ``after`` (oldest ts seen) until we have enough. Raises on transport errors
    or an empty response so the caller can fall back.
    """
    inst = to_okx(symbol)  # BTC -> BTC-USDT
    rows: list[list] = []
    after = ""
    while len(rows) < bars:
        params = {"instId": inst, "bar": "1Dutc", "limit": str(_OKX_MAX_PER_CALL)}
        if after:
            params["after"] = after
        resp = requests.get(f"{_OKX}/market/history-candles", params=params,
                            headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        batch = resp.json().get("data", [])
        if not batch:
            break
        rows.extend(batch)
        after = batch[-1][0]  # oldest ts in this batch → next page goes older
        if len(batch) < _OKX_MAX_PER_CALL:
            break

    if not rows:
        raise ValueError(f"OKX returned no candles for {inst}")

    # OKX candle row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[_OHLCV_COLS].dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)


def _yfinance_daily(symbol: str) -> pd.DataFrame:
    """~3y of daily ``BASE-USD`` bars from Yahoo (full OHLCV)."""
    import yfinance as yf

    from .stockstats_utils import yf_retry

    sym = to_yfinance(symbol)  # BTC -> BTC-USD
    data = yf_retry(lambda: yf.download(
        sym, period="3y", interval="1d",
        multi_level_index=False, progress=False, auto_adjust=True,
    ))
    if data is None or data.empty:
        raise ValueError(f"yfinance returned no data for {sym}")
    data = data.reset_index()
    if "Date" not in data.columns and "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Date"})
    keep = [c for c in _OHLCV_COLS if c in data.columns]
    return data[keep].copy()


def _coingecko_daily(symbol: str, days: int = 365) -> pd.DataFrame:
    """Last-resort: daily close + volume from CoinGecko (no intraday range)."""
    coin_id = to_coingecko_id(symbol)
    if not coin_id:
        raise ValueError(f"no CoinGecko id for {symbol}")
    data = get_json(
        f"{_COINGECKO}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": str(days)},
        cache_key=f"cg_chart_{coin_id}_{days}", ttl_seconds=3 * 3600,
    )
    prices = data.get("prices", [])
    volumes = {int(ts): v for ts, v in data.get("total_volumes", [])}
    if not prices:
        raise ValueError(f"CoinGecko returned no prices for {coin_id}")
    rows = []
    for ts, close in prices:
        rows.append({
            "Date": pd.to_datetime(int(ts), unit="ms").normalize(),
            "Open": close, "High": close, "Low": close, "Close": close,
            "Volume": volumes.get(int(ts), 0.0),
        })
    df = pd.DataFrame(rows, columns=_OHLCV_COLS)
    return df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


def fetch_crypto_ohlcv_df(symbol: str) -> pd.DataFrame:
    """Daily OHLCV for a crypto *symbol* (OKX → yfinance → CoinGecko).

    Columns: ``Date, Open, High, Low, Close, Volume``. Raises only if ALL three
    sources fail, which the caller treats like any other data outage.
    """
    errors = []
    for name, fn in (("OKX", _okx_daily), ("yfinance", _yfinance_daily), ("CoinGecko", _coingecko_daily)):
        try:
            return fn(symbol)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("crypto OHLCV %s failed for %s (%s)", name, symbol, exc)
    raise RuntimeError(f"all crypto OHLCV sources failed for {symbol} — " + " | ".join(errors))


def get_crypto_ohlcv_csv(symbol: str, start_date: str, end_date: str) -> str:
    """OHLCV as a CSV string for the ``get_stock_data`` tool (crypto path).

    Mirrors the header/format of ``y_finance.get_YFin_data_online`` so the market
    analyst sees an identical shape for crypto and equities.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    try:
        df = fetch_crypto_ohlcv_df(symbol)
    except Exception as exc:
        return f"No crypto data found for symbol '{symbol}': {exc}"

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()

    if df.empty:
        return f"No crypto data found for symbol '{symbol}' between {start_date} and {end_date}"

    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col].round(2)
    df = df.set_index("Date")

    canonical = crypto_base(symbol) + "-USD"
    header = f"# Crypto data for {canonical} from {start_date} to {end_date}\n"
    header += f"# Source: OKX 1D candles (yfinance/CoinGecko fallback)\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + df.to_csv()
