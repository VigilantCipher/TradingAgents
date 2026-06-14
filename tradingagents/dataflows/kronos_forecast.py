"""Kronos forward-price forecast — calls the cortex-kronos inference service.

This is the quantitative trajectory layer: it feeds recent OHLCV (via the same
per-asset path used everywhere — crypto→OKX/yfinance, equity/commodity→yfinance)
to the Kronos K-line model and returns a probabilistic forward path: P(up) over
the horizon, expected return, and 10/50/90 terminal quantile bands. Asset-agnostic.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated

import requests

from .crypto_symbols import is_crypto

_KRONOS_URL = os.getenv("KRONOS_URL", "http://cortex-kronos:8092")
_TIMEOUT = 180  # N CPU predict() passes can take ~1 min
_LOOKBACK = 256


def _recent_ohlcv(ticker: str, curr_date: str) -> list[dict]:
    """Recent daily OHLCV as the forecast input, using the per-asset fetch path."""
    if is_crypto(ticker):
        from .crypto_data import fetch_crypto_ohlcv_df
        df = fetch_crypto_ohlcv_df(ticker)
    else:
        from .stockstats_utils import load_ohlcv
        df = load_ohlcv(ticker, curr_date)  # handles equity + =F commodity/futures

    if df is None or df.empty:
        return []
    df = df.tail(_LOOKBACK)
    out = []
    for _, r in df.iterrows():
        d = r["Date"]
        out.append({
            "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10],
            "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]),
            "close": float(r["Close"]), "volume": float(r.get("Volume", 0.0) or 0.0),
        })
    return out


def get_kronos_forecast(
    ticker: Annotated[str, "ticker the server resolved (e.g. BTC-USD, GC=F, NVDA)"],
    curr_date: Annotated[str, "current date yyyy-mm-dd"] = None,
    pred_len: int = 10,
) -> str:
    """Probabilistic forward price path for *ticker* from the Kronos model."""
    header = f"# Kronos Price Forecast — {ticker}\n# Model: Kronos (K-line foundation model)\n"

    try:
        bars = _recent_ohlcv(ticker, curr_date or datetime.now().strftime("%Y-%m-%d"))
    except Exception as exc:
        return header + f"\nCould not load OHLCV for the forecast: {exc}"
    if len(bars) < 30:
        return header + f"\nNot enough price history ({len(bars)} bars) to forecast."

    try:
        r = requests.post(
            f"{_KRONOS_URL}/forecast",
            json={"ohlcv": bars, "symbol": ticker, "pred_len": pred_len},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        f = r.json()
    except Exception as exc:
        return header + f"\nForecast service (cortex-kronos) unavailable: {exc}"

    t = f.get("terminal", {})
    direction = "UP" if f.get("prob_up", 0.5) >= 0.5 else "DOWN"
    lines = [
        f"Horizon: {f.get('horizon_bars')} bars  |  Last close: {f.get('last_close')}  "
        f"|  Sampled paths: {f.get('paths')}",
        f"**P(up over horizon): {f.get('prob_up', 0) * 100:.0f}%** → lean {direction}",
        f"Expected return (median): {f.get('expected_return_pct')}%  "
        f"(model confidence proxy: {f.get('confidence', 0) * 100:.0f}%)",
        f"Terminal quantiles: p10 {t.get('p10')} ({t.get('p10_return_pct')}%) | "
        f"p50 {t.get('p50')} | p90 {t.get('p90')} ({t.get('p90_return_pct')}%)",
    ]
    guide = (
        "\n\nReading guide: this is a pattern-based forecast from price history only — it does NOT "
        "see exogenous catalysts (Fed, earnings, news). Treat P(up) and the quantile band as a base "
        "rate to weigh against the technical/positioning/news picture, not a certainty. A wide "
        "p10–p90 band = high uncertainty; align/divergence vs the trend is the signal."
    )
    return header + "\n" + "\n".join(f"- {l}" for l in lines) + guide
