from __future__ import annotations

import os

import requests


_MLSIGNAL_URL = os.getenv("MLSIGNAL_URL", "http://cortex-mlsignal:8095")
_TIMEOUT = 15


def create_ml_analyst(_llm=None):
    """Create an ML Analyst node that calls cortex-mlsignal instead of Nemotron.

    Pass analysts=["ml", "social", "news", "fundamentals"] to use.
    The _llm argument is accepted for interface parity but not used — the
    signal narrative is generated deterministically by cortex-mlsignal.
    """

    def ml_analyst_node(state):
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]

        try:
            resp = requests.post(
                f"{_MLSIGNAL_URL}/v1/analyst/technical",
                json={"symbol": ticker, "trade_date": trade_date, "lookback_days": 90},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            report = _format_report(ticker, trade_date, data)
        except Exception as exc:
            report = (
                f"## ML Technical Analysis — {ticker} ({trade_date})\n\n"
                f"cortex-mlsignal unavailable: {exc}\n"
                f"Proceed with other analyst reports."
            )

        return {"ml_report": report}

    return ml_analyst_node


def _format_report(ticker: str, trade_date: str, data: dict) -> str:
    signal = data.get("signal", "neutral").capitalize()
    strength = data.get("signal_strength", "weak")
    confidence = data.get("confidence", 0.0)
    regime = data.get("regime", "unknown").capitalize()
    regime_conf = data.get("regime_confidence", 0.0)
    momentum = data.get("momentum_score", 0.5)
    narrative = data.get("narrative", "")
    provider = data.get("provider", "rule_based")

    levels = data.get("key_levels", {})
    support = levels.get("support", 0)
    resistance = levels.get("resistance", 0)
    bb_upper = levels.get("bb_upper", 0)
    bb_lower = levels.get("bb_lower", 0)

    features = data.get("feature_summary", {})
    rsi = features.get("rsi_14", 0)
    macd = features.get("macd_histogram", 0)
    adx = features.get("adx_14", 0)
    atr = features.get("atr_pct", 0)
    vol = features.get("volume_ratio_20", 1)
    bb_pos = features.get("bb_position", 0.5)

    return f"""## ML Technical Analysis — {ticker} ({trade_date})

**Signal:** {signal} ({strength}, confidence: {confidence:.0%})
**Regime:** {regime} (confidence: {regime_conf:.0%})
**Momentum Score:** {momentum:.2f} / 1.0
**Provider:** {provider}

### Summary
{narrative}

### Key Levels
| Level | Price |
|---|---|
| Resistance | {resistance:.2f} |
| BB Upper | {bb_upper:.2f} |
| BB Lower | {bb_lower:.2f} |
| Support | {support:.2f} |

### Indicator Snapshot
| Indicator | Value | Reading |
|---|---|---|
| RSI (14) | {rsi:.1f} | {"Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")} |
| MACD Histogram | {macd:+.4f} | {"Positive" if macd > 0 else "Negative"} |
| ADX (14) | {adx:.1f} | {"Strong trend" if adx > 25 else "Weak/no trend"} |
| ATR % | {atr:.2%} | {"High volatility" if atr > 0.05 else "Normal"} |
| Volume Ratio | {vol:.2f}x | {"Above avg" if vol > 1.1 else "Below avg"} |
| BB Position | {bb_pos:.2f} | {"Upper zone" if bb_pos > 0.7 else ("Lower zone" if bb_pos < 0.3 else "Mid-band")} |
"""
