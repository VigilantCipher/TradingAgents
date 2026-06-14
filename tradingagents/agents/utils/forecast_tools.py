"""LangChain tool wrapper for the Kronos forecast dataflow."""
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.kronos_forecast import get_kronos_forecast as _forecast


@tool
def get_kronos_forecast(
    ticker: Annotated[str, "ticker, e.g. BTC-USD, GC=F, NVDA"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve a probabilistic forward price forecast from the Kronos K-line model:
    P(up) over the horizon, expected return, and 10/50/90 terminal quantile bands.
    A pattern-based forecast from price history only (blind to exogenous catalysts).
    """
    return _forecast(ticker, curr_date)
