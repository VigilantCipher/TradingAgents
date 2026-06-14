"""LangChain tool wrapper for the MiroShark scenario dataflow.

The scenario analyst is pre-fetch (it does not call tools), but the graph wiring
provisions a tool node for every analyst type, so this exists for that slot and
as an optional explicit tool.
"""
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.miroshark_scenario import get_scenario_simulation as _scenario


@tool
def get_scenario_simulation(
    ticker: Annotated[str, "instrument, e.g. BTC-USD, GC=F"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Run a MiroShark agent-based scenario simulation — a forward narrative/sentiment
    trajectory + prediction-market odds for how a synthetic crowd reacts to the
    current catalyst. Heavy/async; degrades gracefully when MiroShark is not configured.
    """
    return _scenario(ticker, ticker, "", curr_date)
