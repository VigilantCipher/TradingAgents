"""LangChain tool wrappers for the commodity/futures dataflows.

Distinct tool names (``get_commodity_*``) so they never collide with the equity
or crypto tools. They call the dataflows directly — one source per metric, so no
vendor-routing layer is needed.
"""
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.commodity_cot import get_commodity_cot as _cot
from tradingagents.dataflows.commodity_macro import get_commodity_macro as _macro
from tradingagents.dataflows.commodity_supply import get_commodity_supply as _supply


@tool
def get_commodity_cot(
    ticker: Annotated[str, "futures ticker, e.g. GC=F"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve CFTC Commitments of Traders positioning for a futures contract:
    non-commercial (large speculators) vs commercial (hedgers/producers) net
    positions, open interest, % of OI, and week-over-week change. This is the
    crowd-vs-smart-money positioning read for the futures complex (the commodity
    analog of crypto funding/OI). COT is weekly (as-of the prior Tuesday).
    """
    return _cot(ticker, curr_date)


@tool
def get_commodity_macro(
    ticker: Annotated[str, "futures ticker, e.g. GC=F"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve the dominant macro drivers for a commodity/future: the US Dollar
    Index (DXY) and 10-year Treasury yield with recent trend, framed by the
    instrument's category (precious metals trade inverse to USD/real yields;
    energy/base metals key off the dollar and growth; rate futures off yields).
    """
    return _macro(ticker, curr_date)


@tool
def get_commodity_supply(
    ticker: Annotated[str, "futures ticker, e.g. CL=F"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve EIA weekly energy inventories for energy futures (crude / gasoline /
    distillate stocks, or natural-gas storage) with the week-over-week build/draw.
    Requires a free EIA_API_KEY; for non-energy futures or without a key it
    returns a clear note and the analysis leans on positioning + macro.
    """
    return _supply(ticker, curr_date)
