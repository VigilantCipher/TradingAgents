"""LangChain tool wrappers for the crypto dataflows.

Distinct tool names (``get_crypto_*``) so they never collide with the equity
fundamental tools. These call the crypto dataflows directly — there is a single
crypto source per metric, so no vendor-routing/fallback layer is needed.
"""
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.crypto_derivatives import get_crypto_derivatives as _derivatives
from tradingagents.dataflows.crypto_fundamentals import get_crypto_fundamentals as _fundamentals
from tradingagents.dataflows.crypto_news import get_crypto_news as _news
from tradingagents.dataflows.crypto_onchain import get_crypto_onchain as _onchain


@tool
def get_crypto_fundamentals(
    ticker: Annotated[str, "crypto ticker, e.g. BTC-USD"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve crypto tokenomics & market structure (CoinGecko): price, market cap,
    FDV, circulating/total/max supply and implied inflation, market-cap rank,
    multi-horizon returns, ATH/ATL drawdowns, and BTC/ETH dominance.
    This is the crypto replacement for equity fundamentals (there is no PE,
    balance sheet, or income statement for a token).
    """
    return _fundamentals(ticker, curr_date)


@tool
def get_crypto_derivatives(
    ticker: Annotated[str, "crypto ticker, e.g. BTC-USD"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve crypto derivatives positioning from OKX perpetual swaps: funding rate
    (8h + annualized), open interest, and the long/short account ratio. Use this
    to gauge leverage, crowding, and squeeze/unwind risk.
    """
    return _derivatives(ticker, curr_date)


@tool
def get_crypto_onchain(
    ticker: Annotated[str, "crypto ticker, e.g. BTC-USD"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve keyless on-chain / network-health metrics: BTC (hash rate, tx count,
    fees, miner revenue, mempool), ETH (validators, total staked, TVL), or
    DefiLlama TVL for other supported L1s. Coverage is limited without a paid
    analytics key and degrades gracefully for unsupported assets.
    """
    return _onchain(ticker, curr_date)


@tool
def get_crypto_news(
    query: Annotated[str, "search focus, e.g. BTC-USD or 'bitcoin etf'"],
    start_date: Annotated[str, "start date yyyy-mm-dd"],
    end_date: Annotated[str, "end date yyyy-mm-dd"],
) -> str:
    """
    Retrieve recent crypto news headlines from CoinDesk / CoinTelegraph / Bitcoin
    Magazine RSS (plus CryptoPanic when CRYPTOPANIC_TOKEN is set), filtered to the
    query terms and date window. Falls back to general crypto-market headlines
    when nothing specifically names the asset.
    """
    return _news(query, start_date, end_date)
