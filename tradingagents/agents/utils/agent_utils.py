from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    from tradingagents.dataflows.crypto_symbols import is_crypto
    from tradingagents.dataflows.commodity_symbols import is_commodity

    if is_commodity(ticker):
        return (
            f"The instrument to analyze is `{ticker}`, a COMMODITY/FUTURES contract (trades nearly "
            f"24h on exchange hours, no company behind it). Use this exact symbol in every tool "
            f"call, report, and recommendation. Futures have no financial statements — assess via "
            f"CFTC Commitments-of-Traders positioning (large speculators vs commercials), macro "
            f"drivers (US dollar & real yields), supply/inventories (EIA for energy), the futures "
            f"term structure, and technical structure."
        )

    if is_crypto(ticker):
        return (
            f"The instrument to analyze is `{ticker}`, a CRYPTO asset (trades 24/7, no exchange "
            f"close or earnings calendar). Use this exact symbol in every tool call, report, and "
            f"recommendation. Crypto has no company financial statements (no balance sheet, income "
            f"statement, or insider filings) — assess it via tokenomics & supply, on-chain network "
            f"activity, derivatives positioning (funding rate, open interest, long/short), liquidity, "
            f"and technical structure."
        )
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
