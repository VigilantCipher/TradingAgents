"""Crypto-fundamentals analyst — the crypto-native replacement for the equity
fundamentals analyst. Reasons over tokenomics, on-chain network health, and
derivatives positioning instead of financial statements. Wired into the graph
under the same ``fundamentals`` selector key when ``asset_class == "crypto"``,
so it writes ``fundamentals_report`` and the downstream debate/risk/PM layers
consume it unchanged.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.crypto_data_tools import (
    get_crypto_derivatives,
    get_crypto_fundamentals,
    get_crypto_onchain,
)


def create_crypto_fundamentals_analyst(llm):
    def crypto_fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_crypto_fundamentals,
            get_crypto_derivatives,
            get_crypto_onchain,
        ]

        system_message = (
            "You are a crypto-asset research analyst. Crypto has no company financial "
            "statements — assess the asset through tokenomics, on-chain network health, and "
            "derivatives positioning. Use the available tools: `get_crypto_fundamentals` "
            "(market cap, FDV, supply schedule & implied inflation, rank, returns, ATH/ATL "
            "drawdown, dominance), `get_crypto_derivatives` (funding rate, open interest, "
            "long/short ratio — leverage & crowding), and `get_crypto_onchain` (hash rate / "
            "validators / fees / TVL — network usage & security). Synthesize: is supply "
            "inflationary or scarce? Is positioning crowded (squeeze/unwind risk)? Is the "
            "network being used and secured? Is valuation stretched vs ATH and vs the asset's "
            "own history? Provide specific, actionable insights with the numbers cited. Note "
            "explicitly where data is unavailable (e.g. limited on-chain coverage) rather than "
            "guessing."
            + " Make sure to append a Markdown table at the end of the report organizing the key"
            " tokenomics / derivatives / on-chain signals."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return crypto_fundamentals_analyst_node
