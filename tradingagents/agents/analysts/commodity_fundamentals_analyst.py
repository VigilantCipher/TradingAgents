"""Commodity-fundamentals analyst — the futures-complex replacement for equity
fundamentals. Reasons over CFTC positioning, macro drivers (USD/yields), and
energy supply instead of financial statements. Wired under the ``fundamentals``
selector key when ``asset_class == "commodity"``; writes ``fundamentals_report``.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.commodity_data_tools import (
    get_commodity_cot,
    get_commodity_macro,
    get_commodity_supply,
)


def create_commodity_fundamentals_analyst(llm):
    def commodity_fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_commodity_cot,
            get_commodity_macro,
            get_commodity_supply,
        ]

        system_message = (
            "You are a commodity & futures research analyst. Futures have no company financial "
            "statements — assess the instrument through positioning, macro drivers, and supply. "
            "Use the available tools: `get_commodity_cot` (CFTC Commitments of Traders — large "
            "speculator vs commercial/hedger net positioning, open interest, and weekly change; "
            "watch for crowded/extreme positioning), `get_commodity_macro` (US Dollar Index and "
            "Treasury yields — the dominant cross-asset drivers; precious metals trade inverse to "
            "USD/real yields), and `get_commodity_supply` (EIA energy inventories for crude/"
            "products/natural gas — builds are bearish, draws bullish). Synthesize: Is "
            "speculative positioning crowded or extreme (reversal/squeeze risk)? Are the dollar "
            "and yields a headwind or tailwind for this category? For energy, is supply building "
            "or drawing vs seasonal norms? State the COT report date (it is weekly, as-of the "
            "prior Tuesday) and note explicitly where data is unavailable rather than guessing."
            + " Make sure to append a Markdown table at the end summarizing positioning, macro,"
            " and supply signals."
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

    return commodity_fundamentals_analyst_node
