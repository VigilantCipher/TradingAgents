"""Commodity news analyst — commodity/macro headlines + context.

Wired under the ``news`` selector key when ``asset_class == "commodity"``; writes
``news_report``. Leans on get_global_news, whose default queries already cover
oil/commodities/Fed/geopolitics.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)


def create_commodity_news_analyst(llm):
    def commodity_news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "You are a commodity & macro news researcher. Write a comprehensive report on what is "
            "moving this commodity/future and the broader macro backdrop over the past week. Use "
            "`get_global_news(curr_date, look_back_days, limit)` for macro headlines (it already "
            "covers Fed/inflation, oil/commodities/energy, and geopolitics) and "
            "`get_news(query, start_date, end_date)` for any instrument-specific items. Pay "
            "particular attention to the catalysts that drive the futures complex: central-bank "
            "policy and the US dollar, real yields and inflation, OPEC+/EIA supply news for "
            "energy, weather/harvest for agriculture, industrial demand/China for base metals, "
            "and safe-haven flows for precious metals. Provide specific, actionable insights with "
            "supporting evidence."
            + " Make sure to append a Markdown table at the end summarizing key points."
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
            "news_report": report,
        }

    return commodity_news_analyst_node
