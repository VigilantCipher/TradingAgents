"""Crypto news analyst — crypto-specific headlines + macro context.

Wired under the ``news`` selector key when ``asset_class == "crypto"``; writes
``news_report`` for the downstream debate/risk/PM layers.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
)
from tradingagents.agents.utils.crypto_data_tools import get_crypto_news


def create_crypto_news_analyst(llm):
    def crypto_news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_crypto_news,
            get_global_news,
        ]

        system_message = (
            "You are a crypto news researcher analyzing recent headlines and trends over the past "
            "week. Write a comprehensive report on what is moving this asset and the broader crypto "
            "market. Use `get_crypto_news(query, start_date, end_date)` for crypto-specific "
            "headlines and `get_global_news(curr_date, look_back_days, limit)` for macro. Pay "
            "particular attention to crypto-specific catalysts: regulation and enforcement, "
            "spot/ETF flows and approvals, exchange or protocol incidents (hacks, depegs, halts), "
            "token unlocks and major listings/delistings, and macro drivers that move risk assets "
            "(Fed policy, real rates, USD, liquidity). Provide specific, actionable insights with "
            "supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points,"
            " organized and easy to read."
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

    return crypto_news_analyst_node
