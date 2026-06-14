"""Scenario analyst — forward event/narrative simulation (MiroShark), pre-fetch.

Pre-fetches a MiroShark scenario simulation (how a synthetic crowd reacts to the
current catalyst over time + prediction-market odds) and has the LLM contextualize
it. Selected via ``analysts=[...,"scenario"]`` (default-on); writes
``scenario_report``. Degrades gracefully when MiroShark is not configured.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.miroshark_scenario import get_scenario_simulation


def create_scenario_analyst(llm):
    def scenario_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(ticker)
        # Use any news already gathered as the catalyst framing.
        catalyst = (state.get("news_report") or "")[:1500]

        scenario_block = get_scenario_simulation(ticker, ticker, catalyst, current_date)

        system_message = f"""You are a scenario analyst. A forward agent-based scenario simulation for {ticker} (as of {current_date}) is provided below (or a note if the simulator is not configured).

<start_of_scenario>
{scenario_block}
<end_of_scenario>

## How to interpret
1. This simulates how a synthetic crowd reacts to the current catalyst over time — a forward **narrative/sentiment** trajectory and prediction-market odds, NOT a price forecast.
2. Use it to gauge consensus, reflexivity, and how positioning/narrative might shift; flag where the simulated reaction diverges from current sentiment.
3. If the simulation is "not configured" or failed, say so plainly and add no speculative content — defer to the other analysts and the Kronos price forecast.

## Output
A concise report: the simulated forward narrative, any prediction-market odds, and what it implies for the trade — or a one-line note that scenario simulation was unavailable. End with a one-line **Scenario takeaway**.
{get_language_instruction()}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "scenario_report": result.content,
        }

    return scenario_analyst_node
