"""Forecast analyst — the quantitative trajectory layer (Kronos), pre-fetch pattern.

Pre-fetches a probabilistic forward path from the Kronos model and has the LLM
contextualize it (base rate vs the technical/positioning picture, uncertainty,
the caveat that it is blind to exogenous catalysts). Selected via
``analysts=[...,"forecast"]`` (default-on); writes ``forecast_report``.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.kronos_forecast import get_kronos_forecast


def create_forecast_analyst(llm):
    def forecast_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(ticker)

        forecast_block = get_kronos_forecast(ticker, current_date)

        system_message = f"""You are a quantitative forecast analyst. A price-trajectory forecast for {ticker} (as of {current_date}) has been produced by the Kronos K-line foundation model and is provided below.

<start_of_forecast>
{forecast_block}
<end_of_forecast>

## How to interpret
1. **Treat the forecast as a base rate, not a certainty.** Kronos extrapolates patterns from price history only — it does NOT see catalysts (Fed, earnings, OPEC, news). Other analysts cover those.
2. **Lead with P(up) and the p10–p90 band.** A high P(up) with a tight band is a stronger forward signal than one with a wide band (high uncertainty).
3. **Call out alignment vs divergence** between the forecast and the broader picture: if the model leans up but technicals/positioning are bearish (or vice-versa), that tension is itself the key insight for the committee.
4. **State the horizon and expected return explicitly**, and be honest if the band is wide / confidence low.

## Output
A concise forward-looking report: the model's directional lean and probability, the expected return and quantile band, your read on its reliability here, and where it agrees or disagrees with a fundamentals/technical view. End with a one-line **Forecast takeaway**.
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
            "forecast_report": result.content,
        }

    return forecast_analyst_node
