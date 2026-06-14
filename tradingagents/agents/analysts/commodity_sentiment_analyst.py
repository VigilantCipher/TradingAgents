"""Commodity sentiment analyst — positioning + macro regime (pre-fetch pattern).

For the futures complex, "sentiment" is best read from speculative positioning
(CFTC COT large-spec crowd) and the macro risk regime (dollar/yields), not retail
social chatter. Mirrors the crypto/equity sentiment analysts' no-tool-call,
pre-fetch design. Wired under the ``social`` selector key when
``asset_class == "commodity"``; writes ``sentiment_report``.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.commodity_cot import get_commodity_cot
from tradingagents.dataflows.commodity_macro import get_commodity_macro


def create_commodity_sentiment_analyst(llm):
    def commodity_sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        instrument_context = build_instrument_context(ticker)

        # Pre-fetch positioning + macro; each degrades to a placeholder string.
        cot_block = get_commodity_cot(ticker, end_date)
        macro_block = get_commodity_macro(ticker, end_date)

        system_message = _build_system_message(
            ticker=ticker, end_date=end_date, cot_block=cot_block, macro_block=macro_block,
        )

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
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "sentiment_report": result.content,
        }

    return commodity_sentiment_analyst_node


def _build_system_message(*, ticker: str, end_date: str, cot_block: str, macro_block: str) -> str:
    return f"""You are a futures market sentiment analyst. Produce a sentiment report for {ticker} as of {end_date}, reading positioning and the macro risk regime (the meaningful sentiment signals for the futures complex — not retail social chatter).

## Data sources (pre-fetched, in this prompt)

### CFTC Commitments of Traders — speculative positioning (the "crowd")
Large speculators are trend-following crowd; commercials are contrarian hedgers/"smart money". Extreme spec net-long can mark crowded positioning (reversal/squeeze risk); extreme net-short can precede short-covering rallies.

<start_of_cot>
{cot_block}
<end_of_cot>

### Macro regime — US Dollar & yields
The dominant cross-asset backdrop. Note the direction of the dollar and yields and whether it is a tailwind or headwind for this instrument's category.

<start_of_macro>
{macro_block}
<end_of_macro>

## How to analyze

1. **Read COT positioning as crowd sentiment** — is the speculative crowd heavily one-sided? How does it compare to commercials? Is positioning building or unwinding (week-over-week)?
2. **Read the macro regime** — is the dollar/yield trend supportive or hostile for this category right now?
3. **Look for alignment vs divergence** — crowded specs + hostile macro is a higher-risk setup; light positioning + supportive macro is constructive.
4. **Be honest about data limits** — COT is weekly (state the report date); if a block returned a placeholder/"unavailable", say so and lower confidence.

## Output

1. **Overall sentiment** — Bullish / Bearish / Neutral / Mixed — with a confidence note.
2. **Positioning breakdown** — specs vs commercials, extremes, weekly change (cite the numbers and report date).
3. **Macro regime read** — dollar/yields and what it implies for this instrument.
4. **Alignment / divergence and key risks.**
5. **Markdown table** summarizing the sentiment signals, direction, source, and evidence.

{get_language_instruction()}"""
