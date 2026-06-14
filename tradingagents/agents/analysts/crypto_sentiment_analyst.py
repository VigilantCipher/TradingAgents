"""Crypto sentiment analyst — pre-fetch pattern (no tool calls).

Mirrors the equity sentiment analyst: it pre-fetches the data sources and injects
them into the prompt as structured blocks, then produces the report in a single
LLM call. Sources: Crypto Fear & Greed index, crypto-subreddit discussion, and
recent crypto news headlines. Wired under the ``social`` selector key when
``asset_class == "crypto"``; writes ``sentiment_report``.
"""
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.crypto_news import get_crypto_news
from tradingagents.dataflows.crypto_sentiment import fetch_crypto_reddit, fetch_fear_greed


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_crypto_sentiment_analyst(llm):
    def crypto_sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = build_instrument_context(ticker)

        # Pre-fetch all sources; each degrades to a placeholder string.
        fng_block = fetch_fear_greed()
        reddit_block = fetch_crypto_reddit(ticker)
        news_block = get_crypto_news(ticker, start_date, end_date)

        system_message = _build_system_message(
            ticker=ticker, start_date=start_date, end_date=end_date,
            fng_block=fng_block, reddit_block=reddit_block, news_block=news_block,
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

    return crypto_sentiment_analyst_node


def _build_system_message(
    *, ticker: str, start_date: str, end_date: str,
    fng_block: str, reddit_block: str, news_block: str,
) -> str:
    return f"""You are a crypto market sentiment analyst. Produce a comprehensive sentiment report for {ticker} covering {start_date} to {end_date}, drawing on three pre-fetched sources.

## Data sources (pre-fetched, in this prompt)

### Crypto Fear & Greed index — alternative.me
A 0-100 market-wide gauge (0 = extreme fear, 100 = extreme greed). Extremes are often contrarian: extreme greed can precede corrections, extreme fear can precede bounces.

<start_of_fear_greed>
{fng_block}
<end_of_fear_greed>

### Reddit — crypto subreddits (past 7 days)
Community discussion and engagement (upvotes/comments). r/CryptoCurrency and r/CryptoMarkets are broad; asset-specific subs are more focused but more partisan.

<start_of_reddit>
{reddit_block}
<end_of_reddit>

### Crypto news headlines (past 7 days)
Event-driven framing.

<start_of_news>
{news_block}
<end_of_news>

## How to analyze

1. **Read Fear & Greed as a contrarian-aware regime gauge**, not a price call. Note the level and any swing over recent days.
2. **Weight Reddit by engagement**; distinguish genuine narrative from noise/shilling. Crypto communities skew bullish on their own asset — discount accordingly.
3. **Separate event from opinion** — a headline ("spot ETF sees record inflows") is an event; a Reddit post ("we're so back") is opinion.
4. **Look for cross-source divergence** (e.g. greedy index + bearish news flow) — divergence is itself a signal.
5. **Be honest about data limits** — if a source returned a placeholder/"unavailable", say so and lower confidence.

## Output

1. **Overall sentiment** — Bullish / Bearish / Neutral / Mixed — with a confidence note grounded in data quality.
2. **Source-by-source breakdown** with specific evidence (cite the F&G value, notable posts, key headlines).
3. **Divergences, alignments, and dominant narratives.**
4. **Catalysts and risks** surfaced by the data.
5. **Markdown table** summarizing key sentiment signals, their direction, source, and evidence.

{get_language_instruction()}"""
