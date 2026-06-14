"""TradingAgents HTTP API Server.

Wraps TradingAgentsGraph in an async job queue so vibe-trading (and any other
caller) can submit an analysis, poll for status, and retrieve the full report
without holding a connection open for the 10-30 min runtime.

Endpoints:
  GET  /health                          — liveness + current job count
  GET  /config                          — defaults and supported option values
  POST /analyze                         — submit job → {job_id, status: "pending"}
  GET  /analyses/{job_id}               — poll: pending → running → complete/failed
  GET  /analyses/{job_id}/progress      — live agent progress log
  GET  /analyses                        — list all jobs (most recent first)
  DELETE /analyses/{job_id}             — remove a job record
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import urllib.request
import urllib.error
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.dataflows.crypto_symbols import (
    is_crypto,
    to_yfinance,
    to_coingecko_id,
    coingecko_name,
)
from tradingagents.dataflows.commodity_symbols import (
    is_commodity,
    to_yfinance as commodity_to_yfinance,
    commodity_name,
)


# ── LLM progress callback ─────────────────────────────────────────────────────

class _ProgressCallback(BaseCallbackHandler):
    """Appends a log entry whenever the LLM starts or finishes generating."""

    def __init__(self, log_fn):
        super().__init__()
        self._log = log_fn
        self._call_count = 0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        self._call_count += 1
        name = serialized.get("name") or serialized.get("id", ["LLM"])[-1]
        self._log(f"  ↳ LLM call #{self._call_count} ({name}) generating…")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        tok = 0
        for gen_list in response.generations:
            for g in gen_list:
                info = getattr(g, "generation_info", None) or {}
                tok += info.get("eval_count", 0) or info.get("completion_tokens", 0)
        suffix = f" · {tok} tokens" if tok else ""
        self._log(f"  ↳ LLM call #{self._call_count} done{suffix}")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self._log(f"  ↳ LLM error: {error}")

app = FastAPI(
    title="TradingAgents API",
    version="1.0.0",
    description="Multi-agent LLM financial analysis — async job queue interface",
)

# Single worker — analyses are GPU-bound; running two simultaneously saturates the LLM backend.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ta-worker")

# Job store — in-memory, but COMPLETED/FAILED/CANCELLED jobs are persisted to the
# data volume so analysis history survives container restarts. (In-flight jobs are
# not persisted; their worker thread dies on restart anyway.)
_jobs: dict[str, dict[str, Any]] = {}

_JOBS_DIR = os.getenv(
    "TRADINGAGENTS_JOBS_DIR",
    os.path.join(os.path.expanduser("~"), ".tradingagents", "jobs"),
)


def _persist_job(job: dict) -> None:
    """Write a terminal job to disk (best-effort)."""
    try:
        os.makedirs(_JOBS_DIR, exist_ok=True)
        path = os.path.join(_JOBS_DIR, f"{job['job_id']}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(job, fh, ensure_ascii=False, default=str)
        os.replace(tmp, path)  # atomic
    except Exception as exc:
        print(f"[persist] could not save job {job.get('job_id')}: {exc}")


def _load_jobs() -> None:
    """Load persisted jobs into the in-memory store on startup."""
    try:
        if not os.path.isdir(_JOBS_DIR):
            return
        for name in os.listdir(_JOBS_DIR):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(_JOBS_DIR, name), "r", encoding="utf-8") as fh:
                    job = json.load(fh)
                if job.get("job_id"):
                    _jobs[job["job_id"]] = job
            except Exception:
                continue
        print(f"[persist] loaded {len(_jobs)} persisted job(s) from {_JOBS_DIR}")
    except Exception as exc:
        print(f"[persist] load failed: {exc}")


_load_jobs()

# In-committee analysts. "forecast" (Kronos, CPU) runs here; "scenario" (MiroShark)
# is NOT a committee analyst — it runs as a post-committee stage (run_scenario)
# to avoid GPU contention with the Nemotron committee.
_ANALYST_ORDER = ["market", "social", "news", "fundamentals", "forecast"]

# Human-readable labels for LangGraph node names
_NODE_LABELS: dict[str, str] = {
    "Market Analyst":       "Market Analyst",
    "tools_market":         "Market data tools",
    "Social Analyst":       "Social Analyst",
    "tools_social":         "Social data tools",
    "News Analyst":         "News Analyst",
    "tools_news":           "News data tools",
    "Fundamentals Analyst": "Fundamentals Analyst",
    "tools_fundamentals":   "Fundamentals data tools",
    "Forecast Analyst":     "Forecast Analyst (Kronos)",
    "tools_forecast":       "Forecast tools",
    "Scenario Analyst":     "Scenario Analyst (MiroShark)",
    "tools_scenario":       "Scenario tools",
    "Bull Researcher":      "Bull Researcher",
    "Bear Researcher":      "Bear Researcher",
    "Research Manager":     "Research Manager",
    "Trader":               "Trader",
    "Aggressive Analyst":   "Aggressive Risk Analyst",
    "Neutral Analyst":      "Neutral Risk Analyst",
    "Conservative Analyst": "Conservative Risk Analyst",
    "Portfolio Manager":    "Portfolio Manager",
}


# ── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. NVDA, 0700.HK, 7203.T")
    date: str = Field(
        default_factory=lambda: datetime.date.today().isoformat(),
        description="Analysis date YYYY-MM-DD (defaults to today)",
    )

    analysts: list[str] = Field(
        default=["market", "social", "news", "fundamentals", "forecast"],
        description=(
            "In-committee analysts: market, social, news, fundamentals, forecast (Kronos price "
            "forecast, runs on CPU). Defaults to all. The MiroShark scenario runs separately as a "
            "post-committee stage (see run_scenario), not as a committee analyst."
        ),
    )
    run_scenario: bool = Field(
        True,
        description=(
            "Run the MiroShark scenario simulation as a post-committee stage (after the committee "
            "finishes, so it never contends with it for the GPU). No-ops with a note when MiroShark "
            "is not configured. Set false to skip for a faster run."
        ),
    )

    research_depth: int = Field(
        1, ge=1, le=5,
        description="Debate rounds for both investment and risk teams (1=fast, 5=thorough)",
    )
    max_debate_rounds: Optional[int] = Field(None)
    max_risk_discuss_rounds: Optional[int] = Field(None)

    llm_provider: str = Field("ollama")
    deep_think_llm: Optional[str] = Field(None)
    quick_think_llm: Optional[str] = Field(None)
    backend_url: Optional[str] = Field(None)

    google_thinking_level: Optional[str] = Field(None)
    openai_reasoning_effort: Optional[str] = Field(None)
    anthropic_effort: Optional[str] = Field(None)

    output_language: str = Field("English")
    news_article_limit: int = Field(20)
    global_news_article_limit: int = Field(10)
    global_news_lookback_days: int = Field(7)
    benchmark_ticker: Optional[str] = Field(None)
    data_vendors: Optional[dict[str, str]] = Field(None)
    checkpoint: bool = Field(False)


# ── Ticker resolution / identity validation ──────────────────────────────────

class TickerResolutionError(Exception):
    """Raised when a user-supplied ticker cannot be resolved to a known instrument."""


def _yfinance_identity(ticker: str) -> tuple[Optional[str], bool]:
    """Probe yfinance for an equity ticker's identity.

    Returns ``(name, exists)``. ``name`` is the company long/short name (or
    None), ``exists`` is True when we have positive evidence the symbol trades.
    Fails *open* (``(None, True)``) when the lookup itself errors, so a
    transient network/library failure never rejects an otherwise-valid ticker —
    the resolved-name echo + agent confirmation are the real guard against
    homophone tickers (FOR/WTF), not this existence check.
    """
    import math

    try:
        import yfinance as yf
    except Exception:
        return None, True

    t = yf.Ticker(ticker)
    name: Optional[str] = None
    has_price = False
    probed = False

    try:
        lp = getattr(t.fast_info, "last_price", None)
        if lp is not None and not (isinstance(lp, float) and math.isnan(lp)):
            has_price = True
        probed = True
    except Exception:
        pass

    try:
        info = t.info or {}
        name = info.get("longName") or info.get("shortName")
        if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
            has_price = True
        probed = True
    except Exception:
        pass

    if not probed:
        return None, True  # couldn't probe at all — fail open
    return name, (has_price or bool(name))


def _resolve_identity(raw_ticker: str) -> tuple[str, str, Optional[str]]:
    """Classify, normalize, and name-resolve a user-supplied ticker.

    Returns ``(normalized_ticker, asset_class, resolved_name)`` where
    ``asset_class`` is ``"crypto"`` or ``"equity"``. Raises
    :class:`TickerResolutionError` when the symbol cannot be resolved — the API
    turns that into a 422 so we never run a full analysis on a bogus string.
    """
    raw = (raw_ticker or "").strip()
    if not raw:
        raise TickerResolutionError("empty ticker")

    if is_crypto(raw):
        if to_coingecko_id(raw) is None:
            raise TickerResolutionError(
                f"'{raw}' looks like a crypto ticker but is not a recognised asset"
            )
        return to_yfinance(raw), "crypto", coingecko_name(raw)

    # Commodities / futures: map name → Yahoo =F symbol, validate via yfinance.
    # (Checked before equity so "gold"/"oil" route to the future, while an
    # uppercase equity ticker like GOLD=Barrick is left to the equity path.)
    if is_commodity(raw):
        sym = commodity_to_yfinance(raw)
        yname, exists = _yfinance_identity(sym)
        if not exists:
            raise TickerResolutionError(f"could not resolve futures symbol '{sym}'")
        return sym, "commodity", commodity_name(raw) or yname

    normalized = raw.upper()
    name, exists = _yfinance_identity(normalized)
    if not exists:
        raise TickerResolutionError(
            f"could not resolve ticker '{normalized}' to a known instrument"
        )
    return normalized, "equity", name


# ── Job runner (blocking — called in thread pool) ────────────────────────────

def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


def _run_analysis(job_id: str, req: AnalyzeRequest) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["started_at"] = _now()
    job["progress"] = []

    def log(msg: str) -> None:
        job["progress"].append({"ts": _now(), "msg": msg})

    try:
        config = DEFAULT_CONFIG.copy()

        depth = req.research_depth
        config["max_debate_rounds"] = req.max_debate_rounds if req.max_debate_rounds is not None else depth
        config["max_risk_discuss_rounds"] = req.max_risk_discuss_rounds if req.max_risk_discuss_rounds is not None else depth
        config["llm_provider"] = req.llm_provider.lower()
        if req.deep_think_llm:
            config["deep_think_llm"] = req.deep_think_llm
        if req.quick_think_llm:
            config["quick_think_llm"] = req.quick_think_llm
        if req.backend_url:
            config["backend_url"] = req.backend_url
        config["output_language"] = req.output_language
        config["google_thinking_level"] = req.google_thinking_level
        config["openai_reasoning_effort"] = req.openai_reasoning_effort
        config["anthropic_effort"] = req.anthropic_effort
        config["news_article_limit"] = req.news_article_limit
        config["global_news_article_limit"] = req.global_news_article_limit
        config["global_news_lookback_days"] = req.global_news_lookback_days
        if req.benchmark_ticker:
            config["benchmark_ticker"] = req.benchmark_ticker
        if req.data_vendors:
            config["data_vendors"].update(req.data_vendors)
        config["checkpoint_enabled"] = req.checkpoint

        analysts = [a for a in _ANALYST_ORDER if a in req.analysts]
        if not analysts:
            analysts = _ANALYST_ORDER[:]

        # Ticker + asset class were resolved/validated at submit time; reuse the
        # normalized form (crypto is stored as BASE-USD) and route the graph by
        # asset class so crypto runs the crypto-native analysts.
        ticker = job["ticker"]
        config["asset_class"] = job.get("asset_class", "equity")
        log(f"Initialising pipeline for {ticker} ({config['asset_class']})…")

        graph = TradingAgentsGraph(
            selected_analysts=analysts,
            config=config,
            debug=False,
            callbacks=[_ProgressCallback(log)],
        )

        # Resolve any pending memory-log entries before running.
        graph.ticker = ticker
        graph._resolve_pending_entries(ticker)

        past_context = graph.memory_log.get_past_context(ticker)

        # Fetch authoritative TV signal from cortex-mlsignal
        tv_signal_context = ""
        mlsignal_url = config.get("mlsignal_url", "http://cortex-mlsignal:8095")
        try:
            with urllib.request.urlopen(
                f"{mlsignal_url}/v1/logger/signals/latest/{ticker}", timeout=5
            ) as resp:
                sig = json.loads(resp.read())
            if sig.get("signal_valid"):
                feats = sig.get("features", {})
                cc = sig.get("confidence_components", {})
                rr = round((sig["tp1_price"] - sig["entry_price"]) / abs(sig["entry_price"] - sig["sl_price"]), 2) if sig.get("sl_price") and sig["entry_price"] != sig["sl_price"] else "N/A"
                tv_signal_context = (
                    f"[CSL v2 — {sig.get('timeframe','?')} — {sig.get('bars_elapsed', '?'):.1f} bars ago]\n"
                    f"Side: {sig.get('dot_direction','?').upper()} | Entry: {sig.get('entry_price','?')} | TP1: {sig.get('tp1_price','?')} | SL: {sig.get('sl_price','?')} | R:R: {rr}\n"
                    f"Fib Zone: {feats.get('fib_zone','?')} | Candle: {feats.get('candle_pattern','?')} (score: {feats.get('candle_score','?')})\n"
                    f"Momentum state: {feats.get('momentum_state', feats.get('markov_state','?'))} (p_bull={feats.get('p_bull','?')}, p_bear={feats.get('p_bear','?')}, signal={feats.get('momentum_signal', feats.get('markov_signal','?'))})\n"
                    f"Market Regime: {feats.get('market_regime','?')} | ADX: {feats.get('adx','?')} | EMA200 dist: {feats.get('ema_200_dist','?')}%\n"
                    f"RSI: {feats.get('rsi','?')} | ATR: {feats.get('atr','?')} | Vol ratio: {feats.get('vol_ratio','?')}x\n"
                    f"Macro tri_score: {feats.get('tri_score','?')}\n"
                    f"Confidence: {sig.get('confidence','?')} | Components: trend={cc.get('trend','?')} candle={cc.get('candle','?')} markov={cc.get('markov','?')} macro={cc.get('macro','?')} volatility={cc.get('volatility','?')}"
                )
                log.append(f"[TV signal context injected — {sig.get('dot_direction','?').upper()} {sig.get('timeframe','?')} {sig.get('bars_elapsed','?'):.1f} bars ago]")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                log.append(f"[TV signal fetch error: HTTP {e.code}]")
        except Exception as e:
            log.append(f"[TV signal fetch skipped: {e}]")

        init_state = graph.propagator.create_initial_state(ticker, req.date, past_context=past_context, tv_signal_context=tv_signal_context)

        # Override stream_mode to get both node names (updates) and full state (values).
        base_args = graph.propagator.get_graph_args()
        stream_args = {**base_args, "stream_mode": ["updates", "values"]}

        final_state: dict | None = None
        seen_nodes: set[str] = set()

        for mode, data in graph.graph.stream(init_state, **stream_args):
            if job.get("cancel_requested"):
                raise RuntimeError("Cancelled by user")
            if mode == "updates":
                for node_name in data:
                    if node_name.startswith("__") or node_name in seen_nodes:
                        continue
                    seen_nodes.add(node_name)
                    label = _NODE_LABELS.get(node_name, node_name)
                    log(f"{label} ✓")
            elif mode == "values":
                final_state = data

        if final_state is None:
            raise RuntimeError("Graph stream produced no output")

        # Post-processing (mirrors TradingAgentsGraph._run_graph)
        graph.curr_state = final_state
        graph._log_state(req.date, final_state)
        graph.memory_log.store_decision(
            ticker=ticker,
            trade_date=req.date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        signal = graph.process_signal(final_state["final_trade_decision"])

        # Post-committee scenario simulation (MiroShark). Runs AFTER the Nemotron
        # committee so it never contends for the GPU — the committee's conclusion
        # becomes the scenario catalyst, then MiroShark runs on its own model.
        # Self-gates to a no-op note when MiroShark isn't configured.
        scenario_report = None
        if req.run_scenario:
            try:
                from tradingagents.dataflows.miroshark_scenario import get_scenario_simulation
                log("Scenario simulation (MiroShark, post-committee)…")
                scenario_report = get_scenario_simulation(
                    ticker,
                    job.get("resolved_name") or ticker,
                    (final_state.get("final_trade_decision") or "")[:2000],
                    req.date,
                )
                log("Scenario simulation ✓")
            except Exception as exc:
                log(f"Scenario simulation skipped: {exc}")

        job["status"] = "complete"
        job["signal"] = signal
        job["market_report"]       = final_state.get("market_report")
        job["sentiment_report"]    = final_state.get("sentiment_report")
        job["news_report"]         = final_state.get("news_report")
        job["fundamentals_report"] = final_state.get("fundamentals_report")
        job["forecast_report"]     = final_state.get("forecast_report")
        job["scenario_report"]     = scenario_report
        job["investment_debate"]   = final_state.get("investment_debate_state")
        job["trader_plan"]         = final_state.get("trader_investment_plan")
        job["risk_debate"]         = final_state.get("risk_debate_state")
        job["final_decision"]      = final_state.get("final_trade_decision")
        job["completed_at"]        = _now()
        log(f"Analysis complete · Signal: {signal}")
        _persist_job(job)

    except Exception as exc:
        if job.get("cancel_requested"):
            job["status"] = "cancelled"
            job["error"] = "Cancelled by user"
            log("Cancelled by user")
        else:
            job["status"] = "failed"
            job["error"] = str(exc)
            log(f"Failed: {exc}")
        job["completed_at"] = _now()
        _persist_job(job)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    running = sum(1 for j in _jobs.values() if j["status"] == "running")
    pending = sum(1 for j in _jobs.values() if j["status"] == "pending")
    return {
        "status": "healthy",
        "service": "TradingAgents API",
        "jobs": {"running": running, "pending": pending, "total": len(_jobs)},
    }


@app.get("/config")
async def get_config():
    return {
        "defaults": {
            "analysts": _ANALYST_ORDER,
            "research_depth": 1,
            "llm_provider": DEFAULT_CONFIG["llm_provider"],
            "deep_think_llm": DEFAULT_CONFIG["deep_think_llm"],
            "quick_think_llm": DEFAULT_CONFIG["quick_think_llm"],
            "backend_url": DEFAULT_CONFIG["backend_url"],
            "output_language": DEFAULT_CONFIG["output_language"],
            "max_debate_rounds": DEFAULT_CONFIG["max_debate_rounds"],
            "max_risk_discuss_rounds": DEFAULT_CONFIG["max_risk_discuss_rounds"],
            "news_article_limit": DEFAULT_CONFIG["news_article_limit"],
            "global_news_article_limit": DEFAULT_CONFIG["global_news_article_limit"],
            "global_news_lookback_days": DEFAULT_CONFIG["global_news_lookback_days"],
            "benchmark_ticker": DEFAULT_CONFIG["benchmark_ticker"],
            "data_vendors": DEFAULT_CONFIG["data_vendors"],
        },
        "options": {
            "analysts": _ANALYST_ORDER,
            "llm_providers": ["ollama", "openai", "anthropic", "google", "deepseek", "qwen", "azure", "minimax", "glm"],
            "output_languages": ["English", "Chinese", "Japanese", "Korean", "Spanish", "French", "German"],
            "data_vendors": ["yfinance", "alpha_vantage"],
        },
    }


@app.post("/analyze", status_code=202)
async def submit_analysis(req: AnalyzeRequest):
    # Resolve + validate the ticker before queuing 10-30 min of work. Runs the
    # (blocking) yfinance probe off the event loop and off the single analysis
    # worker so a running job never stalls submission.
    try:
        ticker, asset_class, resolved_name = await asyncio.to_thread(_resolve_identity, req.ticker)
    except TickerResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "ticker": ticker,
        "asset_class": asset_class,
        "resolved_name": resolved_name,
        "date": req.date,
        "analysts": [a for a in _ANALYST_ORDER if a in req.analysts],
        "research_depth": req.research_depth,
        "llm_provider": req.llm_provider,
        "signal": None,
        "market_report": None,
        "sentiment_report": None,
        "news_report": None,
        "fundamentals_report": None,
        "forecast_report": None,
        "scenario_report": None,
        "investment_debate": None,
        "trader_plan": None,
        "risk_debate": None,
        "final_decision": None,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "progress": [],
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_analysis, job_id, req)
    return {
        "job_id": job_id,
        "status": "pending",
        "ticker": ticker,
        "asset_class": asset_class,
        "resolved_name": resolved_name,
        "date": req.date,
    }


@app.get("/analyses/{job_id}/progress")
async def get_progress(job_id: str):
    """Return the live agent progress log for a running or completed job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", []),
    }


@app.get("/analyses/{job_id}")
async def get_analysis(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/analyses")
async def list_analyses(limit: int = 20):
    jobs = sorted(_jobs.values(), key=lambda j: j.get("started_at") or "", reverse=True)
    return jobs[:limit]


@app.post("/analyses/{job_id}/cancel", status_code=200)
async def cancel_analysis(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] not in ("pending", "running"):
        raise HTTPException(status_code=409, detail=f"Job is already {job['status']}")
    job["cancel_requested"] = True
    return {"job_id": job_id, "status": "cancelling"}


@app.delete("/analyses/{job_id}", status_code=204)
async def delete_analysis(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _jobs[job_id]
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    del _jobs[job_id]
    try:
        os.remove(os.path.join(_JOBS_DIR, f"{job_id}.json"))
    except OSError:
        pass
    return JSONResponse(status_code=204, content=None)


if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(
        "server:app",
        host=os.getenv("TRADINGAGENTS_BIND", "0.0.0.0"),
        port=int(os.getenv("TRADINGAGENTS_PORT", "8090")),
        log_level="info",
    )
