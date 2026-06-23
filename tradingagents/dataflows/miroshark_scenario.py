"""MiroShark scenario simulation — forward event/narrative trajectory.

MiroShark (services/MiroShark) is an agent-based scenario simulator: given a
catalyst it spawns hundreds of agents that react hour-by-hour across a social
graph + prediction market, then produces a report with prediction-market odds.
That is a *forward narrative/sentiment* trajectory — complementary to the Kronos
*price* forecast.

Integration status: env-gated and fully graceful. Until ``MIROSHARK_URL`` is set AND
MiroShark is deployed (Neo4j + an LLM), this returns a clear note so default-on
committee runs are unaffected. A scenario run is heavy/async (minutes), so it is
only attempted when wired.

MiroShark API is a MULTI-STEP async pipeline (Flask, admin-token-gated) — NOT a
single call, and it does NOT take a free-text scenario directly. The real flow
(see services/MiroShark/backend/scripts/test_e2e_api.py — the reference client):

  1. POST /api/graph/ontology/generate  (multipart) — ``simulation_requirement``
     (required) + at least one document (``files=`` upload or ``url_docs``).
     Returns ``project_id``. We synthesize the document from the committee catalyst.
  2. POST /api/graph/build {project_id}   — build the Neo4j knowledge graph (async)
     → poll GET /api/graph/task/<task_id> until ``graph_id`` is produced.
  3. POST /api/simulation/create {project_id, enable_twitter/reddit/polymarket}
     → ``simulation_id`` (note: create needs the project, not a scenario string).
  4. POST /api/simulation/prepare {simulation_id} — agent profiles + config (async)
     → poll POST /api/simulation/prepare/status until ready.
  5. POST /api/simulation/start {simulation_id, platform:"parallel", max_rounds, force}
     → poll GET /api/simulation/<id>/run-status until the runner stops.
  6. POST /api/report/generate {simulation_id} (async)
     → poll POST /api/report/generate/status until completed.
  7. GET /api/report/by-simulation/<id> → ``markdown_content`` (the report).

Auth: the core pipeline (ontology/build/create/prepare/start/report) is OPEN — only
the publish/resolve/webhook endpoints (which we never call) are gated by
``require_admin_token``. We still send ``Authorization: Bearer $MIROSHARK_ADMIN_TOKEN``
when that env var is set (harmless, and future-proof if the deploy gates more). Each
response is the envelope ``{"success": bool, "data": {...}}``.

Failure policy: any error/timeout in the pipeline is caught and converted to a
short note — a scenario hiccup must never sink the committee's other analysts or
the Kronos forecast.
"""
from __future__ import annotations

import os
import time
from typing import Annotated, Any, Dict, Optional

import requests

_MIROSHARK_URL = os.getenv("MIROSHARK_URL", "").rstrip("/")
_MIROSHARK_TOKEN = os.getenv("MIROSHARK_ADMIN_TOKEN", "")

# Overall deadline (seconds) granted to each *async* phase's polling loop, and the
# per-HTTP-call socket timeout for the synchronous calls.
_TIMEOUT = int(os.getenv("MIROSHARK_TIMEOUT", "1800"))        # per async phase
_REQ_TIMEOUT = int(os.getenv("MIROSHARK_REQ_TIMEOUT", "300"))  # per HTTP call
_POLL_INTERVAL = int(os.getenv("MIROSHARK_POLL_INTERVAL", "5"))

# Keep sims short by default — a full run is slow/expensive.
_MAX_ROUNDS = int(os.getenv("MIROSHARK_MAX_ROUNDS", "3"))
_ENABLE_TWITTER = os.getenv("MIROSHARK_ENABLE_TWITTER", "true").lower() != "false"
_ENABLE_REDDIT = os.getenv("MIROSHARK_ENABLE_REDDIT", "true").lower() != "false"
_ENABLE_POLYMARKET = os.getenv("MIROSHARK_ENABLE_POLYMARKET", "true").lower() != "false"

# Reports can run many KB; cap what we feed back into the committee/job payload.
_MAX_REPORT_CHARS = int(os.getenv("MIROSHARK_REPORT_MAXCHARS", "6000"))


class MiroSharkError(RuntimeError):
    """Any non-success from the MiroShark API or a polling timeout."""


def _enabled() -> bool:
    return bool(_MIROSHARK_URL)


# ── HTTP plumbing ────────────────────────────────────────────────────────────

def _api(method: str, path: str, *, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Call MiroShark, unwrap the ``{success, data}`` envelope, raise on failure.

    Attaches the admin bearer token to every request (read at call time so a
    rotated token is picked up without a code reload).
    """
    url = f"{_MIROSHARK_URL}{path}"
    headers = kwargs.pop("headers", {}) or {}
    if _MIROSHARK_TOKEN:
        headers["Authorization"] = f"Bearer {_MIROSHARK_TOKEN}"
    try:
        resp = requests.request(method, url, headers=headers, timeout=timeout or _REQ_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise MiroSharkError(f"{method} {path}: {type(exc).__name__}") from exc

    try:
        body = resp.json()
    except ValueError as exc:
        raise MiroSharkError(f"{method} {path}: non-JSON response ({resp.status_code})") from exc

    if resp.status_code >= 400 or not body.get("success", False):
        err = body.get("error") or resp.text[:200]
        raise MiroSharkError(f"{method} {path} failed ({resp.status_code}): {err}")
    return body.get("data", {}) or {}


def _poll(make_call, label: str, timeout: Optional[int] = None):
    """Generic async-task poller. ``make_call`` returns the task's data dict.

    Treats ``completed``/``ready`` (and the ``already_*`` short-circuits) as done,
    ``failed`` as an error, and times out against a per-phase deadline.
    """
    deadline = time.time() + (timeout or _TIMEOUT)
    while True:
        data = make_call()
        status = data.get("status", "")
        if status in ("completed", "ready"):
            return data
        if status == "failed":
            raise MiroSharkError(f"{label} failed: {data.get('message') or data.get('error') or 'unknown'}")
        if data.get("already_prepared") or data.get("already_completed") or data.get("already_generated"):
            return data
        if time.time() > deadline:
            raise MiroSharkError(f"{label} timed out after {timeout or _TIMEOUT}s")
        time.sleep(_POLL_INTERVAL)


def _poll_get(task_path: str, task_id: str, label: str):
    return _poll(lambda: _api("GET", f"{task_path}/{task_id}"), label)


def _poll_post(path: str, body: Dict[str, Any], label: str):
    return _poll(lambda: _api("POST", path, json=body), label)


def _poll_run_status(simulation_id: str, label: str = "simulation run"):
    """Poll the runner until it leaves the ``running`` state."""
    deadline = time.time() + _TIMEOUT
    time.sleep(_POLL_INTERVAL)  # let the runner flip from idle→running before first check
    while True:
        data = _api("GET", f"/api/simulation/{simulation_id}/run-status")
        status = data.get("runner_status", "idle")
        if status == "failed" or status == "error":
            raise MiroSharkError(f"{label} {status}: {data.get('message') or ''}")
        if status in ("completed", "idle", "stopped"):
            return data
        if time.time() > deadline:
            raise MiroSharkError(f"{label} timed out after {_TIMEOUT}s")
        time.sleep(_POLL_INTERVAL)


# ── Scenario document construction ───────────────────────────────────────────

def _requirement(label: str, ticker: str) -> str:
    return (
        f"Simulate how market participants and public sentiment around {label} ({ticker}) "
        f"evolve over the next 1-2 weeks across Twitter, Reddit, and Polymarket, given the "
        f"committee analysis below. Focus on bullish vs bearish reactions, reflexivity, and "
        f"where the crowd's reaction diverges from current positioning."
    )


def _document(label: str, ticker: str, catalyst: str, curr_date: Optional[str]) -> str:
    return (
        f"# Trading Committee Conclusion — {label} ({ticker})\n"
        f"Date: {curr_date or 'n/a'}\n\n"
        f"{catalyst or 'General market conditions; no specific catalyst provided.'}\n"
    )


def _summarize_report(markdown: str) -> str:
    md = (markdown or "").strip()
    if not md:
        return "(MiroShark returned an empty report)"
    if len(md) <= _MAX_REPORT_CHARS:
        return md
    return md[:_MAX_REPORT_CHARS].rstrip() + f"\n\n…[truncated {len(md) - _MAX_REPORT_CHARS} chars]"


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _run_pipeline(label: str, ticker: str, catalyst: str, curr_date: Optional[str]) -> str:
    # 1) Ontology — synthesize a document from the committee catalyst and upload it.
    doc = _document(label, ticker, catalyst, curr_date)
    files = {"files": (f"{ticker}_committee.md", doc, "text/markdown")}
    form = {"simulation_requirement": _requirement(label, ticker), "project_name": ticker}
    project = _api("POST", "/api/graph/ontology/generate", files=files, data=form)
    project_id = project["project_id"]

    # 2) Build the knowledge graph (async) — prepare needs its entities for profiles.
    build = _api("POST", "/api/graph/build", json={
        "project_id": project_id,
        "graph_name": f"{ticker} scenario graph",
        "chunk_size": 500,
        "chunk_overlap": 50,
    })
    _poll_get("/api/graph/task", build["task_id"], "graph build")

    # 3) Create the simulation from the project.
    sim = _api("POST", "/api/simulation/create", json={
        "project_id": project_id,
        "enable_twitter": _ENABLE_TWITTER,
        "enable_reddit": _ENABLE_REDDIT,
        "enable_polymarket": _ENABLE_POLYMARKET,
    })
    simulation_id = sim["simulation_id"]

    # 4) Prepare agent profiles + config (async).
    prep = _api("POST", "/api/simulation/prepare", json={
        "simulation_id": simulation_id,
        "use_llm_for_profiles": True,
        "parallel_profile_count": 5,
    })
    if not prep.get("already_prepared"):
        _poll_post(
            "/api/simulation/prepare/status",
            {"task_id": prep.get("task_id"), "simulation_id": simulation_id},
            "preparation",
        )

    # 5) Run the simulation, then wait for the runner to finish.
    _api("POST", "/api/simulation/start", json={
        "simulation_id": simulation_id,
        "platform": "parallel",
        "max_rounds": _MAX_ROUNDS,
        "force": True,
    })
    _poll_run_status(simulation_id)

    # 6) Generate the analysis report (async).
    gen = _api("POST", "/api/report/generate", json={
        "simulation_id": simulation_id,
        "force_regenerate": True,
    })
    if not gen.get("already_generated"):
        _poll_post(
            "/api/report/generate/status",
            {"task_id": gen.get("task_id"), "simulation_id": simulation_id},
            "report generation",
        )

    # 7) Fetch the report markdown.
    report = _api("GET", f"/api/report/by-simulation/{simulation_id}")
    markdown = report.get("markdown_content") or report.get("markdown") or ""
    preamble = (
        f"_Simulation {simulation_id} — {_MAX_ROUNDS} rounds · "
        f"twitter={_ENABLE_TWITTER} reddit={_ENABLE_REDDIT} polymarket={_ENABLE_POLYMARKET}_\n\n"
    )
    return preamble + _summarize_report(markdown)


def get_scenario_simulation(
    ticker: Annotated[str, "instrument, e.g. BTC-USD"],
    name: Annotated[str, "human name, e.g. Bitcoin"] = "",
    catalyst: Annotated[str, "current catalyst/news framing the scenario"] = "",
    curr_date: Annotated[str, "current date yyyy-mm-dd"] = None,
) -> str:
    """Run a MiroShark scenario simulation for the instrument (or a graceful note)."""
    label = name or ticker
    header = f"# Scenario Simulation — {label}\n# Engine: MiroShark (agent-based scenario simulation)\n"

    if not _enabled():
        return header + (
            "\nScenario simulation is NOT configured. To enable: deploy MiroShark "
            "(services/MiroShark/docker-compose.yml) pointed at your local ollama, then set "
            "MIROSHARK_URL for cortex-tradingagents (MIROSHARK_ADMIN_TOKEN is optional). Until "
            "then the committee proceeds on the other analysts + the Kronos price forecast."
        )

    try:
        return header + "\n" + _run_pipeline(label, ticker, catalyst, curr_date)
    except Exception as exc:  # noqa: BLE001 — a scenario failure must never sink the committee result
        return header + (
            f"\nScenario simulation unavailable ({type(exc).__name__}: {str(exc)[:200]}). "
            "The committee proceeds on its other analysts + the Kronos forecast."
        )
