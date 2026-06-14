"""MiroShark scenario simulation — forward event/narrative trajectory.

MiroShark (services/MiroShark) is an agent-based scenario simulator: given a
catalyst it spawns hundreds of agents that react hour-by-hour across a social
graph + prediction market, then produces a report with prediction-market odds.
That is a *forward narrative/sentiment* trajectory — complementary to the Kronos
*price* forecast.

Integration status: env-gated and fully graceful. Until ``MIROSHARK_URL`` (and an
admin token) are set AND MiroShark is deployed with an OpenRouter key, this
returns a clear "not configured" note so default-on committee runs are unaffected.
A scenario run is heavy/async (~10 min, ~$1), so it is only attempted when wired.

MiroShark API is a MULTI-STEP async pipeline (Flask, admin-token-gated) — NOT a
single call. The real sequence to drive a sim end-to-end:
  1. POST /api/simulation/create        — create from scenario (returns simulation_id)
  2. POST /api/simulation/prepare       — build agent graph/profiles (async)
     POST /api/simulation/prepare/status  — poll until ready
  3. POST /api/simulation/start         — run hours/rounds (async)
     GET  /api/simulation/<id>/run-status — poll until complete
  4. POST /api/report/generate          — generate report (async)
     GET  /api/report/by-simulation/<id>  — fetch the report
  5. GET  /api/simulation/<id>/polymarket/markets — prediction-market odds
Auth: Bearer admin token (see backend/app/api/simulation.py::_load_admin_token).

STATUS: this client is a graceful placeholder. The full multi-step orchestration
above must be implemented and verified against a deployed MiroShark instance
before scenario simulation produces real output; until then this returns a clear
note and the committee proceeds on its other analysts + the Kronos forecast.
"""
from __future__ import annotations

import os
from typing import Annotated, Optional

import requests

_MIROSHARK_URL = os.getenv("MIROSHARK_URL", "").rstrip("/")
_MIROSHARK_TOKEN = os.getenv("MIROSHARK_ADMIN_TOKEN", "")
_TIMEOUT = int(os.getenv("MIROSHARK_TIMEOUT", "900"))  # sims are slow


def _enabled() -> bool:
    return bool(_MIROSHARK_URL)


def get_scenario_simulation(
    ticker: Annotated[str, "instrument, e.g. BTC-USD"],
    name: Annotated[str, "human name, e.g. Bitcoin"] = "",
    catalyst: Annotated[str, "current catalyst/news framing the scenario"] = "",
    curr_date: Annotated[str, "current date yyyy-mm-dd"] = None,
) -> str:
    """Run (or describe) a MiroShark scenario simulation for the instrument."""
    label = name or ticker
    header = f"# Scenario Simulation — {label}\n# Engine: MiroShark (agent-based scenario simulation)\n"

    if not _enabled():
        return header + (
            "\nScenario simulation is NOT configured. To enable: deploy MiroShark "
            "(services/MiroShark/docker-compose.yml) pointed at your local ollama, then set "
            "MIROSHARK_URL (+ MIROSHARK_ADMIN_TOKEN) for cortex-tradingagents. Until then the "
            "committee proceeds on the other analysts + the Kronos price forecast."
        )

    # MIROSHARK_URL is set, but driving a sim requires the full multi-step
    # orchestration documented in the module docstring (create → prepare → start
    # → poll → report), implemented and verified against the live instance. That
    # is a dedicated follow-up; until then we return an honest pending note rather
    # than firing an unverified call.
    scenario = (
        f"Over the next 1-2 weeks, how will market participants and sentiment around {label} "
        f"({ticker}) evolve? Current catalyst/context: {catalyst or 'general market conditions'}."
    )
    return header + (
        f"\nMiroShark endpoint configured ({_MIROSHARK_URL}), but the multi-step simulation client "
        f"(create → prepare → start → poll → report) is pending implementation/verification against "
        f"the live instance. Prepared scenario prompt:\n  {scenario}\n"
        f"Committee proceeds on its other analysts + the Kronos forecast in the meantime."
    )
