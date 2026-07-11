import asyncio
from datetime import datetime
from typing import Any
import uuid

from backend.packages.harness.my_df.agents.lead_agent.agent import make_lead_agent
from backend.packages.harness.runtime.runs.manager import RunRecord
from fastapi import FastAPI, HTTPException, Request
from app.gateway.deps import get_run_manager
from backend.packages.harness.runtime.runs.schema import DisconnectMode, RunStatus
from backend.packages.harness.runtime.runs.worker import run_agent_mini


async def start_run(
    body: Any,
    thread_id: str,
    request: Request,
) -> RunRecord:
    """Create a RunRecord and launch the background agent task.

    Parameters
    ----------
    body : RunCreateRequest
        The validated request body (typed as Any to avoid circular import
        with the router module that defines the Pydantic model).
    thread_id : str
        Target thread.
    request : Request
        FastAPI request — used to retrieve singletons from ``app.state``.
    """
    run_mgr = get_run_manager(request)

    now = datetime.now().isoformat()

    disconnect = (
        DisconnectMode.cancel
        if body.on_disconnect == "cancel"
        else DisconnectMode.continue_
    )
    run_id = str(uuid.uuid4())

    try:
        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id="assistant_id",
            status=RunStatus.pending,
            on_disconnect=disconnect,
            multitask_strategy="rollback",
            metadata={},
            kwargs={},
            created_at=now,
            updated_at=now,
            model_name="deepseek-v4-flash",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    agent_factory = make_lead_agent(body.assistant_id)
    graph_input = body.input
    config = build_run_config()
    task = asyncio.create_task(
        run_agent_mini(
            agent_factory=agent_factory,
            graph_input=graph_input,
            config=config,
        )
    )

    record.task = task
    return record


def build_run_config() -> dict[str, Any]:
    """Build a RunnableConfig dict for the agent.

    When *assistant_id* refers to a custom agent (anything other than
    ``"lead_agent"`` / ``None``), the name is forwarded as ``agent_name`` in
    whichever runtime options container is active: ``context`` for
    LangGraph >= 0.6.0 requests, otherwise ``configurable``.
    ``make_lead_agent`` reads this key to load the matching
    ``agents/<name>/SOUL.md`` and per-agent config — without it the agent
    silently runs as the default lead agent.

    This mirrors the channel manager's ``_resolve_run_params`` logic so that
    the LangGraph Platform-compatible HTTP API and the IM channel path behave
    identically.
    """
    config: dict[str, Any] = {"recursion_limit": 100}

    return config
