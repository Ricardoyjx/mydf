import asyncio
from datetime import datetime
import json
from typing import Any
import uuid

from my_df.agents.lead_agent.agent import make_lead_agent
from my_df.runtime.runs.manager import RunManager, RunRecord
from fastapi import HTTPException, Request
from my_df.runtime.stream_bridge.base import StreamBridge, StreamEvent
from my_df.runtime.runs.schema import DisconnectMode, RunStatus
from my_df.runtime.runs.worker import run_agent_mini


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
    # run_mgr = get_run_manager(request)

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

    agent_config: dict[str, Any] = {"recursion_limit": 100}
    if body.assistant_id:
        agent_config.setdefault("configurable", {})["assistant_id"] = body.assistant_id
    agent_factory = make_lead_agent(agent_config)
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


HEARTBEAT_SENTINEL = StreamEvent(id="", event="__heartbeat__", data=None)
END_SENTINEL = StreamEvent(id="", event="__end__", data=None)


async def see_consumer(
    bridge: StreamBridge,
    record: RunRecord,
    request: Request,
    run_mgr: RunManager,
):
    """Async generator that yields SSE frames from the bridge.

    The ``finally`` block implements ``on_disconnect`` semantics:
    - ``cancel``: abort the background task on client disconnect.
    - ``continue``: let the task run; events are discarded.
    """
    last_event_id = request.headers.get("Last-Event-ID")
    try:
        async for entry in bridge.subscribe(record.run_id, last_event_id=last_event_id):
            if await request.is_disconnected():
                break

            if entry is HEARTBEAT_SENTINEL:
                yield ": heartbeat\n\n"
                continue

            if entry is END_SENTINEL:
                yield format_sse("end", None, event_id=entry.id or None)
                return

            yield format_sse(entry.event, entry.data, event_id=entry.id or None)

    finally:
        if record.status in (RunStatus.pending, RunStatus.running):
            if record.on_disconnect == DisconnectMode.cancel:
                await run_mgr.cancel(record.run_id)


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


def format_sse(event: str, data: Any, *, event_id: str | None = None) -> str:
    """Format a single SSE frame.

    Field order: ``event:`` -> ``data:`` -> ``id:`` (optional) -> blank line.
    This matches the LangGraph Platform wire format consumed by the
    ``useStream`` React hook and the Python ``langgraph-sdk`` SSE decoder.
    """
    payload = json.dumps(data, default=str, ensure_ascii=False)
    parts = [f"event: {event}", f"data: {payload}"]
    if event_id:
        parts.append(f"id: {event_id}")
    parts.append("")
    parts.append("")
    return "\n".join(parts)
