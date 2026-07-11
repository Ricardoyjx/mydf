from collections.abc import Callable
from my_df.runtime.runs.manager import RunManager
from fastapi import FastAPI, HTTPException, Request
from typing import TYPE_CHECKING, TypeVar, cast

T = TypeVar("T")


def _require(attr: str, label: str) -> Callable[[Request], T]:
    """Create a FastAPI dependency that returns ``app.state.<attr>`` or 503."""

    def dep(request: Request) -> T:
        val = getattr(request.app.state, attr, None)
        if val is None:
            raise HTTPException(status_code=503, detail=f"{label} not available")
        return cast(T, val)

    dep.__name__ = dep.__qualname__ = f"get_{attr}"
    return dep


get_stream_bridge: Callable[[Request], StreamBridge] = _require(
    "stream_bridge", "Stream bridge"
)
get_run_manager: Callable[[Request], RunManager] = _require(
    "run_manager", "Run manager"
)
# get_checkpointer: Callable[[Request], Checkpointer] = _require(
#     "checkpointer", "Checkpointer"
# )
# get_run_event_store: Callable[[Request], RunEventStore] = _require(
#     "run_event_store", "Run event store"
# )
# get_feedback_repo: Callable[[Request], FeedbackRepository] = _require(
#     "feedback_repo", "Feedback"
# )
# get_run_store: Callable[[Request], RunStore] = _require("run_store", "Run store")
