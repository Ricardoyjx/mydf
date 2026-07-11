import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging

from my_df.runtime.runs.schema import DisconnectMode, RunStatus
from my_df.runtime.runs.store.base import RunStore

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Mutable record for a single run."""

    run_id: str
    thread_id: str
    assistant_id: str | None
    status: RunStatus
    on_disconnect: DisconnectMode
    multitask_strategy: str = "reject"
    metadata: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    task: asyncio.Task | None = field(default=None, repr=False)
    abort_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    abort_action: str = "interrupt"
    error: str | None = None
    model_name: str | None = None
    store_only: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    llm_call_count: int = 0
    lead_agent_tokens: int = 0
    subagent_tokens: int = 0
    middleware_tokens: int = 0
    message_count: int = 0
    last_ai_message: str | None = None
    first_human_message: str | None = None


class RunManager:
    """In-memory run registry with optional persistent RunStore backing.

    All mutations are protected by an asyncio lock. When a ``store`` is
    provided, serializable metadata is also persisted to the store so
    that run history survives process restarts.
    """

    def __init__(
        self,
        store: RunStore | None = None,
    ) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()
        self._store = store

    # async def _persist_status(
    #     self, record: RunRecord, status: RunStatus, *, error: str | None = None
    # ) -> bool:
    #     """Best-effort persist a status transition to the backing store."""
    #     if self._store is None:
    #         return True
    #     row_recovery_payload = self._store_put_payload(record, error=error)
    #     try:
    #         updated = await self._call_store_with_retry(
    #             "update_status",
    #             record.run_id,
    #             lambda: self._store.update_status(
    #                 record.run_id, status.value, error=error
    #             ),
    #         )
    #         if updated is False:
    #             return await self._persist_snapshot_to_store(
    #                 record.run_id, row_recovery_payload
    #             )
    #         return True
    #     except Exception:
    #         logger.warning(
    #             "Failed to persist status update for run %s",
    #             record.run_id,
    #             exc_info=True,
    #         )
    #         return False

    async def cancel(self, run_id: str, *, action: str = "interrupt") -> bool:
        """Request cancellation of a run.

        Args:
            run_id: The run ID to cancel.
            action: "interrupt" keeps checkpoint, "rollback" reverts to pre-run state.

        Sets the abort event with the action reason and cancels the asyncio task.
        Returns ``True`` if cancellation was initiated **or** the run was already
        interrupted (idempotent — a second cancel is a no-op success).
        Returns ``False`` only when the run is unknown to this worker or has
        reached a terminal state other than interrupted (completed, failed, etc.).
        """
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            if record.status == RunStatus.interrupted:
                return True  # idempotent — already cancelled on this worker
            if record.status not in (RunStatus.pending, RunStatus.running):
                return False
            record.abort_action = action
            record.abort_event.set()
            if record.task is not None and not record.task.done():
                record.task.cancel()
            record.status = RunStatus.interrupted
            record.updated_at = datetime.now().isoformat()
        # await self._persist_status(record, RunStatus.interrupted)
        logger.info("Run %s cancelled (action=%s)", run_id, action)
        return True
