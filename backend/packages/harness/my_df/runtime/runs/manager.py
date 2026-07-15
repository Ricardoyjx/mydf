"""运行管理器：基于内存的运行注册表，可选持久化 RunStore 支持。"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sqlite3
from typing import Any
import uuid

from my_df.runtime.runs.schema import DisconnectMode, RunStatus
from my_df.runtime.runs.store.base import RunStore

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """单次运行的可变记录。"""

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


@dataclass(frozen=True)
class PersistenceRetryPolicy:
    """Bounded retry policy for short run-store writes."""

    max_attempts: int = 5
    initial_delay: float = 0.05
    max_delay: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class RunManager:
    """内存运行注册表，可选的持久化 RunStore 支持。

    所有变更操作受 asyncio 锁保护。当提供了 ``store`` 时，
    可序列化的元数据也会持久化到存储中，以便进程重启后恢复运行历史。
    """

    def __init__(
        self,
        store: RunStore | None = None,
        *,
        persistence_retry_policy: PersistenceRetryPolicy | None = None,
    ) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()
        self._store = store
        self._persistence_retry_policy = (
            persistence_retry_policy or PersistenceRetryPolicy()
        )

    @staticmethod
    def _store_put_payload(
        record: RunRecord, *, error: str | None = None
    ) -> dict[str, Any]:
        return {
            "thread_id": record.thread_id,
            "assistant_id": record.assistant_id,
            "status": record.status.value,
            "multitask_strategy": record.multitask_strategy,
            "metadata": record.metadata or {},
            "kwargs": record.kwargs or {},
            "error": error if error is not None else record.error,
            "created_at": record.created_at,
            "model_name": record.model_name,
        }

    async def cancel(self, run_id: str, *, action: str = "interrupt") -> bool:
        """请求取消一次运行。

        参数：
            run_id: 要取消的运行 ID。
            action: "interrupt" 保留检查点，"rollback" 恢复到运行前状态。

        设置 abort_event 并取消 asyncio 任务。
        返回 ``True`` 表示取消已发起**或**运行已被中断（幂等）。
        返回 ``False`` 仅当运行未知或已处于终端状态（完成、失败等）。
        """
        async with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            # 已中断 → 幂等成功
            if record.status == RunStatus.interrupted:
                return True
            # 非待处理/运行中状态 → 不可取消
            if record.status not in (RunStatus.pending, RunStatus.running):
                return False
            record.abort_action = action
            record.abort_event.set()
            if record.task is not None and not record.task.done():
                record.task.cancel()
            record.status = RunStatus.interrupted
            record.updated_at = datetime.now().isoformat()
        logger.info("运行 %s 已取消（action=%s）", run_id, action)
        return True

    async def _call_store_with_retry(
        self,
        operation_name: str,
        run_id: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Run a short store operation with bounded retries for SQLite pressure."""
        policy = self._persistence_retry_policy
        attempt = 1
        delay = policy.initial_delay
        while True:
            try:
                return await operation()
            except Exception as exc:
                retryable = _is_retryable_persistence_error(exc)
                if attempt >= policy.max_attempts or not retryable:
                    raise
                logger.warning(
                    "Transient persistence failure during %s for run %s (attempt %d/%d); retrying",
                    operation_name,
                    run_id,
                    attempt,
                    policy.max_attempts,
                    exc_info=True,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                delay = min(
                    policy.max_delay,
                    delay * policy.backoff_factor if delay else policy.initial_delay,
                )
                attempt += 1

    async def _persist_new_run_to_store(self, record: RunRecord) -> None:
        """将新创建的运行记录保存到后备存储中

        初始运行创建是运行可见性边界的一部分：调用者
        不应观察内存中的运行，除非其后备存储行存在
        与后续状态/模型更新不同，故障会传播，因此
        调用者可以将创建视为失败回滚是调用者的
        将记录插入``runs``后的责任
        """
        if self._store is None:
            return
        await self._call_store_with_retry(
            "put",
            record.run_id,
            lambda: self._store.put(record.run_id, **self._store_put_payload(record)),  # type: ignore
        )

    async def create(
        self,
        thread_id: str,
        assistant_id: str | None = None,
        *,
        on_disconnect: DisconnectMode = DisconnectMode.cancel,
        metadata: dict | None = None,
        kwargs: dict | None = None,
        multitask_strategy: str = "reject",
    ) -> RunRecord:
        """以原子方式检查是否存在正在执行（inflight）的运行任务，并创建一个新的运行任务。
        如果采用 reject（拒绝）策略，当该线程（thread）已存在待处理或正在运行的任务时，
        将抛出 ConflictError 异常。如果采用 interrupt（中断）或 rollback（回滚）策略，
        则会在创建新任务之前，先取消当前正在执行的任务。
        此方法在执行“检查”和“插入”这两个操作时会持续持有锁，
        从而消除了将 has_inflight（检查是否存在正在执行的任务）
        与 create（创建任务）分开执行时可能出现的“检查与使用时的条件竞争”（TOCTOU race）问题
        """
        run_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            status=RunStatus.pending,
            on_disconnect=on_disconnect,
            metadata=metadata or {},
            kwargs=kwargs or {},
            created_at=now,
            updated_at=now,
        )

        async with self._lock:
            self._runs[run_id] = record
            persisted = False
            try:
                await self._persist_new_run_to_store(record)
                persisted = True
            except Exception:
                logger.warning(
                    "Failed to persist run %s; rolled back in-memory record",
                    run_id,
                    exc_info=True,
                )
                raise
            finally:
                # Also covers cancellation, which bypasses ``except Exception``.
                if not persisted:
                    self._runs.pop(run_id, None)
        logger.info("Run created: run_id=%s thread_id=%s", run_id, thread_id)
        return record


_RETRYABLE_SQLITE_ERROR_CODES = {
    sqlite3.SQLITE_BUSY,
    sqlite3.SQLITE_LOCKED,
}

_RETRYABLE_SQLITE_MESSAGES = (
    "database is locked",
    "database table is locked",
    "database is busy",
)


def _is_retryable_persistence_error(exc: BaseException) -> bool:
    """对于暂时的 SQLite 持久性失败返回 True

    SQLite 锁争用通常通过 sqlite 3 异常出现
    或 SQLAlchemy 包装器这里的短限制重试可以保护运行状态
    从暂时的作者压力中完成，而不隐藏永久的
    永远的失败
    """

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        message = str(current).lower()
        if any(fragment in message for fragment in _RETRYABLE_SQLITE_MESSAGES):
            return True
        if isinstance(current, (sqlite3.OperationalError, sqlite3.DatabaseError)):
            error_code = getattr(current, "sqlite_errorcode", None)
            if error_code in _RETRYABLE_SQLITE_ERROR_CODES:
                return True
        for chained in (
            getattr(current, "orig", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(chained, BaseException):
                pending.append(chained)
    return False
