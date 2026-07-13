"""运行管理器：基于内存的运行注册表，可选持久化 RunStore 支持。"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import logging

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


class RunManager:
    """内存运行注册表，可选的持久化 RunStore 支持。

    所有变更操作受 asyncio 锁保护。当提供了 ``store`` 时，
    可序列化的元数据也会持久化到存储中，以便进程重启后恢复运行历史。
    """

    def __init__(
        self,
        store: RunStore | None = None,
    ) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()
        self._store = store

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
