"""运行存储的抽象接口。

RunManager 依赖此接口。实现类：
- MemoryRunStore：基于内存字典（开发、测试）
- 未来：基于 SQLAlchemy ORM 的 RunRepository

所有方法接受可选的 user_id 参数以实现用户隔离。
当 user_id 为 None 时不进行用户过滤（单用户模式）。
"""

from __future__ import annotations

import abc
from typing import Any


class RunStore(abc.ABC):
    """运行记录的持久化存储抽象基类。"""

    @abc.abstractmethod
    async def put(
        self,
        run_id: str,
        *,
        thread_id: str,
        assistant_id: str | None = None,
        user_id: str | None = None,
        model_name: str | None = None,
        status: str = "pending",
        multitask_strategy: str = "reject",
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        error: str | None = None,
        created_at: str | None = None,
    ) -> None:
        """写入一条运行记录。"""

    @abc.abstractmethod
    async def get(
        self,
        run_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """查询单条运行记录。"""

    @abc.abstractmethod
    async def list_by_thread(
        self,
        thread_id: str,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按线程列出运行记录。"""

    @abc.abstractmethod
    async def update_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> bool | None:
        """更新运行状态。

        返回 ``False`` 表示存储可以证明没有行被更新；
        轻量级或旧版存储可能返回 ``None``（无法报告行数）。
        """

    @abc.abstractmethod
    async def delete(self, run_id: str) -> bool:
        """删除运行记录。"""

    @abc.abstractmethod
    async def update_model_name(
        self,
        run_id: str,
        model_name: str | None,
    ) -> None:
        """更新已有运行的 model_name 字段。"""

    @abc.abstractmethod
    async def update_run_completion(
        self,
        run_id: str,
        *,
        status: str,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_tokens: int = 0,
        llm_call_count: int = 0,
        lead_agent_tokens: int = 0,
        subagent_tokens: int = 0,
        middleware_tokens: int = 0,
        message_count: int = 0,
        last_ai_message: str | None = None,
        first_human_message: str | None = None,
        error: str | None = None,
    ) -> bool | None:
        """持久化最终完成状态字段。"""

    async def update_run_progress(
        self,
        run_id: str,
        *,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_tokens: int | None = None,
        llm_call_count: int | None = None,
        lead_agent_tokens: int | None = None,
        subagent_tokens: int | None = None,
        middleware_tokens: int | None = None,
        message_count: int | None = None,
        last_ai_message: str | None = None,
        first_human_message: str | None = None,
    ) -> None:
        """运行时进度快照（尽力而为，不改变运行状态）。"""

    @abc.abstractmethod
    async def list_pending(self, *, before: str | None = None) -> list[dict[str, Any]]:
        """列出指定时间之前的待处理运行。"""

    @abc.abstractmethod
    async def list_inflight(self, *, before: str | None = None) -> list[dict[str, Any]]:
        """返回仍处于 ``pending`` 或 ``running`` 状态的已持久化运行。"""

    @abc.abstractmethod
    async def aggregate_tokens_by_thread(
        self, thread_id: str, *, include_active: bool = False
    ) -> dict[str, Any]:
        """聚合指定线程中已完成运行的 token 用量。

        返回字典包含：
        total_tokens, total_input_tokens, total_output_tokens, total_runs,
        by_model（model_name → {tokens, runs}）, by_caller（lead_agent/subagent/middleware）。
        """

    @abc.abstractmethod
    async def list(
        self,
        *,
        user_id: str | None = None,
        status: str | None = None,
        thread_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """通用运行列表：按用户 / 状态 / 线程过滤，创建时间倒序分页。"""
