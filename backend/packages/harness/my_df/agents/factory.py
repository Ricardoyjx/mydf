"""Agent 工厂函数：封装 langchain.agents.create_agent 调用。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)


def create_mydf_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: list[AgentMiddleware] | None = None,
    plan_mode: bool = False,
    state_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    name: str = "default",
) -> CompiledStateGraph:
    """创建 my-df Agent 的便捷工厂函数。

    参数：
        model:         聊天模型实例。
        tools:         用户提供的工具。特性注入的工具会自动追加。
        system_prompt: 系统消息。``None`` 使用最小默认值。
        middleware:    **完全接管** — 若提供则使用此精确列表。
                       不可与 *features* 或 *extra_middleware* 同时使用。
        plan_mode:     启用 TodoMiddleware 进行任务追踪。
        state_schema:  LangGraph 状态类型。默认 ``ThreadState``。
        checkpointer:  可选的持久化后端。
        name:          Agent 名称（传递给关心此信息的中间件，如 MemoryMiddleware）。

    抛出：
        ValueError: 同时提供了 *middleware* 和 *features*/*extra_middleware*。
    """

    effective_tools: list[BaseTool] = list(tools or [])
    effective_state = state_schema

    if middleware is not None:
        effective_middleware = list(middleware)

    return create_agent(
        model=model,
        tools=effective_tools or None,
        middleware=effective_middleware,
        system_prompt=system_prompt,
        state_schema=effective_state,
        checkpointer=checkpointer,
        name=name,
    )
