"""动态上下文注入中间件：每次模型调用前注入当前日期时间。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, Runtime

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState

logger = logging.getLogger(__name__)


class DynamicContextMiddleware(AgentMiddleware):
    """在每次模型调用前，将当前日期作为 <system-reminder> 注入到首条 HumanMessage
    中，保持 system prompt 完全静态以最大化 prefix-cache 复用。

    通过 ``agent_name`` 区分不同代理实例，``app_config`` 保留供后续读取应用级
    配置（如时区格式、记忆开关等）。
    """

    def __init__(
        self,
        agent_name: str | None = None,
        app_config: Any | None = None,
    ) -> None:
        self._agent_name = agent_name
        self._app_config = app_config

    @property
    def name(self) -> str:
        """返回中间件名称（含 agent 标识）。"""
        return f"DynamicContextMiddleware({self._agent_name or 'default'})"

    @staticmethod
    def _build_reminder() -> str:
        """构造包含当前 UTC 时间的 <system-reminder> XML 块。"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"<system-reminder>\n当前日期 (UTC): {now}\n</system-reminder>"

    @staticmethod
    def _inject_into_first_human(
        messages: list[Any],
        reminder: str,
    ) -> list[Any] | None:
        """将 reminder 注入到列表中首条 HumanMessage 的 content 前。

        返回：
            更新后的 messages 列表；若没有 HumanMessage 则返回 None。
        """
        for msg in messages:
            if getattr(msg, "type", None) != "human":
                continue
            original = msg.content or ""
            if isinstance(original, str):
                msg.content = f"{reminder}\n\n{original}"
            elif isinstance(original, list):
                # 多模态消息（文本 + 图片等）
                msg.content = [
                    {"type": "text", "text": reminder},
                    *original,
                ]
            return messages
        return None

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """同步 hook：模型调用前注入动态上下文。"""
        messages = state.get("messages")
        if not messages:
            return None

        reminder = self._build_reminder()
        result = self._inject_into_first_human(messages, reminder)
        if result is not None:
            return {"messages": result}
        return None

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """异步 hook：委托给同步实现。"""
        return self.before_model(state, runtime)
