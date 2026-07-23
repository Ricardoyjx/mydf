"""MemoryMiddleware：在模型调用前从持久化存储加载记忆并注入上下文。

工作流程：
1. ``before_model``：从 FileMemoryStorage 读取 memory → 格式化为 <memory_context> XML 块
   → 注入到首条 HumanMessage 前，让模型感知用户背景和历史。
2. ``after_model``：将当前对话摘要写回存储，更新 lastUpdated。

依赖：
- my_df.agents.memory.storage.get_memory_storage() — 存储后端
- my_df.agents.memory.storage.utc_now_iso_z() — 时间戳
- user_id 通过 ``config.configurable.user_id`` 传入，默认 "default"。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, Runtime
from my_df.agents.memory.storage import get_memory_storage, utc_now_iso_z

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState

logger = logging.getLogger(__name__)


def _has_content(memory: dict[str, Any]) -> bool:
    """检查 memory 是否有实质内容（非空结构）。

    遍历 ``user`` 和 ``history`` 各字段的 ``summary``，
    以及 ``facts`` 列表是否有条目。
    """
    for section_key in ("user", "history"):
        section = memory.get(section_key, {})
        if isinstance(section, dict):
            for field in section.values():
                if isinstance(field, dict) and field.get("summary"):
                    return True
    return bool(memory.get("facts"))


def _format_memory_block(memory: dict[str, Any]) -> str:
    """将 memory dict 格式化为 <memory_context> XML 块。

    保留结构信息便于 LLM 理解不同层级的上下文。
    """
    lines: list[str] = ["<memory_context>"]

    # —— 用户上下文 ——
    user = memory.get("user", {})
    for key, tag in [
        ("workContext", "work_context"),
        ("personalContext", "personal_context"),
        ("topOfMind", "top_of_mind"),
    ]:
        summary = user.get(key, {}).get("summary", "")
        if summary:
            lines.append(f"  <{tag}>{summary}</{tag}>")

    # —— 历史上下文 ——
    history = memory.get("history", {})
    for key, tag in [
        ("recentMonths", "recent_history"),
        ("earlierContext", "earlier_context"),
        ("longTermBackground", "long_term_background"),
    ]:
        summary = history.get(key, {}).get("summary", "")
        if summary:
            lines.append(f"  <{tag}>{summary}</{tag}>")

    # —— 事实列表 ——
    for fact in memory.get("facts", []):
        content = fact.get("content", "")
        if content:
            lines.append(f"  <fact>{content}</fact>")

    lines.append("</memory_context>")
    return "\n".join(lines)


def _inject_into_first_human(
    messages: list[Any],
    block: str,
) -> list[Any] | None:
    """将 memory block 注入到列表中首条 HumanMessage 的 content 前。

    返回：
        更新后的 messages 列表；若没有 HumanMessage 则返回 None。
    """
    for msg in messages:
        if getattr(msg, "type", None) != "human":
            continue
        original = msg.content or ""
        if isinstance(original, str):
            msg.content = f"{block}\n\n{original}"
        elif isinstance(original, list):
            msg.content = [{"type": "text", "text": block}, *original]
        return messages
    return None


def _extract_conversation_summary(messages: list[Any], max_turns: int = 20) -> str:
    """从对话记录中提取 user ↔ assistant 交换摘要。

    自动过滤中间件注入的系统内容（<system-reminder>、<memory_context>），
    只保留真实的用户消息和 AI 回复。
    取最后 *max_turns* 轮（默认 20 轮，覆盖较长对话）。
    """
    import re

    _SYSTEM_BLOCKS_RE = re.compile(
        r"<system-reminder>.*?</system-reminder>|"
        r"<memory_context>.*?</memory_context>",
        re.DOTALL,
    )

    turns: list[str] = []
    for msg in messages[-max_turns * 2 :]:
        role = getattr(msg, "type", "")
        raw = str(getattr(msg, "content", ""))

        # 跳过纯系统/中间件消息
        if role not in ("human", "ai"):
            continue
        # 检查 hide_from_ui 标记
        kw = getattr(msg, "additional_kwargs", {}) or {}
        if kw.get("hide_from_ui"):
            continue

        # 去除注入的系统块
        cleaned = _SYSTEM_BLOCKS_RE.sub("", raw).strip()
        if not cleaned:
            continue

        # 截取合理长度，去换行
        preview = cleaned[:2000].replace("\n", " ").replace("\r", "")
        turns.append(f"[{role}] {preview}")
    return "\n".join(turns)


class MemoryMiddleware(AgentMiddleware):
    """在模型调用前后同步记忆上下文。

    - ``before_model``：从存储加载 memory，注入到首条 HumanMessage。
    - ``after_model``：将最近对话写入 memory 的 history.recentMonths。
    """

    def __init__(
        self,
        agent_name: str | None = None,
        user_id: str = "default",
    ) -> None:
        self._agent_name = (agent_name or "").replace("_", "-") or None
        self._user_id = user_id
        self._storage = get_memory_storage()
        logger.info(
            "MemoryMiddleware 初始化: agent=%s, user=%s",
            agent_name,
            user_id,
        )

    # ── property ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"MemoryMiddleware({self._agent_name or 'default'})"

    # ── before_model：注入记忆 ────────────────────────────────────────────

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """模型调用前：加载 memory → 格式化 → 注入首条 HumanMessage。"""
        messages = state.get("messages")
        if not messages:
            return None

        try:
            memory = self._storage.load(
                agent_name=self._agent_name, user_id=self._user_id
            )
        except Exception as e:
            logger.warning("读取 memory 失败: %s", e)
            return None

        if not memory or not _has_content(memory):
            logger.debug("memory 为空，跳过注入")
            return None

        block = _format_memory_block(memory)
        result = _inject_into_first_human(messages, block)
        if result is not None:
            logger.debug("已注入 memory 上下文到首条 HumanMessage")
            return {"messages": result}
        return None

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """异步 hook：委托给同步实现。"""
        return self.before_model(state, runtime)

    # ── after_model：回写记忆 ─────────────────────────────────────────────

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """模型调用后：将最近对话写入 memory 存储。"""
        messages = state.get("messages")
        if not messages:
            return None

        try:
            # 读取当前 memory（保留已有内容）
            memory = self._storage.load(
                agent_name=self._agent_name, user_id=self._user_id
            )
            if not memory:
                return None

            # 更新历史上下文中的 recentMonths（最近对话摘要）
            summary = _extract_conversation_summary(messages)
            if "history" not in memory:
                memory["history"] = {}
            memory["history"]["recentMonths"] = {
                "summary": summary,
                "updatedAt": utc_now_iso_z(),
            }
            memory["lastUpdated"] = utc_now_iso_z()

            self._storage.save(
                memory, agent_name=self._agent_name, user_id=self._user_id
            )
            logger.debug("memory 已回写 (user=%s)", self._user_id)
        except Exception as e:
            logger.warning("回写 memory 失败: %s", e)

        return None

    async def abefore_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> None:
        """Agent 开始前 hook（当前无操作）。"""
        pass

    async def aafter_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """异步 hook：委托给同步实现。"""
        return self.after_model(state, runtime)
