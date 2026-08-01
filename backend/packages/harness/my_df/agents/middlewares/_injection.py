"""中间件通用的消息注入工具。

两个中间件（MemoryMiddleware / RagMiddleware）都需要：
- 提取最后一条 HumanMessage 的纯文本；
- 把系统上下文块注入到首条 HumanMessage。
"""

from __future__ import annotations

import re
from typing import Any

_SYSTEM_BLOCKS_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>|"
    r"<memory_context>.*?</memory_context>|"
    r"<semantic_memory>.*?</semantic_memory>|"
    r"<rag_context>.*?</rag_context>",
    re.DOTALL,
)


def get_latest_human_text(messages: list[Any]) -> str | None:
    """提取最后一条 HumanMessage 的纯文本（去除系统注入块）。

    返回 cleaned text；若无可用的 HumanMessage 则返回 None。
    """
    for msg in reversed(messages):
        if getattr(msg, "type", "") != "human":
            continue
        raw = str(getattr(msg, "content", ""))
        cleaned = _SYSTEM_BLOCKS_RE.sub("", raw).strip()
        return cleaned if cleaned else None
    return None


def inject_block_into_first_human(
    messages: list[Any],
    block: str,
) -> list[Any] | None:
    """将 block 注入到列表中首条 HumanMessage 的 content 前。

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
