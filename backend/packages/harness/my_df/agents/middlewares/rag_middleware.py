"""RagMiddleware：在模型调用前检索用户知识库并注入 <rag_context>。

与 MemoryMiddleware 的分工：
- MemoryMiddleware 负责对话记忆与跨会话语义记忆（content_type="conversation"）；
- RagMiddleware 只检索上传的知识库文档（content_type="knowledge"）。
"""

from __future__ import annotations

import logging
from html import escape
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, Runtime

from my_df.agents.middlewares._injection import (
    get_latest_human_text,
    inject_block_into_first_human,
)
from my_df.runtime.milvus.base import MilvusStorage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState

logger = logging.getLogger(__name__)


def _format_rag_context(
    results: list[Any],
    min_score: float = 0.3,
) -> str:
    """把知识库检索结果格式化为 <rag_context> XML 块。"""
    lines = ["<rag_context>"]
    for result in results:
        score = getattr(result, "score", 0.0)
        if score < min_score:
            continue

        metadata = getattr(result, "metadata", {}) or {}
        title = escape(str(metadata.get("title") or "知识库"), quote=True)
        source = escape(str(metadata.get("source") or ""), quote=True)
        text = (
            str(getattr(result, "text", "") or "")[:800]
            .replace("\n", " ")
            .replace("\r", "")
        )
        lines.append(
            f'  <document title="{title}" source="{source}" score="{score:.2f}">'
        )
        lines.append(f"    <content>{escape(text, quote=True)}</content>")
        lines.append("  </document>")
    lines.append("</rag_context>")
    return "\n".join(lines) if len(lines) > 2 else ""


class RagMiddleware(AgentMiddleware):
    """知识库 RAG 中间件，仅在 Milvus 与 Embedding 可用时执行。"""

    def __init__(
        self,
        agent_name: str | None = None,
        user_id: str = "default",
        milvus: MilvusStorage | None = None,
        embedding_model: Any | None = None,
        knowledge_service: Any | None = None,
        *,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> None:
        self._agent_name = (agent_name or "").replace("_", "-") or None
        self._user_id = user_id
        self._milvus = milvus
        self._embedding = embedding_model
        self._knowledge_service = knowledge_service
        self._top_k = top_k
        self._min_score = min_score

    @property
    def name(self) -> str:
        return f"RagMiddleware({self._agent_name or 'default'})"

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """对用户最新消息做语义检索并注入知识库上下文。"""
        if self._milvus is None or self._embedding is None:
            return None

        messages = state.get("messages")
        if not messages:
            return None

        user_text = get_latest_human_text(messages)
        if not user_text:
            return None

        try:
            query_vector = await self._embedding.encode(user_text)
            results = await self._milvus.search(
                user_id=self._user_id,
                query_vector=query_vector,
                top_k=self._top_k,
                content_type="knowledge",
            )
            # 兼容旧版知识库
            if self._knowledge_service is not None:
                results = await self._knowledge_service.search(
                    user_id=self._user_id,
                    query=user_text,
                    top_k=self._top_k,
                    min_score=self._min_score,
                )
            else:
                query_vector = await self._embedding.encode(user_text)
                results = await self._milvus.search(
                    user_id=self._user_id,
                    query_vector=query_vector,
                    top_k=self._top_k,
                    content_type="knowledge",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("RAG 语义检索失败: %s", e)
            return None

        block = _format_rag_context(results, min_score=self._min_score)
        if not block:
            logger.info("RAG 未命中知识库（user=%s）", self._user_id)
            return None

        injected = inject_block_into_first_human(messages, block)
        if injected is None:
            return None
        logger.info(
            "RAG 已注入知识库上下文: user=%s, docs=%d, top_k=%d",
            self._user_id,
            len(results),
            self._top_k,
        )
        return {"messages": injected}
