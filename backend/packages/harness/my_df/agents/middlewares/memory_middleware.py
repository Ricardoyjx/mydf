"""MemoryMiddleware：在模型调用前从持久化存储和向量检索加载记忆并注入上下文。

工作流程：
1. abefore_model：从 FileMemoryStorage 读取 memory → 格式化为 <memory_context> XML 块；
   再调用 Embedding 模型对用户最新消息编码 → 在 Milvus 中检索对话记忆 →
   将 top-k 相关记忆格式化为 <semantic_memory> XML 块 → 一并注入首条 HumanMessage。
2. aafter_model：将当前对话摘要写回 JSON 存储，并把摘要向量化后写入 Milvus。

依赖：
- my_df.agents.memory.storage.get_memory_storage() — 存储后端
- my_df.agents.memory.storage.utc_now_iso_z() — 时间戳
- user_id 通过 config.configurable.user_id 传入，默认 "default"。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, Runtime

from my_df.agents.memory.storage import get_memory_storage, utc_now_iso_z
from my_df.agents.middlewares._injection import (
    get_latest_human_text as _get_latest_user_text,
)
from my_df.agents.middlewares._injection import (
    inject_block_into_first_human as _inject_into_first_human,
)
from my_df.runtime.milvus.base import MilvusStorage

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


def _extract_conversation_summary(messages: list[Any], max_turns: int = 20) -> str:
    """从对话记录中提取 user ↔ assistant 交换摘要。

    自动过滤中间件注入的系统内容（<system-reminder>、<memory_context>），
    只保留真实的用户消息和 AI 回复。
    取最后 *max_turns* 轮（默认 20 轮，覆盖较长对话）。
    """
    import re

    _SYSTEM_BLOCKS_RE = re.compile(
        r"<system-reminder>.*?</system-reminder>|"
        r"<memory_context>.*?</memory_context>|"
        r"<semantic_memory>.*?</semantic_memory>|"
        r"<rag_context>.*?</rag_context>",
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


def _extract_facts(
    messages: list[Any], max_facts: int = 10, thread_id: str = ""
) -> list[dict[str, Any]]:
    """从用户最新消息中提取结构化事实。

    匹配用户自我披露的信息（我叫、我喜欢、我是、我有 等关键词），
    忽略 AI 回复中的解释性内容。无匹配时返回空列表。
    """
    if not messages:
        return []

    # 取最后一条用户消息
    user_msg = None
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            user_msg = msg
            break
    if not user_msg:
        return []

    raw = str(getattr(user_msg, "content", ""))
    cleaned = re.sub(
        r"<system-reminder>.*?</system-reminder>|"
        r"<memory_context>.*?</memory_context>|"
        r"<semantic_memory>.*?</semantic_memory>|"
        r"<rag_context>.*?</rag_context>",
        "",
        raw,
        flags=re.DOTALL,
    ).strip()
    if not cleaned:
        return []

    now = utc_now_iso_z()
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 从用户消息中提取事实的模式
    patterns = [
        (r"我(?:叫|是)\s*(.+?)(?:[。，；!！?？]|$)", "identity", 0.8),
        (
            r"我(?:的\s*)?(?:名字|姓名|名称)\s*(?:是|叫)\s*(.+?)(?:[。，；!！?？]|$)",
            "identity",
            0.9,
        ),
        (r"我(?:喜欢|热爱|钟情|偏爱)\s*(.+?)(?:[。，；!！?？]|$)", "preference", 0.8),
        (r"我(?:想|要|希望|想要|打算)\s*(.+?)(?:[。，；!！?？]|$)", "preference", 0.6),
        (r"我(?:是|做|从事)\s*(.+?)(?:[。，；!！?？]|$)", "attribute", 0.6),
        (r"我有\s*(.+?)(?:[。，；!！?？]|$)", "attribute", 0.5),
        (r"我住在\s*(.+?)(?:[。，；!！?？]|$)", "attribute", 0.8),
        (r"我(?:的\s*)?(.+?)\s*是\s*(.+?)(?:[。，；!！?？]|$)", "attribute", 0.7),
    ]

    for pattern, category, confidence in patterns:
        for match in re.finditer(pattern, cleaned):
            content = match.group(0).strip()
            if not content or content in seen:
                continue
            seen.add(content)

            fact_id = f"fact_{int(datetime.now().timestamp())}_{len(facts)}"  # noqa: DTZ005
            facts.append(
                {
                    "id": fact_id,
                    "content": content,
                    "category": category,
                    "confidence": confidence,
                    "createdAt": now,
                    "source": thread_id,
                }
            )
            if len(facts) >= max_facts:
                return facts

    return facts


def _format_search_results(results: list[Any]) -> str:
    """将 Milvus 搜索结果格式化为 <semantic_memory> XML 块。

    只保留 score >= 0.3 的结果，避免低质量噪音。
    """
    lines = ["<semantic_memory>"]
    for r in results:
        score = getattr(r, "score", 0.0)
        if score < 0.3:
            continue
        text = (getattr(r, "text", "") or "")[:500].replace("\n", " ").replace("\r", "")
        lines.append(f'  <result score="{score:.2f}">')
        lines.append(f"    <content>{text}</content>")
        lines.append("  </result>")
    lines.append("</semantic_memory>")
    return "\n".join(lines)


class MemoryMiddleware(AgentMiddleware):
    """在模型调用前后异步同步记忆上下文。

    - ``abefore_model``：从存储加载 memory + 检索对话语义记忆，注入到首条 HumanMessage。
    - ``aafter_model``：将最近对话写入 memory 的 history.recentMonths。
    """

    def __init__(
        self,
        agent_name: str | None = None,
        user_id: str = "default",
        milvus: MilvusStorage | None = None,
        embedding_model: Any | None = None,
    ) -> None:
        self._agent_name = (agent_name or "").replace("_", "-") or None
        self._user_id = user_id
        self._storage = get_memory_storage()
        self._milvus = milvus  # Milvus 向量存储实例（可选）
        self._embedding = embedding_model  # Embedding 模型（可选）
        logger.info(
            "MemoryMiddleware 初始化: agent=%s, user=%s, milvus=%s",
            agent_name,
            user_id,
            "yes" if milvus else "no",
        )

    # ── property ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"MemoryMiddleware({self._agent_name or 'default'})"

    # ── before_model：注入记忆 ────────────────────────────────────────────

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """异步 hook：JSON 记忆注入 + Milvus 语义检索 + 合并注入。

        步骤：
        1. 从 FileMemoryStorage 加载持久化记忆 → 格式化为 <memory_context> 块
        2. 用 Embedding 模型对用户最新消息编码 → Milvus search → 格式化为 <semantic_memory> 块
        3. 合并两个块注入到首条 HumanMessage
        """
        logger.info(
            "abefore_model 被调用, messages=%d, milvus=%s, embedding=%s",
            len(state.get("messages", []) or []),
            self._milvus is not None,
            self._embedding is not None,
        )
        messages = state.get("messages")
        if not messages:
            return None

        # — 第 1 步：JSON 文件记忆 —
        block_parts: list[str] = []
        try:
            memory = self._storage.load(
                agent_name=self._agent_name, user_id=self._user_id
            )
            if memory and _has_content(memory):
                block_parts.append(_format_memory_block(memory))
        except Exception as e:
            logger.warning("读取 memory 失败: %s", e)

        # — 第 2 步：Milvus 语义检索 —
        if self._milvus is not None and self._embedding is not None:
            logger.info(
                "语义检索就绪: milvus=%s, embedding=%s, user=%s, agent=%s",
                type(self._milvus).__name__,
                type(self._embedding).__name__,
                self._user_id,
                self._agent_name,
            )
            user_text = _get_latest_user_text(messages)
            if user_text:
                try:
                    query_vec = await self._embedding.encode(user_text)
                    results = await self._milvus.search(
                        user_id=self._user_id,
                        query_vector=query_vec,
                        top_k=10,
                        agent_name=self._agent_name,
                        content_type="conversation",
                    )
                    if results:
                        block_parts.append(_format_search_results(results))
                        logger.info("语义检索返回 %d 条相关记忆", len(results))
                except Exception as search_err:
                    logger.warning("语义检索失败: %s", search_err)

        logger.info(
            "合并注入前: block_parts=%d, milvus=%s, embedding=%s",
            len(block_parts),
            self._milvus is not None,
            self._embedding is not None,
        )

        # — 第 3 步：合并注入 —
        if not block_parts:
            logger.info("无记忆内容，跳过注入")
            return None

        combined = "\n\n".join(block_parts)
        result = _inject_into_first_human(messages, combined)
        if result is not None:
            logger.info(
                "已注入 memory 上下文到首条 HumanMessage (块数=%d, 含语义检索=%s)",
                len(block_parts),
                self._milvus is not None and self._embedding is not None,
            )
            return {"messages": result}
        return None

    def _save_json_memory(self, messages: list[Any]) -> dict[str, Any] | None:
        """将最近对话写入 JSON memory 存储（同步）。"""
        if not messages:
            return None

        try:
            memory = self._storage.load(
                agent_name=self._agent_name, user_id=self._user_id
            )
            if not memory:
                return None

            summary = _extract_conversation_summary(messages)
            if "history" not in memory:
                memory["history"] = {}
            memory["history"]["recentMonths"] = {
                "summary": summary,
                "updatedAt": utc_now_iso_z(),
            }

            max_facts = getattr(get_memory_storage(), "max_facts", 100)
            new_facts = _extract_facts(messages, max_facts)
            existing_facts = memory.get("facts", [])
            existing_ids = {f["id"] for f in existing_facts}
            for f in new_facts:
                if f["id"] not in existing_ids:
                    existing_facts.append(f)
            memory["facts"] = existing_facts[:max_facts]

            memory["lastUpdated"] = utc_now_iso_z()

            self._storage.save(
                memory, agent_name=self._agent_name, user_id=self._user_id
            )
            logger.debug(
                "memory 已回写 (user=%s), facts=%d", self._user_id, len(new_facts)
            )
        except Exception as e:
            logger.warning("回写 memory 失败: %s", e)

        return None

    async def abefore_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> None:
        """Agent 开始前 hook（当前无操作）。"""

    async def aafter_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """异步 hook：先执行同步回写，再尝试 Milvus 向量存储。"""
        # 1. 先执行同步回写（JSON 存储）
        messages = state.get("messages")
        sync_result = self._save_json_memory(messages)

        # 2. 将对话摘要向量化并存入 Milvus
        if self._milvus is not None:
            messages = state.get("messages")
            if messages:
                summary = _extract_conversation_summary(messages)
                if summary.strip():
                    thread_id = ""
                    try:
                        thread_id = str(
                            runtime.config.get("configurable", {}).get("thread_id", "")  # type: ignore[union-attr]
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning("无法获取 thread_id，使用空字符串作为默认值")

                    # 生成向量：优先使用真实 Embedding 模型，回退到 MD5 占位
                    if self._embedding is not None:
                        try:
                            vector = await self._embedding.encode(summary[:2000])
                        except Exception as emb_err:  # noqa: BLE001
                            logger.warning(
                                "Embedding 编码失败，跳过 Milvus 存储: %s", emb_err
                            )
                            return sync_result
                    else:
                        import hashlib

                        raw_hash = hashlib.md5(summary.encode()).digest()
                        vector = [
                            (raw_hash[i % 16] / 255.0) * 2 - 1
                            for i in range(self._milvus.vector_dim)
                        ]

                    try:
                        inserted_id = await self._milvus.insert(
                            user_id=self._user_id,
                            agent_name=self._agent_name or "default",
                            text=summary[:2000],
                            vector=vector,
                            content_type="conversation",
                            metadata={"thread_id": thread_id},
                        )
                        logger.info(
                            "Milvus 记忆已存储: id=%s, dim=%d, user=%s (embedding=%s)",
                            inserted_id,
                            len(vector),
                            self._user_id,
                            self._embedding is not None,
                        )
                    except Exception as mv_err:  # noqa: BLE001
                        logger.warning("Milvus 记忆存储失败: %s", mv_err)

        return sync_result
