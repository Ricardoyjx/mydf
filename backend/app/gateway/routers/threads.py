"""线程管理路由：查询对话历史消息。"""

import logging
import re

from app.gateway.deps import get_checkpointer
from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import message_to_dict
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/threads", tags=["threads"])


@router.get("/{thread_id}/messages")
async def get_thread_messages(thread_id: str, request: Request):
    """获取指定线程的完整对话历史。

    从 checkpointer 读取该线程的最新 checkpoint，
    提取 messages 字段并序列化为 JSON 返回。

    如果线程不存在返回 404。
    """
    checkpointer = get_checkpointer(request)
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="Checkpointer not available")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        # 获取最新 checkpoint
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception as e:  # noqa: BLE001
        logger.warning("读取 checkpoint 失败 (thread=%s): %s", thread_id, e)
        raise HTTPException(status_code=500, detail=f"读取历史失败: {e}")

    if checkpoint_tuple is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # 从 checkpoint 中提取 messages
    checkpoint = checkpoint_tuple.checkpoint
    # InMemorySaver 的 checkpoint 是 TypedDict，channel_values 存实际 state
    channel_values = checkpoint.get("channel_values") or checkpoint.get("values", {})
    messages = channel_values.get("messages", [])

    if not messages:
        return {"thread_id": thread_id, "messages": []}

    # 清洗 content：去掉中间件注入的系统标签
    _SYSTEM_BLOCKS_RE = re.compile(
        r"<system-reminder>.*?</system-reminder>|"
        r"<memory_context>.*?</memory_context>",
        re.DOTALL,
    )

    def _clean_content(raw: str) -> str:
        """去掉 <system-reminder> 和 <memory_context> 等注入内容，保留纯对话文本。"""
        cleaned = _SYSTEM_BLOCKS_RE.sub("", raw).strip()
        return cleaned

    # 序列化 BaseMessage 对象为可 JSON 序列化的 dict
    serialized = []
    for msg in messages:
        try:
            d = message_to_dict(msg)
            serialized.append(
                {
                    "role": d.get("type", "unknown"),
                    "content": _clean_content(
                        d.get("data", {}).get(
                            "content",
                            str(msg.content) if hasattr(msg, "content") else "",
                        )
                    ),
                    "id": d.get("id", ""),
                }
            )
        except Exception as e:  # noqa: BLE001
            # 回退：直接读 .content 属性
            raw = str(getattr(msg, "content", ""))
            role = str(getattr(msg, "type", "unknown"))
            serialized.append({"role": role, "content": _clean_content(raw), "id": ""})
            logger.warning("序列化消息失败 (thread=%s): %s", thread_id, e)
    return {
        "thread_id": thread_id,
        "messages": serialized,
        "checkpoint_ts": checkpoint.get("ts", ""),
    }
