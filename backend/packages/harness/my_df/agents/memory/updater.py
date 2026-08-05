from typing import Any

from my_df.agents.memory.storage import get_memory_storage
from my_df.agents.middlewares.memory_middleware import MemoryMiddleware
from langgraph.store.base import BaseStore


def get_memory_data(
    agent_name: str | None = None, *, user_id: str | None = None
) -> dict[str, Any]:
    """从文件存储读取记忆（回退路径，兼容旧调用）。"""
    return get_memory_storage().load(agent_name, user_id=user_id)


async def get_memory_data_async(
    store: BaseStore,
    *,
    user_id: str = "default",
    agent_name: str | None = "lead_agent",
) -> dict[str, Any]:
    """从 LangGraph Store 读取记忆（namespace=("user", user_id)）。

    与 MemoryMiddleware 使用同一 key 归一化规则，保证 API 与中间件数据源一致。
    Store 中无记录时返回空记忆结构。
    """
    key = MemoryMiddleware._memory_key(agent_name)
    item = await store.aget(("user", user_id), key)
    if item is None:
        from my_df.agents.memory.storage import create_empty_memory

        return create_empty_memory()
    return item.value
