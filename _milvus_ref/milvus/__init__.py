"""Milvus 向量存储模块。

为 my-df 提供基于 Milvus 的向量化记忆检索能力。
当前用于替代 ``MemoryMiddleware`` 中的全文 memory 注入，
通过语义相似度搜索找到最相关的历史记忆。

模块结构：
    base.py            抽象基类 ``MilvusStorage`` + ``SearchResult``
    client.py          pymilvus 实现 ``PymilvusStorage``
    async_provider.py  Async context manager 工厂 ``make_milvus_storage``

用法示例：
    from my_df.runtime.milvus.async_provider import make_milvus_storage

    async with make_milvus_storage() as milvus:
        await milvus.ensure_collection("user_001")
        results = await milvus.search("user_001", query_vector, top_k=3)
"""

from my_df.runtime.milvus.async_provider import make_milvus_storage
from my_df.runtime.milvus.base import MilvusStorage, SearchResult
from my_df.runtime.milvus.client import PymilvusStorage

__all__ = [
    "MilvusStorage",
    "PymilvusStorage",
    "SearchResult",
    "make_milvus_storage",
]
