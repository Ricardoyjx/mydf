"""Milvus 向量存储抽象基类。

定义统一的向量读写接口，使用方（如 MemoryMiddleware）不依赖具体的 Milvus 客户端实现。
"""

from __future__ import annotations

import abc
from typing import Any, Protocol


class EmbeddingFn(Protocol):
    """嵌入函数类型签名：接收文本，返回向量列表（float）。"""

    async def __call__(self, text: str) -> list[float]: ...


class SearchResult:
    """单条向量搜索结果。"""

    def __init__(
        self,
        id: int,
        score: float,
        text: str,
        content_type: str,
        agent_name: str,
        metadata: dict[str, Any] | None = None,
        timestamp: str = "",
    ) -> None:
        self.id = id
        self.score = score  # 相似度分数（越大越相似）
        self.text = text
        self.content_type = content_type
        self.agent_name = agent_name
        self.metadata = metadata or {}
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.id}, score={self.score:.4f}, "
            f"type={self.content_type}, text={self.text[:40]}...)"
        )


class MilvusStorage(abc.ABC):
    """向量存储抽象接口。

    所有方法均为 async，Embedding 由外部通过 ``embed_fn`` 参数传入，
    实现 Embedding 模型与向量存储的解耦。
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """连接到 Milvus 服务。

        在 lifespan 初始化时调用一次。实现类应在此方法中完成
        ``pymilvus.connections.connect()`` 调用。
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """关闭 Milvus 连接，释放资源。

        在 lifespan 拆除时调用。
        """
        ...

    @abc.abstractmethod
    async def ensure_collection(self, user_id: str) -> None:
        """确保指定用户的集合存在。

        如果集合不存在则自动创建（含 Schema 和索引定义）。
        如果已存在则跳过。

        参数：
            user_id: 用户标识。集合名称为 ``{config.prefix}_{user_id}``。
        """
        ...

    @abc.abstractmethod
    async def insert(
        self,
        user_id: str,
        agent_name: str,
        text: str,
        vector: list[float],
        content_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """插入一条向量记录。

        参数：
            user_id:      用户标识。
            agent_name:   代理名称（如 "lead-agent"）。
            text:         原始文本内容。
            vector:       文本对应的嵌入向量。
            content_type: 内容类型（conversation / fact / memory_summary）。
            metadata:     附加元数据（JSON 可序列化）。

        返回：
            插入记录的 ID（Milvus 自动生成）。
        """
        ...

    @abc.abstractmethod
    async def search(
        self,
        user_id: str,
        query_vector: list[float],
        top_k: int = 5,
        agent_name: str | None = None,
        content_type: str | None = None,
    ) -> list[SearchResult]:
        """向量相似度搜索。

        参数：
            user_id:       用户标识（必填，用于隔离搜索范围）。
            query_vector:  查询向量。
            top_k:         返回最相似的 top_k 条结果。
            agent_name:    可选，按 agent 名称过滤。
            content_type:  可选，按内容类型过滤。

        返回：
            SearchResult 列表，按相似度降序排列。
        """
        ...

    @abc.abstractmethod
    async def delete_by_filter(self, user_id: str, expr: str) -> int:
        """按表达式删除记录。

        参数：
            user_id: 用户标识。
            expr:    Milvus 布尔表达式，如 ``agent_name == "lead-agent"``。

        返回：
            被删除的记录数。
        """
        ...

    @abc.abstractmethod
    async def count(self, user_id: str) -> int:
        """统计指定用户集合中的记录总数。"""
        ...

    @abc.abstractmethod
    async def drop_collection(self, user_id: str) -> None:
        """删除指定用户的整个集合（谨慎使用）。"""
        ...
