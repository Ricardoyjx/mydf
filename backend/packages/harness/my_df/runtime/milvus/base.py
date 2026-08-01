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

    @property
    @abc.abstractmethod
    def vector_dim(self) -> int:
        """返回向量维度。具体值由实现类（如 PyMilvusStorage）根据配置决定。"""
        ...

    @abc.abstractmethod
    async def connect(self) -> None:
        """连接到 Milvus 服务。

        在 lifespan 初始化时调用一次。实现类应在此方法中完成
        ``pymilvus.connections.connect()`` 调用。
        """
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """关闭 Milvus 连接。

        在 lifespan 结束时调用一次。实现类应在此方法中完成
        ``pymilvus.connections.disconnect()`` 调用。
        """
        ...

    @abc.abstractmethod
    async def ensure_collection(self, user_id: str) -> None:
        """确保 Milvus 中存在对应的 Collection。

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
        """向量相似度搜索。

        参数：
            user_id:       用户标识（必填，用于隔离搜索范围）。
            query_vector:  查询向量。
            top_k:         返回最相似的 top_k 条结果。
            agent_name:    可选，按 agent 名称过滤。
            content_type:  可选，按内容类型过滤。

        返回：
            int。
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
    async def list_records(
        self,
        user_id: str,
        content_type: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SearchResult]:
        """按过滤条件列出向量记录（不含相似度分数）。

        参数：
            user_id:      用户标识（必填，用于隔离搜索范围）。
            content_type: 可选，按内容类型过滤。
            agent_name:   可选，按 agent 名称过滤。
            limit:        返回条数上限。
            offset:       跳过条数，用于分页。

        返回：
            SearchResult 列表，按插入顺序返回。
        """
        ...

    @abc.abstractmethod
    async def delete_by_ids(self, user_id: str, ids: list[int]) -> int:
        """按向量主键批量删除记录。

        参数：
            user_id: 用户标识（必填，用于隔离搜索范围）。
            ids:     待删除的向量主键列表。

        返回：
            删除的向量数量。
        """
        ...

    @abc.abstractmethod
    async def delete_by_filter(self, user_id: str, expr: str) -> int:
        """删除满足过滤条件的向量。

        参数：
            user_id: 用户标识（必填，用于隔离搜索范围）。
            expr:    过滤条件。

        返回：
            删除的向量数量。
        """
        ...

    @abc.abstractmethod
    async def count(self, user_id: str) -> int:
        """获取集合中向量的数量。

        参数：
            user_id: 用户标识（必填，用于隔离搜索范围）。

        返回：
            向量数量。
        """
        ...

    @abc.abstractmethod
    async def drop_collection(self, user_id: str) -> None:
        """删除集合。

        参数：
            user_id: 用户标识（必填，用于隔离搜索范围）。
        """
        ...
