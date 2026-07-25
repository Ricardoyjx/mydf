import json
import logging
from typing import Any

from my_df.config.milvus_config import MilvusConfig
from my_df.runtime.milvus.base import MilvusStorage, SearchResult
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

logger = logging.getLogger(__name__)


class PyMilvusStorage(MilvusStorage):
    """pymilvus 实现的 Milvus 向量存储。

    设计要点：
    - 集合以用户为粒度隔离（``{prefix}_{user_id}``），天然多租户
    - 索引类型、向量维度通过 MilvusConfig 配置，灵活切换
    - 所有公共方法均记录操作日志，便于排查
    """

    def __init__(self, config: MilvusConfig | None = None) -> None:
        self._config = config or MilvusConfig(alias="default")
        self._alias = self._config.alias
        self._connected = False
        logger.info(
            "初始化 MilvusStorage，host=%s, port = %s, dim=%s ,index= %s",
            self._config.host,
            self._config.port,
            self._config.vector_dim,
            self._config.index_type,
        )

    def _collection_name(self, user_id: str) -> str:
        """生成用户隔离的集合名称。"""
        # 替换 user_id 中的非法字符（Milvus 集合名只允许字母、数字、下划线）
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in user_id)
        return f"{self._config.collection_name_prefix}_{safe}"

    def _build_schema(self) -> CollectionSchema:
        """构建 Collection Schema。

        字段定义：
        - id (INT64, 主键, 自动生成)
        - vector (FLOAT_VECTOR, 可配置维度)
        - user_id (VARCHAR, 多租户过滤用)
        - agent_name (VARCHAR)
        - text (VARCHAR, 原始内容)
        - content_type (VARCHAR, 内容分类)
        - metadata (JSON)
        - timestamp (VARCHAR, ISO 格式)
        """

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="自增ID",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._config.vector_dim,
                description=f"文本嵌入向量（{self._config.vector_dim} 维）",
            ),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=128,
                description="用户标识（多租户隔离）",
            ),
            FieldSchema(
                name="agent_name",
                dtype=DataType.VARCHAR,
                max_length=128,
                description="代理名称",
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="原始文本内容",
            ),
            FieldSchema(
                name="content_type",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="内容类型: conversation / fact / memory_summary",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="附加元数据（JSON 对象）",
            ),
            FieldSchema(
                name="timestamp",
                dtype=DataType.VARCHAR,
                max_length=32,
                description="ISO 格式时间戳",
            ),
        ]

        return CollectionSchema(
            fields=fields, description="my_df 存储向量集合", enable_dynamic_field=False
        )

    def _build_index_params(self) -> dict[str, Any]:
        """根据配置构建索引参数。"""
        index_type = self._config.index_type

        params_map: dict[str, dict[str, Any]] = {
            "IVF_FLAT": {"nlist": self._config.nlist},
            "IVF_SQ8": {"nlist": self._config.nlist},
            "HNSW": {"M": 16, "efConstruction": 200},
        }
        index_params = params_map.get(index_type, {"nlist": self._config.nlist})

        return {
            "index_type": index_type,
            "metric_type": "IP",  # 内积相似度（归一化后等价于余弦）
            "params": index_params,
        }

    # ── 生命周期 ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        连接到 Milvus 服务。"""

        if self._connected:
            logger.debug("MilvusStorage 已经连接，无需重复连接。")
            return

        try:
            connections.connect(
                alias=self._alias,
                host=self._config.host,
                port=self._config.port,
            )
            self._connected = True
            logger.info(
                "成功连接到 Milvus 服务: host=%s, port = %s alias= %s",
                self._config.host,
                self._config.port,
                self._alias,
            )
        except Exception as e:
            logger.error("无法连接到 Milvus 服务: %s", e)
            raise

    async def close(self) -> None:
        """
        断开 Milvus 服务连接。"""

        if not self._connected:
            logger.debug("MilvusStorage 未连接，无需断开连接。")
            return

        try:
            connections.disconnect(self._alias)
            self._connected = False
            logger.info("已断开 Milvus 服务连接: alias= %s", self._alias)
        except Exception as e:  # noqa: BLE001
            logger.warning("无法断开 Milvus 服务连接: %s", e)

    # ── 集合管理 ──────────────────────────────────────────────────────

    async def ensure_collection(self, user_id: str) -> None:
        """确保用户集合已创建，含 Schema 和 IVF_FLAT 索引。

        如果集合已存在则跳过不覆盖，保留已有数据。
        """

        name = self._collection_name(user_id)
        try:
            if utility.has_collection(name, using=self._alias):
                logger.debug("用户集合已存在: %s", name)
                coll = Collection(name, using=self._alias)
                if not coll.has_index():
                    index_params = self._build_index_params()
                    coll.create_index(
                        field_name="vector",
                        index_params=index_params,
                    )  # type: ignore
                    logger.info("已创建索引: %s", name)
                    return

            # 创建合集
            schema = self._build_schema()
            coll = Collection(
                name=name,
                schema=schema,
                using=self._alias,
            )
            logger.info("已创建用户集合: %s", name)

            # 创建索引
            index_params = self._build_index_params()
            coll.create_index(
                field_name="vector",
                index_params=index_params,
            )  # type: ignore
            logger.info("已创建索引: %s", name)

            # 加载集合到内存
            coll.load()
            logger.info("已加载集合到内存: %s", name)

        except Exception as e:
            logger.error("无法创建用户集合:%s %s", name, e)
            raise

    async def drop_collection(self, user_id: str) -> None:
        """删除指定用户的集合（危险操作！）。"""
        name = self._collection_name(user_id)
        try:
            utility.drop_collection(name, using=self._alias)  # type: ignore
            logger.warning("集合已删除: %s", name)
        except Exception as e:
            logger.error("删除集合 %s 失败: %s", name, e)
            raise

    # ── 写入操作 ──────────────────────────────────────────────────────
    async def insert(
        self,
        user_id: str,
        agent_name: str,
        text: str,
        vector: list[float],
        content_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return 1

    # ── 搜索操作 ──────────────────────────────────────────────────────
    async def search(
        self,
        user_id: str,
        query_vector: list[float],
        top_k: int = 5,
        agent_name: str | None = None,
        content_type: str | None = None,
    ) -> list[SearchResult]:
        """向量相似度搜索，支持按 agent_name 和 content_type 过滤。

        搜索前加载集合，确保查询性能。
        """
        name = self._collection_name(user_id)
        coll = Collection(name, using=self._alias)
        coll.load()

        # 构建过滤表达式
        expr_parts = [f'user_id == "{user_id}"']
        if agent_name:
            expr_parts.append(f'agent_name == "{agent_name}"')
        if content_type:
            expr_parts.append(f'content_type == "{content_type}"')
        expr = " and ".join(expr_parts)

        try:
            results = coll.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=top_k,
                expr=expr,
                output_fields=[],
            )
        except Exception as e:
            logger.error("搜索失败: %s", e)
            raise

        parsed: list[SearchResult] = []
        for hits in results:  # type: ignore
            for hit in hits:
                text = hit.entity.get("text") or ""
                content_type_val = hit.entity.get("content_type") or ""
                agent_name_val = hit.entity.get("agent_name") or ""
                metadata_raw = hit.entity.get("metadata") or "{}"
                ts = hit.entity.get("timestamp") or ""

                # metadata 是字符串，需要反序列化
                try:
                    meta = (
                        json.loads(metadata_raw)
                        if isinstance(metadata_raw, str)
                        else {}
                    )
                except (json.JSONDecodeError, TypeError):
                    meta = {}

                parsed.append(
                    SearchResult(
                        id=hit.id,
                        score=hit.score,
                        text=text,
                        content_type=content_type_val,
                        agent_name=agent_name_val,
                        metadata=meta,
                        timestamp=ts,
                    )
                )
        logger.debug("向量搜索完成: user=%s, hits=%d", user_id, len(parsed))
        return parsed

    # ── 管理集合 ──────────────────────────────────────────────────────
    async def delete_by_filter(
        self,
        user_id: str,
        expr: str,
    ) -> int:
        return 1

    async def count(
        self,
        user_id: str,
    ) -> int:
        return 1
