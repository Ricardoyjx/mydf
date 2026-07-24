"""基于 pymilvus 的 MilvusStorage 实现。

负责：
- 连接管理（connect / close）
- Collection 自动创建（含 Schema + Index）
- 向量插入与批量插入
- 向量搜索（支持标量过滤）
- 按条件删除与统计

依赖：
    pymilvus>=2.4.0
    my_df.config.milvus_config.MilvusConfig

用法（通过 async_provider.py 工厂创建，不直接实例化）：
    client = PymilvusStorage(config)
    await client.connect()
    await client.ensure_collection("user_001")
    await client.insert("user_001", "lead-agent", "你好", [0.1, 0.2, ...])
    results = await client.search("user_001", [0.1, 0.2, ...], top_k=3)
    await client.close()
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from my_df.config.milvus_config import MilvusConfig
from my_df.runtime.milvus.base import MilvusStorage, SearchResult

logger = logging.getLogger(__name__)


# def _require_pymilvus() -> None:
#     """检查 pymilvus 是否已安装，未安装时抛出清晰的错误提示。"""
#     if _IMPORT_ERROR is not None:
#         raise ImportError(
#             "pymilvus 未安装。请执行以下命令安装：\n"
#             "    uv pip install pymilvus>=2.4.0\n"
#             "或确保 .venv 中已包含该依赖。"
#         ) from _IMPORT_ERROR


class PymilvusStorage(MilvusStorage):
    """pymilvus 实现的 Milvus 向量存储。

    设计要点：
    - 集合以用户为粒度隔离（``{prefix}_{user_id}``），天然多租户
    - 索引类型、向量维度通过 MilvusConfig 配置，灵活切换
    - 所有公共方法均记录操作日志，便于排查
    """

    def __init__(self, config: MilvusConfig | None = None) -> None:
        # _require_pymilvus()

        self._config = config or MilvusConfig()
        self._alias = self._config.alias
        self._connected = False
        logger.info(
            "PymilvusStorage 初始化: host=%s, port=%s, dim=%d, index=%s",
            self._config.host,
            self._config.port,
            self._config.vector_dim,
            self._config.index_type,
        )

    # ── 内部辅助 ──────────────────────────────────────────────────────

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
                description="自增主键",
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
            fields=fields,
            description=f"my-df 用户记忆集合（{self._config.vector_dim}d）",
            enable_dynamic_field=False,
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
        """连接到 Milvus 服务。

        使用配置中的 host:port 建立 gRPC 连接。
        连接已存在时跳过（幂等操作）。
        """
        if self._connected:
            logger.debug("Milvus 已连接，跳过")
            return

        try:
            connections.connect(
                alias=self._alias,
                host=self._config.host,
                port=self._config.port,
            )
            self._connected = True
            logger.info(
                "Milvus 连接成功: %s:%s (alias=%s)",
                self._config.host,
                self._config.port,
                self._alias,
            )
        except Exception as e:
            logger.error("Milvus 连接失败: %s", e)
            raise

    async def close(self) -> None:
        """断开 Milvus 连接。

        幂等操作，可多次调用。
        """
        if not self._connected:
            return
        try:
            connections.disconnect(alias=self._alias)
            self._connected = False
            logger.info("Milvus 连接已关闭 (alias=%s)", self._alias)
        except Exception as e:
            logger.warning("关闭 Milvus 连接时出错: %s", e)

    # ── 集合管理 ──────────────────────────────────────────────────────

    async def ensure_collection(self, user_id: str) -> None:
        """确保用户集合已创建，含 Schema 和 IVF_FLAT 索引。

        如果集合已存在则跳过不覆盖，保留已有数据。
        """
        name = self._collection_name(user_id)
        try:
            if utility.has_collection(name, using=self._alias):
                logger.debug("集合已存在: %s", name)
                # 确保在已有集合的 vector 字段上建立索引（首次连接时可能没有）
                coll = Collection(name, using=self._alias)
                if not coll.has_index():
                    index_params = self._build_index_params()
                    coll.create_index(
                        field_name="vector",
                        index_params=index_params,
                    )  # type: ignore
                    logger.info("已为已有集合 %s 创建索引", name)
                return

            # 创建集合
            schema = self._build_schema()
            coll = Collection(
                name=name,
                schema=schema,
                using=self._alias,
            )
            logger.info("集合已创建: %s (dim=%d)", name, self._config.vector_dim)

            # 创建索引
            index_params = self._build_index_params()
            coll.create_index(
                field_name="vector",
                index_params=index_params,
            )  # type: ignore
            logger.info(
                "索引已创建: %s, type=%s, metric=IP",
                name,
                self._config.index_type,
            )

            # 加载集合到内存
            coll.load()
            logger.info("集合已加载到内存: %s", name)

        except Exception as e:
            logger.error("创建集合 %s 失败: %s", name, e)
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
        """插入一条向量记录。

        返回 Milvus 自动生成的主键 ID（仅第一条，批量插入时通常只有一个）。
        """
        from datetime import UTC, datetime

        name = self._collection_name(user_id)
        coll = Collection(name, using=self._alias)

        timestamp = datetime.now(UTC).isoformat()

        entities = [
            [vector],  # vector
            [user_id],  # user_id
            [agent_name],  # agent_name
            [text],  # text
            [content_type],  # content_type
            [json.dumps(metadata or {})],  # metadata
            [timestamp],  # timestamp
        ]

        try:
            insert_result = coll.insert(entities)
            coll.flush()
            ids = insert_result.primary_keys
            inserted_id = int(ids[0]) if ids else 0
            logger.debug(
                "向量插入成功: id=%s, user=%s, type=%s, text_len=%d",
                inserted_id,
                user_id,
                content_type,
                len(text),
            )
            return inserted_id
        except Exception as e:
            logger.error("向量插入失败 (user=%s): %s", user_id, e)
            raise

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
                output_fields=[
                    "text",
                    "content_type",
                    "agent_name",
                    "metadata",
                    "timestamp",
                ],
            )
        except Exception as e:
            logger.error("向量搜索失败 (user=%s): %s", user_id, e)
            raise

        # 解析搜索结果
        parsed: list[SearchResult] = []
        for hits in results:
            for hit in hits:
                # hit.entity.get() 返回字段值（字符串或 None）
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

    # ── 管理操作 ──────────────────────────────────────────────────────

    async def delete_by_filter(self, user_id: str, expr: str) -> int:
        """按表达式删除记录。

        示例表达式：
            ``agent_name == "lead-agent" and content_type == "fact"``
            ``timestamp < "2026-01-01T00:00:00"``
        """
        name = self._collection_name(user_id)
        coll = Collection(name, using=self._alias)
        coll.load()

        try:
            delete_result = coll.delete(expr)
            coll.flush()
            deleted_count = (
                delete_result.delete_count
                if hasattr(delete_result, "delete_count")
                else -1
            )
            logger.info(
                "向量删除完成: user=%s, expr=%s, count=%s", user_id, expr, deleted_count
            )
            return deleted_count
        except Exception as e:
            logger.error("向量删除失败 (user=%s, expr=%s): %s", user_id, expr, e)
            raise

    async def count(self, user_id: str) -> int:
        """返回用户集合中的记录总数。"""
        name = self._collection_name(user_id)
        try:
            if not utility.has_collection(name, using=self._alias):
                return 0
            coll = Collection(name, using=self._alias)
            coll.load()
            return coll.num_entities
        except Exception as e:
            logger.warning("查询集合数量失败 (user=%s): %s", user_id, e)
            return 0
