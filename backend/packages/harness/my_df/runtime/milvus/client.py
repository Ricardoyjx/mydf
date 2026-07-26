"""基于 pymilvus MilvusClient（新版 API）的向量存储实现。

使用 ``pymilvus.MilvusClient`` 替代已废弃的 ORM 风格 API
（``connections.connect`` / ``Collection``），避免 PyMilvus 3.1 后的兼容问题。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pymilvus import (
    CollectionSchema,
    DataType,
    MilvusClient,
    MilvusException,
)

from my_df.config.milvus_config import MilvusConfig
from my_df.runtime.milvus.base import MilvusStorage, SearchResult

logger = logging.getLogger(__name__)


class PyMilvusStorage(MilvusStorage):
    """基于 ``MilvusClient`` 的向量存储。

    设计要点：
    - 集合以用户为粒度隔离（``{prefix}_{user_id}``），天然多租户
    - 索引类型、向量维度通过 MilvusConfig 配置
    - 所有公共方法均记录操作日志
    """

    def __init__(self, config: MilvusConfig | None = None) -> None:
        self._config = config or MilvusConfig()
        self._client: MilvusClient | None = None
        logger.info(
            "PyMilvusStorage 初始化: host=%s, port=%s, dim=%s, index=%s",
            self._config.host,
            self._config.port,
            self._config.vector_dim,
            self._config.index_type,
        )

    # ── 内部辅助 ──────────────────────────────────────────────────────

    def _collection_name(self, user_id: str) -> str:
        """生成用户隔离的集合名称。"""
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in user_id)
        return f"{self._config.collection_name_prefix}_{safe}"

    def _build_schema(self) -> CollectionSchema:
        """构建 Collection Schema (MilvusClient 兼容)."""
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._config.vector_dim,
        )
        schema.add_field(
            field_name="user_id", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(
            field_name="agent_name", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="content_type", datatype=DataType.VARCHAR, max_length=64
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(
            field_name="timestamp", datatype=DataType.VARCHAR, max_length=32
        )
        return schema

    def _build_index_params(self) -> Any:
        """构建索引参数。"""
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self._config.index_type,
            metric_type="IP",
            params={"nlist": self._config.nlist}
            if self._config.index_type != "HNSW"
            else {"M": 16, "efConstruction": 200},
        )
        return index_params

    def _ensure_client(self) -> MilvusClient:
        """返回当前客户端，如果未连接则抛出异常。"""
        if self._client is None:
            raise RuntimeError("MilvusClient 未连接，请先调用 connect()")
        return self._client

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def vector_dim(self) -> int:
        return self._config.vector_dim

    # ── 生命周期 ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        """连接到 Milvus 服务。

        使用指数退避重试，最多等待约 60 秒，应对 Milvus 首次启动时的延迟。
        """
        if self._client is not None:
            logger.debug("PyMilvusStorage 已连接，跳过")
            return

        import asyncio

        max_retries = 12
        base_delay = 1.0
        uri = f"tcp://{self._config.host}:{self._config.port}"

        for attempt in range(1, max_retries + 1):
            try:
                self._client = MilvusClient(uri=uri)
                # 验证连接：发送一个轻量请求
                colls = self._client.list_collections()
                logger.info(
                    "MilvusClient 连接成功: %s (第 %d 次尝试),集合：%s",
                    uri,
                    attempt,
                    colls,
                )
                return
            except Exception as e:  # noqa: BLE001
                delay = base_delay * (1.5 ** (attempt - 1))
                logger.warning(
                    "Milvus 连接失败（第 %d/%d 次）: %s，%.1f 秒后重试...",
                    attempt,
                    max_retries,
                    e,
                    delay,
                )
                self._client = None
                await asyncio.sleep(delay)

        logger.error("Milvus 服务连接超时，已重试 %d 次", max_retries)
        raise RuntimeError(f"无法连接到 Milvus: {uri}")

    async def close(self) -> None:
        """断开 Milvus 连接。"""
        if self._client is None:
            return
        try:
            self._client.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("关闭 MilvusClient 时出错: %s", e)
        finally:
            self._client = None
            logger.info("PyMilvusStorage 已关闭")

    # ── 集合管理 ──────────────────────────────────────────────────────

    async def ensure_collection(self, user_id: str) -> None:
        """确保用户集合已创建（含 Schema 和索引）。"""
        client = self._ensure_client()
        name = self._collection_name(user_id)

        try:
            if client.has_collection(name):
                # 检查维度是否匹配，不匹配则重建
                try:
                    existing_schema = client.describe_collection(name)
                    existing_dim = 0
                    for f in existing_schema.get("fields", []):  # type: ignore
                        if f.get("name") == "vector":
                            existing_dim = f.get("params", {}).get("dim", 0)
                            break
                    if existing_dim == self._config.vector_dim:
                        logger.debug("集合已存在且维度匹配: %s", name)
                        return
                    logger.warning(
                        "集合 %s 维度不匹配（现有=%d, 期望=%d），删除重建",
                        name,
                        existing_dim,
                        self._config.vector_dim,
                    )
                    client.drop_collection(name)
                except Exception as desc_err:
                    logger.warning("描述集合 %s 失败，尝试重建: %s", name, desc_err)
                    try:
                        client.drop_collection(name)
                    except Exception:
                        logger.error("删除集合 %s 失败: %s", name, desc_err)

            schema = self._build_schema()
            client.create_collection(
                collection_name=name,
                schema=schema,
            )
            logger.info("集合已创建: %s (dim=%d)", name, self._config.vector_dim)

            index_params = self._build_index_params()
            client.create_index(
                collection_name=name,
                index_params=index_params,
            )
            logger.info("索引已创建: %s, type=%s", name, self._config.index_type)
        except Exception as e:
            logger.error("创建集合 %s 失败: %s", name, e)
            raise

    async def drop_collection(self, user_id: str) -> None:
        """删除指定用户的集合（危险操作！）。"""
        client = self._ensure_client()
        name = self._collection_name(user_id)
        try:
            client.drop_collection(name)
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
        client = self._ensure_client()
        name = self._collection_name(user_id)

        from datetime import UTC, datetime

        data = {
            "vector": vector,
            "user_id": user_id,
            "agent_name": agent_name,
            "text": text,
            "content_type": content_type,
            "metadata": json.dumps(metadata or {}),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            res = client.insert(collection_name=name, data=data)
            inserted_id = res.get("ids", [0])[0] if isinstance(res, dict) else 0
            logger.debug(
                "向量插入成功: id=%s, user=%s, type=%s",
                inserted_id,
                user_id,
                content_type,
            )
            return int(inserted_id)
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
        client = self._ensure_client()
        name = self._collection_name(user_id)

        # 构建过滤表达式
        expr_parts = [f'user_id == "{user_id}"']
        if agent_name:
            expr_parts.append(f'agent_name == "{agent_name}"')
        if content_type:
            expr_parts.append(f'content_type == "{content_type}"')
        expr = " and ".join(expr_parts)

        try:
            results = client.search(
                collection_name=name,
                data=[query_vector],
                limit=top_k,
                search_params={"metric_type": "IP", "params": {"nprobe": 10}},
                filter=expr,
                output_fields=[
                    "text",
                    "content_type",
                    "agent_name",
                    "metadata",
                    "timestamp",
                ],
            )
        except MilvusException as e:
            logger.error("向量搜索失败 (user=%s): %s", user_id, e)
            raise

        parsed: list[SearchResult] = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                raw_meta = entity.get("metadata", "{}")
                try:
                    meta = (
                        json.loads(raw_meta)
                        if isinstance(raw_meta, str)
                        else raw_meta or {}
                    )
                except (json.JSONDecodeError, TypeError):
                    meta = {}

                parsed.append(
                    SearchResult(
                        id=hit.get("id", 0),
                        score=hit.get("distance", 0.0),
                        text=entity.get("text", ""),
                        content_type=entity.get("content_type", ""),
                        agent_name=entity.get("agent_name", ""),
                        metadata=meta,
                        timestamp=entity.get("timestamp", ""),
                    )
                )

        logger.debug("向量搜索完成: user=%s, hits=%d", user_id, len(parsed))
        return parsed

    # ── 管理操作 ──────────────────────────────────────────────────────

    async def delete_by_filter(self, user_id: str, expr: str) -> int:
        """按表达式删除记录。"""
        client = self._ensure_client()
        name = self._collection_name(user_id)

        try:
            res = client.delete(collection_name=name, filter=expr)
            count = res.get("delete_count", 0) if isinstance(res, dict) else 0
            logger.info(
                "向量删除完成: user=%s, expr=%s, count=%d", user_id, expr, count
            )
            return count
        except Exception as e:
            logger.error("向量删除失败 (user=%s, expr=%s): %s", user_id, expr, e)
            raise

    async def count(self, user_id: str) -> int:
        """统计指定用户集合中的记录总数。"""
        client = self._ensure_client()
        name = self._collection_name(user_id)

        try:
            if not client.has_collection(name):
                return 0
            res = client.query(
                collection_name=name, filter="", output_fields=["count(*)"]
            )
            return res[0]["count(*)"]
        except Exception as e:  # noqa: BLE001
            logger.warning("查询集合数量失败 (user=%s): %s", user_id, e)
            return 0
