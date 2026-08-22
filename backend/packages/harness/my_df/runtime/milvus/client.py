"""基于 pymilvus MilvusClient（新版 API）的向量存储实现。

使用 ``pymilvus.MilvusClient`` 替代已废弃的 ORM 风格 API
（``connections.connect`` / ``Collection``），避免 PyMilvus 3.1 后的兼容问题。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pymilvus import (
    CollectionSchema,
    DataType,
    Function,
    FunctionType,
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
        self._last_health_check: float | None = None
        self._bm25_ready: set[str] = set()
        self._bm25_unavailable: set[str] = set()
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
        """构建 Collection Schema (MilvusClient 兼容)。

        除稠密向量外，额外声明 BM25 全文索引所需的：
        - ``text`` 字段开启分析器（中文分词）；
        - ``sparse`` 稀疏向量字段，由 BM25 函数自动生成；
        - ``text_bm25_emb`` 函数：text -> sparse。
        """
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
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(
            field_name="content_type", datatype=DataType.VARCHAR, max_length=64
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(
            field_name="timestamp", datatype=DataType.VARCHAR, max_length=32
        )
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )
        return schema

    def _build_index_params(self) -> Any:
        """构建索引参数（稠密向量 + BM25 稀疏倒排）。"""
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=self._config.index_type,
            metric_type="IP",
            params=(
                {"nlist": self._config.nlist}
                if self._config.index_type != "HNSW"
                else {"M": 16, "efConstruction": 200}
            ),
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
        return index_params

    async def _ensure_connected(self) -> MilvusClient:
        """确保客户端可用；channel 断开时自动重连（5 秒健康检查缓存）。"""
        now = time.monotonic()
        if (
            self._client is not None
            and self._last_health_check is not None
            and now - self._last_health_check < 5.0
        ):
            return self._client

        if self._client is None:
            await self.connect()
        else:
            try:
                self._client.get_server_version()
            except Exception as e:  # noqa: BLE001
                logger.warning("Milvus 连接已断开，自动重连: %s", e)
                self._client = None
                await self.connect()
        self._last_health_check = time.monotonic()
        return self._client

    def _ensure_client(self) -> MilvusClient:
        """返回当前客户端，如果未连接则抛出异常。"""
        if self._client is None:
            raise RuntimeError("MilvusClient 未连接，请先调用 connect()")
        return self._client

    @staticmethod
    def _build_filter_expr(
        user_id: str,
        agent_name: str | None = None,
        content_type: str | None = None,
    ) -> str:
        """构建 Milvus 过滤表达式，默认按用户隔离。"""
        expr_parts = [f'user_id == "{user_id}"']
        if agent_name:
            expr_parts.append(f'agent_name == "{agent_name}"')
        if content_type:
            expr_parts.append(f'content_type == "{content_type}"')
        return " and ".join(expr_parts)

    @staticmethod
    def _parse_metadata(raw: Any) -> dict[str, Any]:
        """解析 Milvus JSON 字段，兼容字符串与 dict 两种返回形态。"""
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        return raw or {}

    # ── 属性 ──────────────────────────────────────────────────────────

    @property
    def vector_dim(self) -> int:
        return self._config.vector_dim

    # ── 生命周期 ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        """连接到 Milvus 服务。

        快速失败：3 次重试（约 5 秒）即放弃，避免 Milvus 未启动时
        长时间阻塞服务启动（连接失败由上层降级处理）。
        """
        if self._client is not None:
            logger.debug("PyMilvusStorage 已连接，跳过")
            return

        import asyncio

        max_retries = 3
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
        client = await self._ensure_connected()
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
                        # 确保集合已加载（Milvus 重启后需重新加载）
                        client.load_collection(name)
                        logger.debug("集合已加载: %s", name)
                        return
                    logger.warning(
                        "集合 %s 维度不匹配（现有=%d, 期望=%d），删除重建",
                        name,
                        existing_dim,
                        self._config.vector_dim,
                    )
                    client.drop_collection(name)
                except Exception as desc_err:  # noqa: BLE001
                    logger.warning("描述集合 %s 失败，尝试重建: %s", name, desc_err)
                    try:
                        client.drop_collection(name)
                    except Exception:  # noqa: BLE001
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

            # 加载集合到内存，搜索前必须执行
            client.load_collection(name)
            logger.info("集合已加载: %s", name)
        except Exception as e:
            logger.error("创建集合 %s 失败: %s", name, e)
            raise

    async def drop_collection(self, user_id: str) -> None:
        """删除指定用户的集合（危险操作！）。"""
        client = await self._ensure_connected()
        name = self._collection_name(user_id)
        self._bm25_ready.discard(name)
        self._bm25_unavailable.discard(name)
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
        client = await self._ensure_connected()
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

    async def embedding_search(
        self,
        user_id: str,
        query_vector: list[float],
        top_k: int = 5,
        agent_name: str | None = None,
        content_type: str | None = None,
    ) -> list[SearchResult]:
        """稠密向量相似度检索（IP 度量）。"""
        client = await self._ensure_connected()
        name = self._collection_name(user_id)

        expr = self._build_filter_expr(user_id, agent_name, content_type)

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
                meta = self._parse_metadata(entity.get("metadata", "{}"))

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

    async def bm25_search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        agent_name: str | None = None,
        content_type: str | None = None,
    ) -> list[SearchResult]:
        """基于 Milvus 内置 BM25 全文索引的关键词检索。

        依赖集合 Schema 中的 ``text_bm25_emb`` 函数（text -> sparse）与
        ``SPARSE_INVERTED_INDEX``（metric_type=BM25）。旧集合缺少该结构时
        返回空列表并告警，不破坏既有向量数据。
        """
        client = await self._ensure_connected()
        name = self._collection_name(user_id)

        if not self._collection_has_bm25(client, name):
            logger.warning(
                "集合 %s 缺少 BM25 全文索引（需重建集合后重新入库），"
                "关键词检索降级为空",
                name,
            )
            return []

        expr = self._build_filter_expr(user_id, agent_name, content_type)

        try:
            results = client.search(
                collection_name=name,
                data=[query],
                anns_field="sparse",
                limit=top_k,
                search_params={"metric_type": "BM25"},
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
            logger.error("BM25 关键词搜索失败 (user=%s): %s", user_id, e)
            raise

        parsed: list[SearchResult] = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                meta = self._parse_metadata(entity.get("metadata", "{}"))

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

        logger.debug("BM25 关键词搜索完成: user=%s, hits=%d", user_id, len(parsed))
        return parsed

    def _collection_has_bm25(self, client: MilvusClient, name: str) -> bool:
        """判断集合是否具备 BM25 全文检索能力（结果按集合缓存）。"""
        if name in self._bm25_ready:
            return True
        if name in self._bm25_unavailable:
            return False

        try:
            desc = client.describe_collection(name)
            fields = desc.get("fields", [])
            functions = desc.get("functions", []) or []
            has_sparse = any(f.get("name") == "sparse" for f in fields)
            has_bm25_fn = any(
                fn.get("name") == "text_bm25_emb" or fn.get("type") == "BM25"
                for fn in functions
            )
            available = has_sparse and has_bm25_fn
        except Exception as e:  # noqa: BLE001
            logger.warning("检查集合 %s 的 BM25 能力失败: %s", name, e)
            available = False

        if available:
            self._bm25_ready.add(name)
        else:
            self._bm25_unavailable.add(name)
        return available

    async def list_records(
        self,
        user_id: str,
        content_type: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SearchResult]:
        """按过滤条件列出向量记录，供知识库管理接口使用。"""
        client = await self._ensure_connected()
        name = self._collection_name(user_id)
        if not client.has_collection(name):
            return []

        expr = self._build_filter_expr(user_id, agent_name, content_type)
        try:
            rows = client.query(
                collection_name=name,
                filter=expr,
                output_fields=[
                    "id",
                    "text",
                    "content_type",
                    "agent_name",
                    "metadata",
                    "timestamp",
                ],
                limit=limit,
                offset=offset,
            )
        except Exception as e:  # noqa: BLE001, RUF100
            logger.error("向量列表查询失败 (user=%s): %s", user_id, e)
            raise

        parsed: list[SearchResult] = []
        for row in rows:
            parsed.append(
                SearchResult(
                    id=row.get("id", 0),
                    score=0.0,
                    text=row.get("text", ""),
                    content_type=row.get("content_type", ""),
                    agent_name=row.get("agent_name", ""),
                    metadata=self._parse_metadata(row.get("metadata", "{}")),
                    timestamp=row.get("timestamp", ""),
                )
            )
        logger.debug("向量列表查询完成: user=%s, rows=%d", user_id, len(parsed))
        return parsed

    async def delete_by_ids(self, user_id: str, ids: list[int]) -> int:
        """按主键批量删除向量记录。"""
        if not ids:
            return 0

        client = await self._ensure_connected()
        name = self._collection_name(user_id)
        try:
            res = client.delete(collection_name=name, ids=list(ids))
            count = (
                res.get("delete_count", len(ids)) if isinstance(res, dict) else len(ids)
            )
            logger.info(
                "向量按 ID 删除完成: user=%s, ids=%d, count=%d",
                user_id,
                len(ids),
                count,
            )
            return int(count)
        except Exception as e:
            logger.error("向量按 ID 删除失败 (user=%s): %s", user_id, e)
            raise

    async def hybrid_search(
        self,
        user_id: str,
        query_vector: list[float],
        top_k: int = 5,
        agent_name: str | None = None,
        content_type: str | None = None,
    ) -> list[SearchResult]:
        # client = self._ensure_client()
        # name = self.ensure_collection(user_id)

        # 构建索引

        parsed: list[SearchResult] = []
        return parsed

    # ── 管理操作 ──────────────────────────────────────────────────────

    async def delete_by_filter(self, user_id: str, expr: str) -> int:
        """按表达式删除记录。"""
        client = await self._ensure_connected()
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
        client = await self._ensure_connected()
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
