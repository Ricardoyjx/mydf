"""Small-to-Big（父子分块）检索：基于 langchain 官方 ParentDocumentRetriever。

架构：
- 子块（child）：短文本（默认 200 字符）向量化后写入 Milvus；
- 父块（parent）：长文本（默认 1000 字符）整段写入 docstore（PostgreSQL
  ``rag_parent_documents`` 表，或 InMemoryStore 兜底），不向量化；
- 检索时先在 Milvus 召回子块，再按 ``parent_id`` 映射取回父块整段注入 LLM，
  用更完整的上下文提升回答质量。

为什么在官方类上做子类：
1. 官方 ``aadd_documents`` 存在子块下标覆盖 ``parent_id`` 的历史问题，
   多个父块的子块会互相串 id；
2. 官方 ``_get_relevant_documents`` 只返回父块、丢弃相似度分数，导致
   ``min_score`` 过滤与 rerank 无法工作。
本模块保持官方 ParentDocumentRetriever 的编排（docstore / vectorstore /
splitter / invoke），只修复上述两点。
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import BaseStore, InMemoryStore
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from pydantic import PrivateAttr

from my_df.config.checkpointer_config import CheckpointerConfig
from my_df.rag.chunker import split_text
from my_df.runtime.milvus.base import MilvusStorage

logger = logging.getLogger(__name__)


class ParagraphTextSplitter(TextSplitter):
    """把项目既有段落优先分块器适配为 langchain TextSplitter。"""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("overlap 必须大于等于 0")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> list[str]:
        return split_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)


class MilvusVectorStore(VectorStore):
    """把现有 MilvusStorage 适配为 langchain_core VectorStore。

    只实现 ParentDocumentRetriever 需要的异步方法；同步方法作为离线/测试
    兜底（事件循环内调用会明确报错，防止阻塞事件循环）。
    """

    def __init__(
        self,
        storage: MilvusStorage,
        embedding: Any,
        *,
        user_id: str = "default",
        agent_name: str = "default",
        content_type: str = "knowledge",
    ) -> None:
        super().__init__()
        self._storage = storage
        self._embedding = embedding
        self._user_id = user_id
        self._agent_name = (agent_name or "default").replace("_", "-")
        self._content_type = content_type

    async def aadd_texts(
        self,
        texts: Sequence[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        vectors = await self._embedding.encode_batch(list(texts))
        if len(vectors) != len(texts):
            raise RuntimeError("向量化结果数量与输入数量不一致")
        saved: list[str] = []

        for index, (text, vector) in enumerate(zip(texts, vectors)):
            meta = (
                dict(metadatas[index]) if metadatas and index < len(metadatas) else {}
            )
            inserted_id = await self._storage.insert(
                user_id=self._user_id,
                agent_name=self._agent_name,
                text=text,
                vector=vector,
                content_type=self._content_type,
                metadata=meta,
            )
            saved.append(str(inserted_id))
        return saved

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        return _run_sync(
            self.aadd_texts(texts, metadatas, ids=ids, **kwargs),
            "MilvusVectorStore.add_texts",
        )

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        query_vector = await self._embedding.encode(query)
        results = await self._storage.search(
            user_id=self._user_id,
            query_vector=query_vector,
            top_k=k,
            agent_name=self._agent_name,
            content_type=self._content_type,
        )

        docs: list[Document] = []
        for result in results:
            meta = dict(result.metadata or [])
            meta["score"] = float(result.score)
            docs.append(Document(page_content=result.text, metadata=meta))
        return docs

    def similarity_search(self, query: str, k=4, **kwargs) -> list[Document]:
        return _run_sync(
            self.asimilarity_search(query, k, **kwargs),
            "MilvusVectorStore.similarity_search",
        )

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        if not ids:
            return True
        int_ids = [int(i) for i in ids if str(i).isdigit()]
        if not int_ids:
            return True
        deleted = await self._storage.delete_by_ids(self._user_id, int_ids)
        return deleted == len(int_ids)

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "MilvusVectorStore":
        raise NotImplementedError("from_text 暂不支持  ")


def _run_sync(coro: Any, caller: str) -> Any:
    """在非事件循环上下文执行协程；事件循环内调用给出明确错误。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(f"{caller} 暂不支持在事件循环内调用，请使用 asyncio.run()")


class ScoredParentDocumentRetriever(ParentDocumentRetriever):
    """保留子块相似度分数的 ParentDocumentRetriever 增强版。"""

    id_key: str = "parent_id"
    _namespace: str = PrivateAttr(default="default")

    def _parent_key(self, parent_id: str) -> str:
        return f"{self._namespace}:{parent_id}"

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        *,
        run_manager: Any = None,
    ) -> None:

        await self._aadd_documents_with_parents(documents, ids)

    async def _aadd_documents_with_parents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[tuple[str, Document]]:
        """父块入库 docstore，子块向量化入库 Milvus。

        与官方实现差异：每个父块生成唯一 uuid 作为 parent_id，子块 metadata
        写入完整 key（``{namespace}:{uuid}``），避免官方下标覆盖导致的串 id。
        """

        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)

        full_docs: list[tuple[str, Document]] = []

        for doc in documents:
            full_docs.append((self._parent_key(uuid.uuid4().hex), doc))
        await self.docstore.amset(full_docs)

        child_docs: list[Document] = []
        for key, doc in zip((k for k, _ in full_docs), documents):
            for child in self.child_splitter.split_documents([doc]):
                child.metadata = dict(child.metadata or {})
                child.metadata[self.id_key] = key
                child_docs.append(child)
        if child_docs:
            await self.vectorstore.aadd_documents(child_docs)

        return full_docs

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        *,
        run_manager: Any = None,
    ) -> None:
        _run_sync(
            self.aadd_documents(documents, ids, run_manager=run_manager),
            "ScoreParentDocumentRetriever.add_documents",
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        """子块召回 → 按 parent_id 去重 → 取回父块，并保留最高子块分数。"""
        sub_docs = await self.vectorstore.asimilarity_search(
            query, **self.search_kwargs
        )
        best: dict[str, tuple[float, Document]] = {}

        for sub in sub_docs:
            key = (sub.metadata or {}).get(self.id_key)
            if not key:
                continue
            score = float((sub.metadata or {}).get("score") or 0.0)
            if key not in best or score > best[key][0]:
                best[key] = (score, sub)

        if not best:
            return []

        parent_docs = await self.docstore.amget(list(best.keys()))
        result: list[Document] = []
        for key, parent in zip(best.keys(), parent_docs):
            if parent is None:
                continue
            score, _ = best[key]
            merged = parent.model_copy(deep=True)
            merged.metadata = dict(merged.metadata or {})
            merged.metadata["score"] = score
            merged.metadata["parent_id"] = key
            result.append(merged)
        return result

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> list[Document]:
        return _run_sync(
            self._aget_relevant_documents(query, run_manager=run_manager),
            "ScoreParentDocumentRetriever._get_relevant_documents",
        )


class PGDocStore(BaseStore[str, Document]):
    """PostgreSQL 父块 docstore，实现 langchain BaseStore 接口。

    表结构（自动建表）：``rag_parent_documents``，键为 ``{user_id}:{uuid}``。
    仅提供异步方法；同步方法明确报错，防止阻塞事件循环。
    """

    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                conninfo=self._conninfo,
                min_size=1,
                max_size=4,
                open=False,
                kwargs={"autocommit": True},
            )

            await pool.open(wait=True, timeout=10)
            self._pool = pool
        return self._pool

    async def setup(self) -> None:
        """创建 rag_parent_documents 表与索引（幂等）。"""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_parent_documents (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL DEFAULT '',
                    document_id TEXT NOT NULL DEFAULT '',
                    title       TEXT NOT NULL DEFAULT '',
                    source      TEXT NOT NULL DEFAULT '',
                    content     TEXT NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_parent_documents_doc 
                ON rag_parent_documents (document_id)
                """
            )

            logger.info("已创建 rag_parent_documents 表与索引")

    async def amset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        if not key_value_pairs:
            return
        pool = await self._get_pool()

        from psycopg.types.json import Jsonb

        rows = []

        for key, doc in key_value_pairs:
            user_id, _, _ = key.partition(":")
            meta = doc.metadata or {}
            rows.append(
                (
                    key,
                    user_id,
                    str(meta.get("document_id", "")),
                    str(meta.get("title", "")),
                    str(meta.get("source", "")),
                    doc.page_content,
                    Jsonb(meta),
                )
            )

        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                """
                    INSERT INTO rag_parent_documents
                        (id, user_id, document_id, title, source, content, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        title       = EXCLUDED.title,
                        source      = EXCLUDED.source,
                        content     = EXCLUDED.content,
                        metadata    = EXCLUDED.metadata
                    """,
                rows,
            )

    async def amget(self, keys: Sequence[str]) -> list[Document | None]:
        if not keys:
            return []
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT id, content, metadata FROM rag_parent_documents "
                "WHERE id = ANY(%s)",
                (list(keys),),
            )
            rows = await cur.fetchall()

        by_id = {row[0]: row for row in rows}
        result: list[Document | None] = []
        for key in keys:
            row = by_id.get(key)
            result.append(
                Document(page_content=row[1], metadata=row[2])
                if row is not None
                else None
            )
        return result

    async def amdelete(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM rag_parent_documents WHERE id = ANY(%s)",
                (list(keys),),
            )

    async def ayield_keys(self, *, prefix: str | None = None) -> AsyncIterator[str]:
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            if prefix:
                await cur.execute(
                    "SELECT id FROM rag_parent_documents WHERE id LIKE %s",
                    (f"{prefix}%",),
                )
            else:
                await cur.execute("SELECT id FROM rag_parent_documents")

            async for row in cur:
                yield row[0]

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("已关闭 rag_parent_documents 表与索引")

    # 同步接口仅为满足 BaseStore 抽象方法；本模块异步优先，直接调用会明确报错
    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        raise RuntimeError(
            "PGDocStore 是异步存储，请使用 amset/amget/amdelete/ayield_keys"
        )

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        raise RuntimeError(
            "PGDocStore 是异步存储，请使用 amset/amget/amdelete/ayield_keys"
        )

    def mdelete(self, keys: Sequence[str]) -> None:
        raise RuntimeError(
            "PGDocStore 是异步存储，请使用 amset/amget/amdelete/ayield_keys"
        )

    def yield_keys(self, *, prefix: str | None = None) -> Any:
        raise RuntimeError(
            "PGDocStore 是异步存储，请使用 amset/amget/amdelete/ayield_keys"
        )


async def make_parent_docstore(
    checkpointer: CheckpointerConfig | None,
) -> BaseStore[str, Document]:

    if (
        checkpointer is not None
        and checkpointer.type == "postgres"
        and checkpointer.connection_string
    ):
        store = PGDocStore(checkpointer.connection_string)
        await store.setup()
        logger.info("已创建 rag_parent_documents 表与索引")
        return store
    logger.warning("未配置 Postgres 存储，将使用内存存储")
    return InMemoryStore()
