"""RAG 知识库服务。

职责：
1. 把上传文档切块并批量向量化；
2. 以 ``content_type="knowledge"`` 写入 Milvus，与对话记忆隔离；
3. 提供语义检索、文档列表和按文档删除。
"""

from __future__ import annotations

import html
import io
import logging
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from my_df.rag.chunker import split_text
from my_df.rag.parent_doc import (
    MilvusVectorStore,
    ParagraphTextSplitter,
    ScoredParentDocumentRetriever,
)
from my_df.runtime.milvus.base import MilvusStorage, SearchResult
from my_df.runtime.reranker.sentence import SentenceRerank

logger = logging.getLogger(__name__)


def extract_text_from_file(filename: str, raw: bytes) -> str:
    """按文件扩展名提取纯文本。

    - ``.docx``：解压 word/document.xml 并提取段落文本；
    - 其他（.txt/.md/.json 等）：按 UTF-8 解码。

    抛出：
        ValueError: 文件类型不支持或 docx 解析失败。
    """
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        return _extract_docx_text(raw)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"不支持的文件类型 {ext!r}，或文件不是 UTF-8 文本") from exc


def _extract_docx_text(raw: bytes) -> str:
    """从 docx（zip 容器）中提取 word/document.xml 的纯文本。"""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"无法解析 docx 文件: {exc}") from exc

    paragraphs = re.findall(r"<w:p(?: [^>]*)?>.*?</w:p>", xml, re.DOTALL)
    lines: list[str] = []
    for paragraph in paragraphs:
        text = "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", paragraph, re.DOTALL))
        text = html.unescape(text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


class EmbeddingService(Protocol):
    """支持单条与批量编码的 Embedding 服务。"""

    async def encode(self, text: str) -> list[float]: ...

    async def encode_batch(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class KnowledgeChunk:
    """单条已入库的知识块。"""

    id: int
    document_id: str
    title: str
    source: str
    chunk_index: int
    text: str
    timestamp: str = ""


class KnowledgeService:
    """面向文档知识库的向量存储服务。"""

    def __init__(
        self,
        milvus: MilvusStorage,
        embedding: EmbeddingService,
        reranker: SentenceRerank | None = None,
        *,
        agent_name: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        content_type: str = "knowledge",
        small_to_big: bool = False,
        docstore: Any = None,
        child_chunk_size: int = 200,
        parent_chunk_size: int = 1000,
        child_chunk_overlap: int = 20,
    ) -> None:
        self._milvus = milvus
        self._embedding = embedding
        self._agent_name = (agent_name or "").replace("_", "-") or None
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._content_type = content_type
        self._rerank = reranker
        self._small_to_big = small_to_big
        self._docstore = docstore
        self._child_chunk_size = child_chunk_size
        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_overlap = child_chunk_overlap

    @property
    def small_to_big_enabled(self) -> bool:
        return self._small_to_big

    def _build_retriever(
        self,
        user_id: str,
        recall_k: int,
        agent_name: str | None = None,
    ) -> ScoredParentDocumentRetriever:
        """创建ParentDocumentRetriever"""
        vectorstroe = MilvusVectorStore(
            self._milvus,
            self._embedding,
            user_id=user_id,
            agent_name=agent_name or self._agent_name or "default",
            content_type=self._content_type,
        )

        child_splitter = ParagraphTextSplitter(
            chunk_size=self._child_chunk_size, chunk_overlap=self._child_chunk_overlap
        )

        parent_splitter = ParagraphTextSplitter(
            chunk_size=self._parent_chunk_size, chunk_overlap=self._child_chunk_overlap
        )

        retriever = ScoredParentDocumentRetriever(
            vectorstore=vectorstroe,
            docstore=self._docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": recall_k},
        )

        retriever._namespace = user_id
        return retriever

    async def add_text(
        self,
        *,
        user_id: str,
        title: str,
        content: str,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[KnowledgeChunk]:
        """切分文本、向量化并写入 Milvus。"""
        text = (content or "").strip()
        if not text:
            raise ValueError("文档内容不能为空")

        await self._milvus.ensure_collection(user_id)
        chunks = split_text(
            text,
            chunk_size=chunk_size or self._chunk_size,
            overlap=chunk_overlap or self._chunk_overlap,
        )
        vectors = await self._embedding.encode_batch(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding 返回数量不匹配: chunks={len(chunks)}, vectors={len(vectors)}"
            )

        document_id = uuid.uuid4().hex
        saved: list[KnowledgeChunk] = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_meta: dict[str, Any] = {
                "document_id": document_id,
                "title": title,
                "source": source,
                "chunk_index": index,
            }
            if metadata:
                chunk_meta.update(metadata)

            inserted_id = await self._milvus.insert(
                user_id=user_id,
                agent_name=self._agent_name or "default",
                text=chunk,
                vector=vector,
                content_type=self._content_type,
                metadata=chunk_meta,
            )
            saved.append(
                KnowledgeChunk(
                    id=inserted_id,
                    document_id=document_id,
                    title=title,
                    source=source,
                    chunk_index=index,
                    text=chunk,
                )
            )

        logger.info(
            "知识库文档已入库: user=%s, doc=%s, title=%s, chunks=%d",
            user_id,
            document_id,
            title,
            len(saved),
        )
        return saved

    async def search(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        agent_name: str | None = None,
    ) -> list[SearchResult]:
        """对用户最新问题做语义检索，只返回知识库内容。

        参数：
            min_score: 最低相似度阈值；低于该分的片段会被过滤。
                       ``None`` 表示不过滤。
        """
        await self._milvus.ensure_collection(user_id)
        query_vector = await self._embedding.encode(query)

        """粗召回tok——p * 4 数据"""
        result = await self._milvus.search(
            user_id=user_id,
            query_vector=query_vector,
            top_k=top_k * 4,
            agent_name=agent_name,
            content_type=self._content_type,
        )

        if self._rerank is not None:
            try:
                await self._rerank.ensure_loaded()
                scores = await self._rerank.reranker(query, [r.text for r in result])
                ranked = sorted(
                    zip(scores, result), key=lambda pair: pair[0], reverse=True
                )
                result = [item for _, item in ranked]
                for score, item in ranked:
                    item.score = score
            except Exception as e:  # noqa: BLE001
                logger.warning("Reranker 懒加载或精排失败，本次跳过精排: %s", e)

        if min_score is not None:
            result = [r for r in result if r.score >= min_score]
        return result[:top_k]

    async def list_documents(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """按文档聚合返回知识库目录。"""
        records = await self._milvus.list_records(
            user_id=user_id,
            content_type=self._content_type,
            limit=limit,
            offset=offset,
        )

        documents: dict[str, dict[str, Any]] = {}
        for record in records:
            meta = record.metadata or {}
            doc_id = str(meta.get("document_id") or "unknown")
            item = documents.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "title": meta.get("title", "未命名文档"),
                    "source": meta.get("source", ""),
                    "chunk_count": 0,
                    "created_at": "",
                    "updated_at": "",
                },
            )
            item["chunk_count"] += 1
            item["created_at"] = item["created_at"] or record.timestamp
            item["updated_at"] = record.timestamp or item["updated_at"]

        return sorted(
            documents.values(),
            key=lambda doc: doc["updated_at"] or "",
            reverse=True,
        )

    async def delete_document(self, *, user_id: str, document_id: str) -> int:
        """删除指定文档的全部知识块，返回删除数量。"""
        records = await self._milvus.list_records(
            user_id=user_id,
            content_type=self._content_type,
            limit=10_000,
        )
        ids = [
            record.id
            for record in records
            if str(record.metadata.get("document_id", "")) == document_id
        ]
        if not ids:
            return 0
        return await self._milvus.delete_by_ids(user_id, ids)
