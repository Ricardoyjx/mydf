"""RAG 知识库服务与中间件单元测试。

使用 FakeMilvus / FakeEmbedding 隔离真实 Milvus 与模型下载，
验证分块、入库、检索、文档聚合删除与上下文注入。
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from langchain.messages import HumanMessage

from my_df.agents.middlewares.rag_middleware import RagMiddleware, _format_rag_context
from my_df.rag.chunker import split_text
from my_df.rag.service import KnowledgeService
from my_df.runtime.milvus.base import SearchResult


class FakeMilvus:
    """内存版 MilvusStorage 替身，仅实现 RAG 需要的接口。"""

    def __init__(self) -> None:
        self.records: list[SearchResult] = []
        self._next_id = 1
        self.insert_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []

    async def ensure_collection(self, user_id: str) -> None:
        pass

    async def insert(
        self,
        user_id: str,
        agent_name: str,
        text: str,
        vector: list[float],
        content_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        self.insert_calls.append(
            {
                "user_id": user_id,
                "agent_name": agent_name,
                "text": text,
                "vector": vector,
                "content_type": content_type,
                "metadata": metadata or {},
            }
        )
        record = SearchResult(
            id=self._next_id,
            score=0.0,
            text=text,
            content_type=content_type,
            agent_name=agent_name,
            metadata=metadata or {},
            timestamp="2026-08-01T00:00:00Z",
        )
        self.records.append(record)
        self._next_id += 1
        return record.id

    async def search(
        self,
        user_id: str,
        query_vector: list[float],
        top_k: int = 5,
        agent_name: str | None = None,
        content_type: str | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append(
            {
                "user_id": user_id,
                "query_vector": query_vector,
                "top_k": top_k,
                "agent_name": agent_name,
                "content_type": content_type,
            }
        )
        return [r for r in self.records if r.content_type == (content_type or "")]

    async def list_records(
        self,
        user_id: str,
        content_type: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SearchResult]:
        return [
            r
            for r in self.records
            if (content_type is None or r.content_type == content_type)
            and (agent_name is None or r.agent_name == agent_name)
        ][offset : offset + limit]

    async def delete_by_ids(self, user_id: str, ids: list[int]) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r.id not in ids]
        return before - len(self.records)


class FakeEmbedding:
    """固定返回二维向量的 Embedding 替身。"""

    async def encode(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_split_text_short_keeps_single_chunk():
    """短文本只返回一块。"""
    assert split_text("你好，世界") == ["你好，世界"]


def test_split_text_long_splits_with_overlap():
    """长文本按 chunk_size 切分，且不产生空块。"""
    text = "段落甲\n\n" + "字" * 200 + "\n\n段落乙"
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_add_text_stores_chunks_with_document_metadata():
    """入库后每块都带 document_id / title / chunk_index。"""
    milvus = FakeMilvus()
    service = KnowledgeService(milvus, FakeEmbedding())
    chunks = asyncio.run(
        service.add_text(
            user_id="u1",
            title="计算器指南",
            content="这是第一段。\n\n这是第二段。" * 50,
            chunk_size=100,
            chunk_overlap=20,
        )
    )

    assert len(chunks) >= 2
    assert len({c.document_id for c in chunks}) == 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(call["content_type"] == "knowledge" for call in milvus.insert_calls)
    assert all(
        call["metadata"]["title"] == "计算器指南" for call in milvus.insert_calls
    )


def test_search_only_returns_knowledge():
    """语义检索按 content_type="knowledge" 过滤。"""
    milvus = FakeMilvus()
    service = KnowledgeService(milvus, FakeEmbedding())
    asyncio.run(
        service.add_text(
            user_id="u1",
            title="知识库",
            content="Python 计算器实现",
        )
    )

    results = asyncio.run(service.search(user_id="u1", query="计算器", top_k=3))
    assert results
    assert milvus.search_calls[0]["content_type"] == "knowledge"
    # 服务端为给 rerank / 过滤留出余量，粗召回 top_k * 4，最后再切片回 top_k
    assert milvus.search_calls[0]["top_k"] == 12


def test_list_documents_groups_by_document_id():
    """文档列表按 document_id 聚合块数。"""
    milvus = FakeMilvus()
    service = KnowledgeService(milvus, FakeEmbedding())
    asyncio.run(
        service.add_text(
            user_id="u1",
            title="文档一",
            content="长内容" * 200,
        )
    )
    asyncio.run(
        service.add_text(
            user_id="u1",
            title="文档二",
            content="短内容",
        )
    )

    docs = asyncio.run(service.list_documents(user_id="u1"))
    assert len(docs) == 2
    by_title = {doc["title"]: doc for doc in docs}
    assert by_title["文档一"]["chunk_count"] >= 1
    assert by_title["文档二"]["chunk_count"] == 1


def test_delete_document_removes_all_chunks():
    """按 document_id 删除文档的全部块。"""
    milvus = FakeMilvus()
    service = KnowledgeService(milvus, FakeEmbedding())
    chunks = asyncio.run(
        service.add_text(
            user_id="u1",
            title="待删除",
            content="长内容" * 200,
        )
    )
    asyncio.run(
        service.add_text(
            user_id="u1",
            title="保留",
            content="短内容",
        )
    )

    deleted = asyncio.run(
        service.delete_document(user_id="u1", document_id=chunks[0].document_id)
    )
    assert deleted == len(chunks)
    docs = asyncio.run(service.list_documents(user_id="u1"))
    assert [doc["title"] for doc in docs] == ["保留"]


def test_format_rag_context_filters_low_score():
    """低分结果不会进入 <rag_context>。"""
    results = [
        SearchResult(
            id=1,
            score=0.5,
            text="高相关内容",
            content_type="knowledge",
            agent_name="lead-agent",
            metadata={"title": "文档A", "source": "upload"},
        ),
        SearchResult(
            id=2,
            score=0.1,
            text="低相关噪音",
            content_type="knowledge",
            agent_name="lead-agent",
        ),
    ]
    block = _format_rag_context(results)
    assert "<rag_context>" in block
    assert "高相关内容" in block
    assert "低相关噪音" not in block


def test_rag_middleware_injects_context():
    """RagMiddleware 会把知识库命中注入首条 HumanMessage。"""
    milvus = FakeMilvus()
    milvus.records.append(
        SearchResult(
            id=1,
            score=0.6,
            text="Python 计算器实现说明",
            content_type="knowledge",
            agent_name="lead-agent",
            metadata={"title": "计算器指南", "source": "manual"},
        )
    )
    mw = RagMiddleware(
        agent_name="lead_agent",
        user_id="u1",
        milvus=milvus,
        embedding_model=FakeEmbedding(),
    )
    state = {"messages": [HumanMessage(content="怎么用 Python 写计算器？")]}
    result = asyncio.run(mw.abefore_model(state, MagicMock()))

    assert result is not None
    content = result["messages"][0].content
    assert "<rag_context>" in content
    assert "Python 计算器实现说明" in content
    assert "怎么用 Python 写计算器？" in content


def test_rag_middleware_skips_without_milvus():
    """Milvus 不可用时 RAG 中间件直接跳过。"""
    mw = RagMiddleware(
        agent_name="lead_agent",
        user_id="u1",
        milvus=None,
        embedding_model=FakeEmbedding(),
    )
    state = {"messages": [HumanMessage(content="你好")]}
    result = asyncio.run(mw.abefore_model(state, MagicMock()))
    assert result is None
