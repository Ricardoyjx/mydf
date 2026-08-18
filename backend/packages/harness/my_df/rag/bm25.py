"""内存 BM25 关键词检索索引（jieba 中文分词）。

作为 RAG 的第二路召回，与 Milvus 向量检索互补：

- 向量检索擅长语义相关；
- 关键词检索擅长精确词命中（人名、编号、术语）；

两路结果由 :func:`my_df.rag.rrf.rrf_fuse` 融合后再进入 rerank。

索引与 KnowledgeService 同步维护：``add_text`` 时 ``add``、
``delete_document`` 时 ``remove``；进程重启后首次使用时从 Milvus
全量记录 ``rebuild``。
"""

from __future__ import annotations

import logging
import threading

import jieba
from rank_bm25 import BM25Okapi

from my_df.runtime.milvus.base import SearchResult

logger = logging.getLogger(__name__)


class BM25Index:
    """线程安全的内存 BM25 索引（按用户隔离，由 KnowledgeService 持有）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: list[SearchResult] = []
        self._corpus: list[list[str]] = []
        self._id_to_pos: dict[int, int] = {}
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 分词，剔除空白符，统一小写。"""
        return [
            token.strip().lower() for token in jieba.cut(text or "") if token.strip()
        ]

    def _rebuild_locked(self) -> None:
        if not self._docs:
            self._corpus = []
            self._bm25 = None
            return
        self._corpus = [self._tokenize(doc.text) for doc in self._docs]
        self._bm25 = BM25Okapi(self._corpus)

    def add(self, item: SearchResult) -> None:
        """新增一条记录；已存在同 id 则覆盖。"""
        with self._lock:
            pos = self._id_to_pos.get(item.id)
            if pos is not None:
                self._docs[pos] = item
            else:
                self._id_to_pos[item.id] = len(self._docs)
                self._docs.append(item)
            self._rebuild_locked()

    def remove(self, ids: list[int]) -> int:
        """按 Milvus 主键删除记录，返回删除条数。"""
        with self._lock:
            id_set = set(ids)
            before = len(self._docs)
            kept = [
                (doc, toks)
                for doc, toks in zip(self._docs, self._corpus)
                if doc.id not in id_set
            ]
            self._docs = [doc for doc, _ in kept]
            self._corpus = [toks for _, toks in kept]
            removed = before - len(self._docs)
            if removed:
                self._id_to_pos = {doc.id: i for i, doc in enumerate(self._docs)}
                self._rebuild_locked()
            return removed

    def rebuild(self, records: list[SearchResult]) -> None:
        """全量重建（启动后首次使用 / 数据不一致兜底）。"""
        with self._lock:
            self._docs = list(records)
            self._id_to_pos = {doc.id: i for i, doc in enumerate(self._docs)}
            self._rebuild_locked()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """BM25 关键词召回，返回按分数降序的 SearchResult 副本。"""
        with self._lock:
            if self._bm25 is None or not self._docs:
                return []
            tokens = self._tokenize(query)
            if not tokens:
                return []
            token_set = set(tokens)
            scores = self._bm25.get_scores(tokens)
            order = [
                i
                for i, doc_tokens in enumerate(self._corpus)
                if token_set & set(doc_tokens)
            ]
            order.sort(key=lambda i: scores[i], reverse=True)
            results: list[SearchResult] = []
            for idx in order[:top_k]:
                src = self._docs[idx]
                results.append(
                    SearchResult(
                        id=src.id,
                        score=float(scores[idx]),
                        text=src.text,
                        content_type=src.content_type,
                        agent_name=src.agent_name,
                        metadata=dict(src.metadata or {}),
                        timestamp=src.timestamp,
                    )
                )
            return results
