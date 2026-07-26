"""基于 sentence-transformers 的本地 Embedding 服务。

用法：
    embedder = SentenceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector = await embedder.encode("你好世界")
    print(embedder.dim)  # 384
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SentenceEmbeddings:
    """将文本转换为向量（异步封装，内部在线程池中同步执行）。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dim: int = 0

    async def load(self) -> None:
        """加载模型（耗时较长，需在 lifespan 中调用一次）。"""
        if self._model is not None:
            return

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            self._load_sync,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "SentenceTransformer 已加载: model=%s, dim=%d",
            self._model_name,
            self._dim,
        )

    def _load_sync(self) -> Any:
        from sentence_transformers import SentenceTransformer

        logger.info("正在加载 SentenceTransformer: %s ...", self._model_name)
        model = SentenceTransformer(self._model_name)
        return model

    @property
    def dim(self) -> int:
        """返回向量维度（如 all-MiniLM-L6-v2 为 384）。"""
        return self._dim

    async def encode(self, text: str) -> list[float]:
        """将单段文本编码为向量。"""
        if self._model is None:
            raise RuntimeError("Embedding 模型未加载，请先调用 load()")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._encode_sync, text)
        return result

    def _encode_sync(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码多段文本（内部使用 GPU/批处理优化）。"""
        if self._model is None:
            raise RuntimeError("Embedding 模型未加载，请先调用 load()")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._encode_batch_sync, texts)
        return result

    def _encode_batch_sync(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    async def close(self) -> None:
        """释放模型资源。"""
        self._model = None
        self._dim = 0
        logger.info("SentenceTransformer 已释放")
