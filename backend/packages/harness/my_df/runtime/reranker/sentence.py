import asyncio
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SentenceRerank:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._load_lock = asyncio.Lock()  # 懒加载并发安全锁

    async def load(self) -> None:
        if self._model is not None:
            return

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_sync)
        logger.info("加载 SentenceReranker 模型 %s", self._model_name)

    async def ensure_loaded(self, timeout: float = 120) -> None:
        """懒加载入口：首次调用时加载模型（并发安全，避免重复加载）。

        加载失败或超时抛异常，由调用方降级处理（如跳过精排）。
        """
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            await asyncio.wait_for(self.load(), timeout=timeout)

    async def close(self) -> None:
        """释放模型资源与 torch 缓存。"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SentenceReranker 已释放")

    def _load_sync(self) -> Any:
        from sentence_transformers import CrossEncoder

        logger.debug("加载模型 %s", self._model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return CrossEncoder(self._model_name, device=device)

    async def reranker(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            raise RuntimeError("Reranker 未加载，请先调用 load()")

        pairs = [(query, t) for t in texts]

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._model.predict, pairs)
        return [float(s) for s in scores]
