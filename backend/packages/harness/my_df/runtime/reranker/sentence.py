import asyncio
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SentenceRerank:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model: Any = None

    async def load(self) -> None:
        if self._model is not None:
            return

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_sync)
        logger.info("加载 SentenceReranker 模型 %s", self._model_name)

    def _load_sync(self) -> Any:
        from sentence_transformers import CrossEncoder

        logger.info("加载模型 %s", self._model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return CrossEncoder(self._model_name, device=device)

    async def reranker(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            raise RuntimeError("Reranker 未加载，请先调用 load()")

        pairs = [(query, t) for t in texts]

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._model.predict, pairs)
        return [float(s) for s in scores]
