"""Embedding 懒加载回归测试。

覆盖：注册后不加载、首次 encode 并发只加载一次、batch 复用已加载模型。
"""

from __future__ import annotations

import asyncio

import pytest
from my_df.runtime.embeddings.sentence import SentenceEmbeddings


class _Vec(list):
    """带 tolist() 的伪 numpy 向量。"""

    def tolist(self) -> list:
        return list(self)


class _DummyModel:
    """最小伪 embedding 模型（不真实加载权重）。"""

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(texts, str):
            return _Vec([0.1] * 384)
        return [_Vec([0.1] * 384) for _ in range(len(texts))]


@pytest.fixture
def embedder(monkeypatch):
    """构造未加载的 SentenceEmbeddings，替换 _load_sync 为计数伪实现。"""
    e = SentenceEmbeddings("fake-model")
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        return _DummyModel()

    monkeypatch.setattr(e, "_load_sync", fake_load)
    return e, calls


async def test_lazy_not_loaded_until_encode(embedder):
    """注册后不加载模型（启动不占内存）。"""
    e, calls = embedder
    assert e._model is None
    assert calls["n"] == 0


async def test_encode_loads_once_concurrently(embedder):
    """并发首次 encode 只加载一次（懒加载并发安全）。"""
    e, calls = embedder
    results = await asyncio.gather(*[e.encode("文本") for _ in range(5)])

    assert calls["n"] == 1
    assert len(results) == 5
    assert len(results[0]) == 384
    assert e.dim == 384


async def test_encode_batch_reuses_loaded(embedder):
    """batch 编码复用已加载模型，不重复加载。"""
    e, calls = embedder
    await e.encode("先加载")
    batch = await e.encode_batch(["b", "c"])

    assert calls["n"] == 1
    assert len(batch) == 2


async def test_encode_after_close_reloads(embedder):
    """close 释放后再次 encode 会重新加载。"""
    e, calls = embedder
    await e.encode("a")
    await e.close()
    await e.encode("b")

    assert calls["n"] == 2
