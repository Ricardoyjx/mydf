"""RRF（Reciprocal Rank Fusion）融合排名工具。

将多路检索结果（各自按相关性降序排列）融合为单一排名：

    score(d) = Σ 1 / (k + rank_i(d))

``k`` 取 60（RRF 论文默认值）。分数只依赖排名、不依赖各路的原始分数，
因此可以公平融合异构检索器（向量 / BM25 / 不同模型）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from my_df.runtime.milvus.base import SearchResult


def rrf_fuse(
    ranked_lists: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    key: Callable[[SearchResult], Any] | None = None,
) -> list[SearchResult]:
    """融合多路检索结果，返回按 RRF 分数降序排列的去重列表。

    参数：
        ranked_lists: 多路结果，每路必须按相关性降序排列。
        k: RRF 平滑常数，默认 60。
        key: 去重键；默认取 ``SearchResult.id``（Milvus 主键）。
             small-to-big 场景 id 恒为 0，应传
             ``lambda r: r.metadata.get("parent_id") or r.id``。

    返回：
        融合后的结果列表，就地复用每路中的 SearchResult 对象并更新 score。
    """
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if not ranked_lists:
        return []

    get_key = key or (lambda r: r.id)
    scores: dict[Any, float] = {}
    by_key: dict[Any, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            item_key = get_key(item)
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (k + rank)
            by_key.setdefault(item_key, item)

    fused = sorted(
        by_key.values(),
        key=lambda r: scores[get_key(r)],
        reverse=True,
    )
    for item in fused:
        item.score = scores[get_key(item)]
    return fused
