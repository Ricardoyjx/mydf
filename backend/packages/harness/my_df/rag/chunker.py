"""轻量文本分块器。

按段落优先合并，超过 ``chunk_size`` 的段落再按字符窗口切分，
相邻块之间保留 ``overlap`` 个字符，降低语义断裂。
"""

from __future__ import annotations

import re


def split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """把长文本切分为带重叠的文本块。

    参数：
        text:       原始文本。
        chunk_size: 单块最大字符数。
        overlap:    相邻块重叠字符数，不能超过 chunk_size 的一半。

    返回：
        非空文本块列表；输入为空时返回空列表。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能为负数")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    overlap = min(overlap, chunk_size // 2)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush()
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                chunk = paragraph[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(paragraph):
                    break
                start = max(end - overlap, start + 1)
            continue

        if not buffer:
            buffer = paragraph
        elif len(buffer) + 1 + len(paragraph) <= chunk_size:
            buffer = f"{buffer}\n{paragraph}"
        else:
            flush()
            buffer = paragraph

    flush()
    return chunks or [normalized]
