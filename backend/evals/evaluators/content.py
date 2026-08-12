"""内容判定器：校验最终回答是否包含期望关键词（或非空）。"""

from __future__ import annotations


def evaluate_content(
    must_contain: list[str],
    last_ai: str,
) -> tuple[bool, list[str]]:
    """关键词存在性判定；must_contain 为空时仅要求回答非空。"""
    if must_contain:
        missing = [kw for kw in must_contain if kw not in last_ai]
        if missing:
            return False, [f"回答缺少关键词: {missing}"]
        return True, []
    if last_ai:
        return True, []
    return False, ["回答为空"]
