"""工具判定器：校验委派场景中子代理是否真的调用了工具。"""

from __future__ import annotations


def evaluate_tool(
    expected_route: str,
    tool_calls: list[str],
) -> tuple[bool, list[str]]:
    """委派场景必须存在工具调用；直接回答场景不判定。"""
    if expected_route.startswith("subagent:"):
        if tool_calls:
            return True, []
        return False, ["子代理未调用任何工具"]
    return True, []
