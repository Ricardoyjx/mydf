"""路由判定器：校验 supervisor 是否委派给期望的子代理（或直接回答）。"""

from __future__ import annotations

from typing import Any


def evaluate_route(
    expected_route: str,
    nodes_visited: set[str],
) -> tuple[bool, list[str]]:
    """根据期望路由与图实际访问节点判定。

    规则：
    - ``subagent:<name>``：必须访问过 ``subagent`` 节点；
    - ``direct``：不得访问 ``subagent`` 节点；
    - 其他值：不判定（视为通过）。
    """
    if expected_route.startswith("subagent:"):
        target = expected_route.split(":", 1)[1]
        if "subagent" in nodes_visited:
            return True, []
        return False, [f"未委派给子代理（期望 {target}）"]
    if expected_route == "direct":
        if "subagent" not in nodes_visited:
            return True, []
        return False, ["本应直接回答，却委派了子代理"]
    return True, []
