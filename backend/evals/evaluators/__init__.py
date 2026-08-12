"""评测器集合：路由 / 工具 / 内容 三组判定，均返回 (ok, reasons)。

    ok: bool        该维度是否通过
    reasons: list   不通过时的原因（通过时为空列表）
"""

from evals.evaluators.content import evaluate_content
from evals.evaluators.route import evaluate_route
from evals.evaluators.tool import evaluate_tool

__all__ = ["evaluate_content", "evaluate_route", "evaluate_tool"]
