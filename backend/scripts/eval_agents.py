"""Agent 评测脚本：加载场景集与评测配置，跑图并输出评估报告。

用法：
    cd backend && .venv/bin/python scripts/eval_agents.py
    cd backend && .venv/bin/python scripts/eval_agents.py --cases evals/datasets/extra.yaml

模块归属：
    - 场景集：evals/datasets/
    - 判定器：evals/evaluators/（route / tool / content）
    - 评测配置：evals/configs/eval_config.yaml
    - 报告产物：evals/reports/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

# 保证 evals 包可导入（脚本从 backend/scripts 运行时 sys.path[0] 为 scripts）
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from evals.evaluators import (
    evaluate_content,
    evaluate_route,
    evaluate_tool,
)
from my_df.agents.supervisor_graph import build_supervisor_graph
from my_df.agents.tools.weather import search_weather
from my_df.config.app_config import get_app_config

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    """加载 YAML 配置/场景文件。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_cases(path: Path) -> list[dict]:
    """加载评测场景集。"""
    return _load_yaml(path).get("cases", [])


def _message_text(msg: Any) -> str:
    """提取消息纯文本。"""
    return str(getattr(msg, "content", "") or "").strip()


def _collect(chunk_output: Any, state: dict) -> None:
    """递归遍历 chunk 输出，收集最后 AI 文本与工具调用名。"""
    if isinstance(chunk_output, dict):
        messages = chunk_output.get("messages")
        if messages:
            for m in messages:
                if getattr(m, "type", "") == "ai":
                    text = _message_text(m)
                    if text:
                        state["last_ai_text"] = text
                    for tc in getattr(m, "tool_calls", None) or []:
                        state["tool_calls"].add(tc.get("name", ""))
        for value in chunk_output.values():
            _collect(value, state)
    elif isinstance(chunk_output, list):
        for item in chunk_output:
            _collect(item, state)


async def _stream_graph(
    graph: Any, graph_input: dict, config: dict
) -> tuple[set[str], dict]:
    """遍历图输出，返回 (访问过的节点集合, 收集状态)。"""
    nodes_visited: set[str] = set()
    state: dict[str, Any] = {"last_ai_text": "", "tool_calls": set()}
    async for chunk in graph.astream(graph_input, config=config, stream_mode="updates"):
        nodes_visited.update(chunk.keys())
        for output in chunk.values():
            _collect(output, state)
    return nodes_visited, state


async def run_case(graph: Any, case: dict, thread_id: str, eval_config: dict) -> dict:
    """运行单个场景，用评估器判定，返回评测结果。"""
    input_text = case["input"]
    expected = case.get("expected", {})
    expected_route = expected.get("route", "")
    must_contain = expected.get("must_contain", [])
    timeout = eval_config.get("timeout_seconds", 120)

    graph_input = {"messages": [{"role": "human", "content": input_text}]}
    run_config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": eval_config.get("user_id", "eval"),
        },
        "recursion_limit": eval_config.get("recursion_limit", 100),
    }

    start = time.monotonic()
    try:
        nodes_visited, state = await asyncio.wait_for(
            _stream_graph(graph, graph_input, run_config),
            timeout=timeout,
        )
        timed_out = False
    except asyncio.TimeoutError:
        nodes_visited, state = set(), {"last_ai_text": "", "tool_calls": set()}
        timed_out = True
    elapsed = time.monotonic() - start

    last_ai = state["last_ai_text"]
    tool_calls = sorted(state["tool_calls"])

    # —— 评估器判定 ——
    checks: dict[str, tuple[bool, list[str]]] = {
        "route": evaluate_route(expected_route, nodes_visited),
        "tool": evaluate_tool(expected_route, tool_calls),
        "content": evaluate_content(must_contain, last_ai),
    }
    fail_reasons: list[str] = []
    if timed_out:
        fail_reasons.append(f"执行超时（>{timeout}s）")
    for ok, reasons in checks.values():
        if not ok:
            fail_reasons.extend(reasons)

    passed = not fail_reasons
    return {
        "name": case["name"],
        "input": input_text,
        "checks": {k: v[0] for k, v in checks.items()},
        "passed": passed,
        "fail_reasons": fail_reasons,
        "nodes": sorted(nodes_visited),
        "tool_calls": tool_calls,
        "last_ai": last_ai[:300],
        "elapsed_s": round(elapsed, 1),
    }


def print_report(results: list[dict]) -> None:
    """打印人类可读的评测报告。"""
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    total_elapsed = sum(r["elapsed_s"] for r in results)

    print("\n========== Agent 评测报告 ==========")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        checks = " ".join(f"{k}={'✓' if v else '✗'}" for k, v in r["checks"].items())
        print(f"\n[{mark}] {r['name']}（{r['elapsed_s']}s）[{checks}]")
        print(f"  input: {r['input']}")
        print(f"  nodes: {', '.join(r['nodes']) or '-'}")
        print(f"  tools: {', '.join(r['tool_calls']) or '-'}")
        print(f"  last_ai: {r['last_ai'][:120] or '-'}")
        if r["fail_reasons"]:
            print(f"  失败原因: {'; '.join(r['fail_reasons'])}")

    print(f"\n通过率: {passed}/{total} ({passed / total * 100:.0f}%)")
    print(f"总耗时: {total_elapsed:.1f}s，平均 {total_elapsed / total:.1f}s/场景")
    print("====================================")


async def main() -> None:
    evals_dir = _BACKEND_ROOT / "evals"
    parser = argparse.ArgumentParser(description="Agent 评测脚本")
    parser.add_argument(
        "--cases",
        type=Path,
        default=evals_dir / "datasets" / "cases.yaml",
        help="评测场景集路径",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=evals_dir / "configs" / "eval_config.yaml",
        help="评测配置路径",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=evals_dir / "reports" / "report.json",
        help="报告输出路径（JSON）",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    eval_config = _load_yaml(args.config)
    cases = load_cases(args.cases)
    if not cases:
        print(f"场景集为空: {args.cases}")
        return

    app_config = get_app_config()
    graph = build_supervisor_graph(
        app_config,
        tools=[search_weather],
        enable_llm_review=bool(eval_config.get("llm_review", False)),
    )

    print(
        f"评测开始: {len(cases)} 个场景，"
        f"LLM 评审={'开' if eval_config.get('llm_review') else '关'}"
    )
    results: list[dict] = []
    for case in cases:
        thread_id = f"eval-{uuid.uuid4().hex[:8]}"
        results.append(await run_case(graph, case, thread_id, eval_config))

    print_report(results)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"报告已保存: {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
