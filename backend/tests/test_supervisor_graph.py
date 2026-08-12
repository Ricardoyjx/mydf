"""Supervisor 多 Agent 编排图 —— 纯函数单元测试。

背景：LangGraph 的异步图执行（ainvoke/astream）在当前测试环境会死锁
（见会话内复现实验），因此这里**不调用整图**，只测试图中可独立验证的部件：
- supervisor_router_node：委派意图解析 / 非法目标兜底 / 工具调用消息清理
- _rule_review：零成本规则检查（空回答 / 过短）
- 条件路由：route_after_supervisor / route_after_reflection（含 else 兜底）
- reflection_node：last_error 分支 / 规则评审 / LLM 评审 / 评审异常回退
- subagent_node：未知目标 / 成功 / 超时 / 异常
- model_call_node：评审反馈注入与消费
- 图构建：不抛异常，注册表包含内置子代理
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, cast
from unittest.mock import patch

from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from my_df.agents.sub_agent.assistant import GENERAL_PURPOSE_CONFIG, filter_tools
from my_df.agents.supervisor_graph import (
    _build_default_registry,
    _last_ai_message,
    _make_model_call_node,
    _make_reflection_node,
    _make_route_after_reflection,
    _make_route_after_supervisor,
    _make_subagent_node,
    _make_supervisor_router,
    _rule_review,
    _sanitize_messages,
    build_supervisor_graph,
    route_to_agent,
)
from my_df.agents.thread_state import ThreadState, merge_route_count
from my_df.config.app_config import AppConfig
from my_df.config.subagent_config import SubagentConfig
from my_df.runtime.events.store.memory import MemoryRunEventStore

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _route_message(agent: str, task: str = "task", msg_id: str = "call-1") -> AIMessage:
    """构造携带 route_to_agent 工具调用的 AIMessage。"""
    return AIMessage(
        content="",
        id=msg_id,
        tool_calls=[
            {
                "name": route_to_agent.name,
                "args": {"agent_name": agent, "task": task},
                "id": msg_id,
            }
        ],
    )


def _registry() -> dict[str, tuple[SubagentConfig, CompiledStateGraph]]:
    """最小注册表：general-purpose → 假子图，其余字段用真实 SubagentConfig。"""

    class _FakeSubgraph:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def ainvoke(self, state: dict, config: dict | None = None) -> dict:
            self.calls.append(state)
            return {"messages": [AIMessage(content="子代理完成，任务已处理完毕。")]}

    return cast(
        dict[str, tuple[SubagentConfig, CompiledStateGraph]],
        {"general-purpose": (GENERAL_PURPOSE_CONFIG, _FakeSubgraph())},
    )


class _FakeReviewModel:
    """最小异步评审模型：行为由 script 控制。"""

    def __init__(self, script: list[dict] | None = None) -> None:
        self.script = list(script or [])

    async def ainvoke(self, messages, **kwargs) -> AIMessage:
        step = self.script.pop(0)
        return AIMessage(content=step["content"])


class _FakeModel:
    """最小异步模型：记录输入，返回固定 AIMessage。"""

    def __init__(self, content: str = "ok") -> None:
        self.content = content
        self.inputs: list[list] = []

    async def ainvoke(self, messages, config=None, **kwargs) -> AIMessage:
        self.inputs.append(messages)
        return AIMessage(content=self.content)


# ---------------------------------------------------------------------------
# supervisor_router_node
# ---------------------------------------------------------------------------


class TestSupervisorRouter:
    def _run(self, node: Any, state: dict) -> dict:
        """异步节点统一执行入口（节点签名为 (state, config)）。"""
        async def _main():
            return await node(state, config={})

        return asyncio.run(_main())

    def test_valid_delegation_sets_next_and_task(self):
        """合法委派：解析 route_to_agent，写 next / last_task。"""
        router: Any = _make_supervisor_router(_registry())
        state = {"messages": [_route_message("general-purpose", "写代码")]}

        result = self._run(router, state)

        assert result["next"] == "general-purpose"
        assert result["last_task"] == "写代码"
        # 工具调用消息被移除，避免残留导致模型 API 400
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], RemoveMessage)
        assert result["messages"][0].id == "call-1"

    def test_invalid_target_keeps_next_none(self):
        """非法子代理名：兜底 next=None，不抛异常，工具调用消息仍被清理。"""
        router: Any = _make_supervisor_router(_registry())
        state = {"messages": [_route_message("no-such-agent")]}

        result = self._run(router, state)

        assert result["next"] is None
        assert "last_task" not in result
        assert len(result["messages"]) == 1

    def test_no_tool_call_keeps_messages(self):
        """直接回答：无工具调用 → next=None，不动消息。"""
        router: Any = _make_supervisor_router(_registry())
        state = {"messages": [AIMessage(content="直接回答")]}

        result = self._run(router, state)

        assert result["next"] is None
        assert "messages" not in result


# ---------------------------------------------------------------------------
# 规则检查
# ---------------------------------------------------------------------------


class TestRuleReview:
    def test_no_last_ai_fails(self):
        passed, feedback = _rule_review("task", None)
        assert passed is False
        assert feedback == "无有效回答"

    def test_empty_content_fails(self):
        passed, feedback = _rule_review("task", AIMessage(content=""))
        assert passed is False
        assert feedback == "无有效回答"

    def test_too_short_fails(self):
        passed, feedback = _rule_review("task", AIMessage(content="太短"))
        assert passed is False
        assert feedback == "回答过短"

    def test_valid_answer_passes(self):
        passed, feedback = _rule_review(
            "task", AIMessage(content="这是一个完整的回答。")
        )
        assert passed is True
        assert feedback is None


# ---------------------------------------------------------------------------
# 条件路由（含 else 兜底）
# ---------------------------------------------------------------------------


class TestConditionalRouters:
    def test_route_after_supervisor_valid(self):
        route = _make_route_after_supervisor(_registry())
        assert route(cast(ThreadState, {"next": "general-purpose"})) == "subagent"

    def test_route_after_supervisor_invalid_target_ends(self):
        route = _make_route_after_supervisor(_registry())
        assert route(cast(ThreadState, {"next": "no-such-agent"})) == "__end__"

    def test_route_after_supervisor_no_target_ends(self):
        route = _make_route_after_supervisor(_registry())
        assert route(cast(ThreadState, {"next": None})) == "__end__"

    def test_route_after_reflection_passed_ends(self):
        route = _make_route_after_reflection(max_routes=5)
        assert route(cast(ThreadState, {"reflection_passed": True})) == "__end__"

    def test_route_after_reflection_retries_below_limit(self):
        route = _make_route_after_reflection(max_routes=5)
        assert (
            route(cast(ThreadState, {"reflection_passed": False, "route_count": 1}))
            == "model_call"
        )

    def test_route_after_reflection_force_stop_at_limit(self):
        route = _make_route_after_reflection(max_routes=1)
        assert (
            route(cast(ThreadState, {"reflection_passed": False, "route_count": 1}))
            == "__end__"
        )


# ---------------------------------------------------------------------------
# reflection_node
# ---------------------------------------------------------------------------


class TestReflectionNode:
    def _run(self, node: Any, state: dict) -> dict:
        async def _main():
            return await node(state, config={})

        return asyncio.run(_main())

    def test_last_error_fails_reflection(self):
        node = _make_reflection_node(
            cast(BaseChatModel, _FakeReviewModel()), enable_llm_review=False
        )
        result = self._run(node, {"last_error": "子代理超时"})

        assert result["reflection_passed"] is False
        assert "子代理超时" in result["reflection_feedback"]
        assert result["route_count"] == 1

    def test_rule_check_passes_without_llm_review(self):
        node = _make_reflection_node(
            cast(BaseChatModel, _FakeReviewModel()), enable_llm_review=False
        )
        result = self._run(
            node,
            {
                "last_task": "写代码",
                "messages": [AIMessage(content="这是一个完整且详细的回答。")],
            },
        )

        assert result["reflection_passed"] is True

    def test_llm_review_pass(self):
        review = _FakeReviewModel(script=[{"content": "PASS|回答完整"}])
        node = _make_reflection_node(
            cast(BaseChatModel, review), enable_llm_review=True
        )
        result = self._run(
            node,
            {
                "last_task": "写代码",
                "messages": [AIMessage(content="这是一个完整且详细的回答。")],
            },
        )

        assert result["reflection_passed"] is True

    def test_llm_review_fail(self):
        review = _FakeReviewModel(script=[{"content": "FAIL|缺少实现细节"}])
        node = _make_reflection_node(
            cast(BaseChatModel, review), enable_llm_review=True
        )
        result = self._run(
            node,
            {
                "last_task": "写代码",
                "messages": [AIMessage(content="这是一个完整且详细的回答。")],
            },
        )

        assert result["reflection_passed"] is False
        assert "缺少实现细节" in result["reflection_feedback"]

    def test_llm_review_exception_falls_back_to_rule(self):
        class _BrokenReview:
            async def ainvoke(self, messages, **kwargs):
                raise RuntimeError("评审模型挂了")

        node = _make_reflection_node(
            cast(BaseChatModel, _BrokenReview()), enable_llm_review=True
        )
        result = self._run(
            node,
            {
                "last_task": "写代码",
                "messages": [AIMessage(content="这是一个完整且详细的回答。")],
            },
        )

        # 评审异常不致命：回退到规则检查结果（回答合法 → 通过）
        assert result["reflection_passed"] is True

    def test_exhausted_writes_last_error(self):
        """达到评审上限仍不通过 → 写 last_error，强制结束标记。"""
        node = _make_reflection_node(
            cast(BaseChatModel, _FakeReviewModel()),
            enable_llm_review=False,
            max_routes=3,
        )
        result = self._run(
            node,
            {"route_count": 2, "messages": [AIMessage(content="太短")]},
        )

        assert result["reflection_passed"] is False
        assert result["route_count"] == 3
        assert "上限" in result["last_error"]

    def test_below_limit_no_last_error(self):
        """未达上限时只回环，不写 last_error。"""
        node = _make_reflection_node(
            cast(BaseChatModel, _FakeReviewModel()),
            enable_llm_review=False,
            max_routes=5,
        )
        result = self._run(
            node,
            {"route_count": 1, "messages": [AIMessage(content="太短")]},
        )

        assert result["reflection_passed"] is False
        assert result["route_count"] == 2
        assert "last_error" not in result


# ---------------------------------------------------------------------------
# subagent_node
# ---------------------------------------------------------------------------


class TestSubagentNode:
    def _run(self, node: Any, state: dict) -> dict:
        async def _main():
            return await node(state, config={})

        return asyncio.run(_main())

    def test_unknown_target_writes_last_error(self):
        node = _make_subagent_node(_registry())
        result = self._run(node, {"next": "no-such-agent"})

        assert result["next"] is None
        assert "未找到" in result["last_error"]

    def test_success_merges_messages_and_clears_error(self):
        registry = _registry()
        node = _make_subagent_node(registry)
        result = self._run(node, {"next": "general-purpose", "last_error": "旧错误"})

        assert result["next"] is None
        assert result["last_error"] is None
        assert any("子代理完成" in str(m.content) for m in result["messages"])

    def test_timeout_writes_last_error(self):
        class _TimeoutSubgraph:
            async def ainvoke(self, state, config=None):
                raise TimeoutError("timeout")

        registry = cast(
            dict[str, tuple[SubagentConfig, CompiledStateGraph]],
            {"general-purpose": (GENERAL_PURPOSE_CONFIG, _TimeoutSubgraph())},
        )
        node = _make_subagent_node(registry)
        result = self._run(node, {"next": "general-purpose"})

        assert result["next"] is None
        assert "超时" in result["last_error"]

    def test_exception_writes_last_error(self):
        class _BrokenSubgraph:
            async def ainvoke(self, state, config=None):
                raise ValueError("子代理炸了")

        registry = cast(
            dict[str, tuple[SubagentConfig, CompiledStateGraph]],
            {"general-purpose": (GENERAL_PURPOSE_CONFIG, _BrokenSubgraph())},
        )
        node = _make_subagent_node(registry)
        result = self._run(node, {"next": "general-purpose"})

        assert result["next"] is None
        assert "子代理炸了" in result["last_error"]


# ---------------------------------------------------------------------------
# model_call_node
# ---------------------------------------------------------------------------


class TestModelCallNode:
    def _run(self, node: Any, state: dict) -> dict:
        async def _main():
            return await node(state, config={"configurable": {"thread_id": "t1"}})

        return asyncio.run(_main())

    def test_feedback_injected_and_cleared(self):
        model = _FakeModel(content="重排后的回答。")
        node = _make_model_call_node(
            model=cast(Any, model),
            app_config=AppConfig(log_level="warning"),
            system_prompt="supervisor 提示词",
        )
        result = self._run(
            node,
            {
                "messages": [HumanMessage(content="帮我写代码")],
                "reflection_feedback": "上一轮回答不完整",
            },
        )

        # 反馈注入为 SystemMessage，且本轮消费后清空
        call_messages = model.inputs[0]
        assert any("上一轮回答不完整" in str(m.content) for m in call_messages)
        assert result["reflection_feedback"] is None
        assert any("重排后的回答" in str(m.content) for m in result["messages"])

    def test_no_feedback_skips_injection(self):
        model = _FakeModel(content="直接回答。")
        node = _make_model_call_node(
            model=cast(Any, model),
            app_config=AppConfig(log_level="warning"),
            system_prompt="supervisor 提示词",
        )
        result = self._run(node, {"messages": [HumanMessage(content="你好")]})

        call_messages = model.inputs[0]
        assert not any("reflection_feedback" in str(m.content) for m in call_messages)
        assert "reflection_feedback" not in result


# ---------------------------------------------------------------------------
# 图构建与注册表
# ---------------------------------------------------------------------------


class TestBuildSupervisorGraph:
    def test_builds_with_default_registry(self):
        """构建不抛异常；注册表包含内置子代理与关键配置。"""

        class _FakeChatModel:
            def bind_tools(self, tools, **kwargs):
                return self

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch(
                    "my_df.agents.sub_agent.assistant.create_chat_model",
                    return_value=_FakeChatModel(),
                )
            )
            stack.enter_context(
                patch(
                    "my_df.agents.sub_agent.weather_search.create_chat_model",
                    return_value=_FakeChatModel(),
                )
            )
            registry = _build_default_registry(AppConfig(log_level="warning"), tools=[])

        assert set(registry) == {"general-purpose", "weather_search"}
        cfg = registry["general-purpose"][0]
        assert isinstance(cfg, SubagentConfig)
        assert cfg.max_turns > 0
        assert cfg.timeout_seconds > 0

    def test_build_supervisor_graph_compiles(self):
        """模型全部替换为 fake 后，图能正常编译（不触发异步执行）。"""

        class _FakeChatModel:
            def bind_tools(self, tools, **kwargs):
                return self

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch(
                    "my_df.agents.supervisor_graph.create_chat_model",
                    return_value=_FakeChatModel(),
                )
            )
            stack.enter_context(
                patch(
                    "my_df.agents.sub_agent.assistant.create_chat_model",
                    return_value=_FakeChatModel(),
                )
            )
            stack.enter_context(
                patch(
                    "my_df.agents.sub_agent.weather_search.create_chat_model",
                    return_value=_FakeChatModel(),
                )
            )
            graph = build_supervisor_graph(
                AppConfig(log_level="warning"), tools=[], max_routes=5
            )

        assert graph is not None


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_last_ai_message_picks_latest_ai(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="第一条"),
            AIMessage(content="最新一条"),
        ]
        last = _last_ai_message(messages)
        assert last is not None
        assert last.content == "最新一条"

    def test_last_ai_message_skips_empty(self):
        messages = [AIMessage(content=""), AIMessage(content="有内容")]
        last = _last_ai_message(messages)
        assert last is not None
        assert last.content == "有内容"

    def test_filter_tools_whitelist(self):
        class _Tool:
            def __init__(self, name: str) -> None:
                self.name = name

        tools = cast(
            list[BaseTool],
            [_Tool("search_weather"), _Tool("write_todos"), _Tool("task")],
        )
        result = filter_tools(
            tools, allowed_tools=["search_weather"], disallowed_tools=None
        )
        assert [t.name for t in result] == ["search_weather"]

    def test_filter_tools_blacklist(self):
        class _Tool:
            def __init__(self, name: str) -> None:
                self.name = name

        tools = cast(list[BaseTool], [_Tool("search_weather"), _Tool("task")])
        result = filter_tools(tools, allowed_tools=None, disallowed_tools=["task"])
        assert [t.name for t in result] == ["search_weather"]


# ---------------------------------------------------------------------------
# _sanitize_messages：孤儿 tool_calls 清洗（防 400 回归）
# ---------------------------------------------------------------------------


class TestSanitizeMessages:
    def test_orphan_tool_calls_removed(self):
        """带 tool_calls 但无 ToolMessage 响应 → 整条移除。"""
        orphan = _route_message("general-purpose", "写代码", msg_id="call-1")
        assert _sanitize_messages([orphan]) == []

    def test_answered_tool_chain_kept(self):
        """AIMessage(tool_calls) + 对应 ToolMessage → 保留完整链。"""
        ai = _route_message("general-purpose", "写代码", msg_id="call-2")
        tool = ToolMessage(content="完成", tool_call_id="call-2")
        cleaned = _sanitize_messages([ai, tool])
        assert len(cleaned) == 2
        assert cleaned[0].tool_calls[0]["id"] == "call-2"

    def test_mixed_orphan_and_normal_keeps_normal(self):
        """孤儿与正常链混合 → 只保留正常链。"""
        orphan = _route_message("no-such", msg_id="call-1")
        ai = _route_message("general-purpose", msg_id="call-2")
        tool = ToolMessage(content="ok", tool_call_id="call-2")
        cleaned = _sanitize_messages([orphan, ai, tool])
        assert len(cleaned) == 2

    def test_partial_answered_removes_whole_message(self):
        """一条 AIMessage 多个 tool_calls 只响应部分 → 整条移除。"""
        ai = AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {"name": "x", "args": {}, "id": "c1", "type": "tool_call"},
                {"name": "y", "args": {}, "id": "c2", "type": "tool_call"},
            ],
        )
        tool = ToolMessage(content="ok", tool_call_id="c1")
        cleaned = _sanitize_messages([ai, tool])
        # 孤儿 AIMessage 被移除，ToolMessage 本身保留
        assert len(cleaned) == 1
        assert cleaned[0].type == "tool"

    def test_plain_messages_untouched(self):
        """无 tool_calls 的普通消息原样保留。"""
        msgs = [HumanMessage(content="hi"), AIMessage(content="回答")]
        assert _sanitize_messages(msgs) == msgs


# ---------------------------------------------------------------------------
# merge_route_count：覆盖式 reducer（跨 run 不累计回归）
# ---------------------------------------------------------------------------


class TestMergeRouteCount:
    def test_inject_zero_resets_persisted(self):
        """每次 run 注入 0，覆盖 checkpoint 恢复的旧值。"""
        assert merge_route_count(7, 0) == 0

    def test_absolute_value_overwrites(self):
        """reflect 返回绝对值（当前+1），覆盖式生效。"""
        assert merge_route_count(0, 3) == 3

    def test_none_new_keeps_existing(self):
        assert merge_route_count(3, None) == 3

    def test_both_none_returns_zero(self):
        assert merge_route_count(None, None) == 0


# ---------------------------------------------------------------------------
# 事件埋点：route / subagent / reflect 写入 event_store
# ---------------------------------------------------------------------------


class TestEventInstrumentation:
    @staticmethod
    def _events(store: MemoryRunEventStore, run_id: str = "r1") -> list[dict]:
        return asyncio.run(store.list_events("t1", run_id))

    def test_router_writes_route_event(self):
        store = MemoryRunEventStore()
        router = _make_supervisor_router(_registry(), event_store=store)
        asyncio.run(
            router(
                {"messages": [_route_message("general-purpose", "写代码")]},
                config={"configurable": {"thread_id": "t1", "run_id": "r1"}},
            )
        )

        events = self._events(store)
        assert len(events) == 1
        assert events[0]["event_type"] == "route"
        assert events[0]["metadata"]["target"] == "general-purpose"
        assert events[0]["metadata"]["retry"] is False

    def test_subagent_writes_start_and_end(self):
        store = MemoryRunEventStore()
        node = _make_subagent_node(_registry(), event_store=store)
        asyncio.run(
            node(
                {"next": "general-purpose"},
                config={"configurable": {"thread_id": "t1", "run_id": "r1"}},
            )
        )

        events = self._events(store)
        assert [e["event_type"] for e in events] == ["subagent_start", "subagent_end"]
        assert events[0]["metadata"]["target"] == "general-purpose"
        assert events[1]["metadata"]["status"] == "ok"

    def test_reflection_writes_event_with_exhausted(self):
        store = MemoryRunEventStore()
        node = _make_reflection_node(
            cast(BaseChatModel, _FakeReviewModel()),
            enable_llm_review=False,
            event_store=store,
            max_routes=3,
        )
        asyncio.run(
            node(
                {"route_count": 2, "messages": [AIMessage(content="太短")]},
                config={"configurable": {"thread_id": "t1", "run_id": "r1"}},
            )
        )

        events = self._events(store)
        assert len(events) == 1
        assert events[0]["event_type"] == "reflect"
        assert events[0]["metadata"]["passed"] is False
        assert events[0]["metadata"]["round"] == 3
        assert events[0]["metadata"]["exhausted"] is True
