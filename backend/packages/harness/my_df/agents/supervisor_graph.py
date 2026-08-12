"""Supervisor 顶层编排图：模型调用 → 路由委派 → 子代理执行 → 质量评审。

架构：
    START → model_call（supervisor LLM，绑定 route_to_agent 工具）
          → supervisor（纯函数：解析 tool_call，写 next / last_task）
          → 条件边 supervisor_router：合法委派 → sub_agent；否则 END
    sub_agent → reflect（规则检查 + 可选 LLM 评审）
          → 条件边 reflect_router：通过 → END；不通过且未超限 → model_call
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from langchain.tools import BaseTool, tool
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import START, CompiledStateGraph, StateNode
from langgraph.store.base import BaseStore

from my_df.agents.sub_agent.assistant import (
    GENERAL_PURPOSE_CONFIG,
    make_assistant_subagent,
)
from my_df.agents.thread_state import ThreadState
from my_df.config.app_config import AppConfig
from my_df.config.subagent_config import SubagentConfig
from my_df.models.factory import create_chat_model
from my_df.runtime.milvus.base import MilvusStorage

logger = logging.getLogger(__name__)

# 路由上限

MAX_ROUTES = 5

SUPERVISOR_SYSTEM_PROMPT = """你是 my-df 的顶层编排者（Supervisor）。

职责：
- 判断当前请求应该直接回答，还是委派给专业子代理处理。
- 需要委派时，调用 route_to_agent 工具，明确指定 agent_name 和 task。
- task 必须完整：包含用户原始需求、必要上下文、期望输出，便于子代理独立完成。

可用子代理：
{registry_desc}

决策规则：
- 简单问答、闲聊、信息查询直接回答，不要委派。
- 复杂多步任务、需要隔离上下文的任务，委派给合适的子代理。
- 若收到质量评审反馈（reflection_feedback），根据反馈重新委派或直接补答。"""


@tool
def route_to_agent(agent_name: str, task: str) -> str:
    """把任务委派给指定子代理执行
    参数：
        agent_name: 子代理名称
        task:       任务描述

    """
    return f"委派给 {agent_name}:{task}"


def _get_runtime_config(config: RunnableConfig) -> dict:
    """从 RunnableConfig 中提取运行时配置（合并 configurable 与 context）。"""
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}

    if isinstance(context, dict):
        cfg.update(context)
    return cfg


def _message_text(msg: Any) -> str:
    """提取消息的纯文本内容"""
    if msg is None:
        return ""
    return str(getattr(msg, "content", "") or "").strip()


def _last_ai_message(messages: list[Any]) -> AIMessage | None:

    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            text = _message_text(msg)
            if text:
                return msg

    return None


# ----节点实现


def _make_model_call_node(
    *,
    model: Runnable[LanguageModelInput, AIMessage],
    app_config: AppConfig,
    store: BaseStore | None = None,
    milvus: MilvusStorage | None = None,
    embedding_model: Any | None = None,
    system_prompt: str | None = None,
) -> StateNode:
    """构建 supervisor LLM 调用节点。
    +
    +    复用 MemoryMiddleware / RagMiddleware 的 abefore_model 注入记忆与知识库
    +    上下文（与 create_agent 中间件链行为一致），再调用绑定 route_to_agent
    +    工具的 supervisor 模型。
    """

    async def model_call_node(
        state: ThreadState, config: RunnableConfig
    ) -> dict[str, Any]:
        messages = state.get("messages") or []

        call_messages: list[AnyMessage] = [SystemMessage(content=system_prompt)]

        # 注入质量评审反馈：让supervisor在重新编排时知道上一轮为何被驳回
        feedback = state.get("reflection_feedback")
        if feedback:
            call_messages.append(
                SystemMessage(
                    content=(
                        "<reflection_feedback>"
                        "上一轮输出未通过质量评审,请据此重新委派或直接补答"
                        f"评审意见：{feedback}"
                        "</refleciton_feedback>"
                    )
                )
            )

        call_messages.extend(messages)

        response = await model.ainvoke(
            call_messages,
            config=config,
        )

        return {"messages": [response]}

    return model_call_node


def _make_supervisor_router(
    registry: dict[str, tuple[SubagentConfig, CompiledStateGraph]],
) -> StateNode:
    """构建 supervisor 纯函数节点：解析 route_to_agent 调用，写 next / last_task。"""

    def supervisor_router_node(state: ThreadState) -> dict[str, Any]:
        target: str | None = None
        task: str | None = None
        route_message_ids: list[str] = []
        for msg in reversed(state.get("messages", [])):
            for call in getattr(msg, "tool_calls", None) or []:
                if call.get("name") == "route_to_agent":
                    args = call.get("args", {})
                    target = args.get("agent_name")
                    task = args.get("task")
                    msg_id = getattr(msg, "id", None)
                    if msg_id:
                        route_message_ids.append(msg_id)
                    break
            if target:
                break

        update: dict[str, Any] = {"next": None}
        # 移除 supervisor 带 tool_calls 的 AIMessage：该工具仅用于声明委派意图、
        # 不真正执行，若残留会导致后续模型调用因“tool_calls 无对应 ToolMessage”
        # 被 API 拒绝（400 invalid_request_error）。
        if route_message_ids:
            update["messages"] = [
                RemoveMessage(id=msg_id) for msg_id in route_message_ids
            ]

        if target and target in registry:
            update.update({"next": target, "last_task": task})
            return update
        if target:
            logger.warning("Agent %s not found", target)
        return update

    return supervisor_router_node


def _make_subagent_node(
    registry: dict[str, tuple[SubagentConfig, CompiledStateGraph]],
) -> StateNode:
    """构建子代理执行节点：以当前 state 运行子图，结果合并回共享 messages。"""

    async def subagent_node(
        state: ThreadState, config: RunnableConfig
    ) -> dict[str, Any]:
        target = state.get("next")
        entry = registry.get(target)  # type: ignore[arg-type]
        if entry is None:
            return {"next": None, "last_error": f"子代理{target}未找到"}

        sub_config, subgraph = entry

        try:
            result = await asyncio.wait_for(
                subgraph.ainvoke(state, config=config),
                timeout=sub_config.timeout_seconds,
            )
            messages = result.get("messages") or []
            return {"messages": messages, "next": None}
        except asyncio.TimeoutError:
            logger.warning(
                "子代理 %s 执行超时（%ss）", target, sub_config.timeout_seconds
            )
            return {
                "next": None,
                "last_error": f"子代理 {target} 执行超时（>{sub_config.timeout_seconds}s）",
            }
        except Exception as e:
            logger.exception("子代理 %s 执行失败", target)
            return {"next": None, "last_error": f"子代理{target}执行失败:{e}"}

    return subagent_node


def _rule_review(
    task: str | None, last_ai: AIMessage | None
) -> tuple[bool, str | None]:
    """零成本规则检查：必须有非空、有结论性的回答。"""

    if last_ai is None:
        return False, "无有效回答"
    text = _message_text(last_ai)
    if not text:
        return False, "无有效回答"
    if len(text) < 10:
        return False, "回答过短"
    return True, None


async def _llm_review(
    review_model: BaseChatModel,
    task: str | None,
    text: str,
) -> tuple[bool, str]:
    """Reflexion 风格 LLM 评审：输出 PASS/FAIL 前缀 + 一句意见。"""
    prompt = f"""
        你是质量评审员，判断子代理是否完成了委派任务。
    任务：{task or "（无）"}
    子代理回答：
    {text[:2000]}
    请评估回答是否完成任务、结论是否清晰。只输出一行：
    PASS|已达标
    或
    FAIL|缺失的具体内容"""

    response = await review_model.ainvoke(
        [
            SystemMessage(content="你是严格但公平的质量评审员。"),
            HumanMessage(content=prompt),
        ]
    )

    raw = _message_text(response)
    passed = raw.upper().startswith("PASS")
    feedback = (
        raw.split("|", 1)[-1].strip() if "|" in raw else raw
    )  # 按照| 分割一次，取右边第二部分再去掉空格符
    return passed, feedback


def _make_reflection_node(
    review_model: BaseChatModel, *, enable_llm_review: bool
) -> StateNode:
    """构建质量评审节点：规则检查 + 可选 LLM 评审，结果写回 state。"""

    async def reflection_node(
        state: ThreadState, config: RunnableConfig
    ) -> dict[str, Any]:
        last_error = state.get("last_error")
        if last_error:
            return {
                "reflection_passed": False,
                "reflection_feedback": f"执行失败:{last_error}",
                "route_count": 1,
            }

        last_ai = _last_ai_message(state.get("messages", []))
        task = state.get("last_task")
        passed, feedback = _rule_review(task, last_ai)

        if passed and enable_llm_review and last_ai is not None:
            try:
                passed, feedback = await _llm_review(
                    review_model, task, _message_text(last_ai)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 审核失败: %s", e)

        logger.info("质量评审: passed=%s, feedback=%s", passed, feedback)

        return {
            "reflection_passed": passed,
            "reflection_feedback": feedback,
            "route_count": 1,
        }

    return reflection_node


# ── 条件边（含 else 兜底）────────────────────────────────────────────


def _make_route_after_supervisor(
    registry: dict[str, tuple[SubagentConfig, CompiledStateGraph]],
) -> Callable[[ThreadState], str]:
    """supervisor 决策后路由：合法委派 → sub_agent；否则 END 兜底。"""

    def route_after_supervisor(state: ThreadState) -> str:
        target = state.get("next")
        if target and target in registry:
            return "subagent"
        return END

    return route_after_supervisor


def _make_route_after_reflection(max_routes: int) -> Callable[[ThreadState], str]:
    """评审后路由：通过 → END；不通过且未超限 → model_call 重新编排。"""

    def route_after_reflection(state: ThreadState) -> str:
        if state.get("reflection_passed"):
            return END
        if state.get("route_count", 0) < max_routes:
            return "model_call"
        logger.warning("达到最大路由轮次%d,强制结束", max_routes)
        return END

    return route_after_reflection


# ── 图构建入口 ──────────────────────────────────────────────────────


def _build_default_registry(
    app_cofig: AppConfig,
    tools: list[BaseTool] | None,
) -> dict[str, tuple[SubagentConfig, CompiledStateGraph]]:
    """构建默认子代理注册表（第一版：general-purpose assistant）。"""

    assistant_graph = make_assistant_subagent(app_cofig, tools=tools or [])
    return {GENERAL_PURPOSE_CONFIG.name: (GENERAL_PURPOSE_CONFIG, assistant_graph)}


def build_supervisor_graph(
    app_config: AppConfig,
    *,
    store: BaseStore | None = None,
    milvus: MilvusStorage | None = None,
    embedding_model: Any | None = None,
    tools: list[BaseTool] | None = None,
    enable_llm_review: bool = True,
    max_routes: int = MAX_ROUTES,
) -> CompiledStateGraph:
    """构建 Supervisor 顶层编排图。
    +
    +    参数：
    +        app_config:       应用配置（模型列表等）。
    +        store:            LangGraph BaseStore（记忆/知识库持久化）。
    +        milvus:           Milvus 向量存储（RAG/语义记忆，可选）。
    +        embedding_model:  Embedding 模型（可选）。
    +        tools:            提供给子代理的工具列表。
    +        enable_llm_review: 是否启用 LLM 质量评审（默认开；关闭时仅规则检查）。
    +        max_routes:       最大委派/评审轮次，防止 supervisor 循环。
    +"""

    supervisor_model = create_chat_model(
        name=None, thinking_enable=False, app_config=app_config, attach_tracing=False
    ).bind_tools([route_to_agent])

    review_model = create_chat_model(
        name=None, thinking_enable=False, app_config=app_config, attach_tracing=False
    )

    registry = _build_default_registry(app_config, tools)
    registry_desc = "\n".join(
        f"-{name}: {cfg.description}" for name, (cfg, _) in registry.items()
    )
    model_name = app_config.models[0].name if app_config.models else "unknown"
    subagent_lines = "\n".join(
        f"    - {name}: model={cfg.model}, max_turns={cfg.max_turns}, "
        f"timeout={cfg.timeout_seconds}s, tools={cfg.tools or 'all'}"
        for name, (cfg, _) in registry.items()
    )
    logger.info(
        "构建 Multi-Agent Supervisor 图: supervisor_model=%s, llm_review=%s, "
        "max_routes=%d, subagents=%d\n%s",
        model_name,
        enable_llm_review,
        max_routes,
        len(registry),
        subagent_lines,
    )
    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(registry_desc=registry_desc)

    builder = StateGraph(ThreadState)
    builder.add_node(
        "model_call",
        _make_model_call_node(
            model=supervisor_model,
            app_config=app_config,
            store=store,
            milvus=milvus,
            embedding_model=embedding_model,
            system_prompt=system_prompt,
        ),
    )

    builder.add_node("supervisor", _make_supervisor_router(registry))
    builder.add_node("subagent", _make_subagent_node(registry))
    builder.add_node(
        "reflect",
        _make_reflection_node(review_model, enable_llm_review=enable_llm_review),
    )

    builder.add_edge(START, "model_call")
    builder.add_edge("model_call", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _make_route_after_supervisor(registry),
        {"subagent": "subagent", END: END},
    )

    builder.add_edge("subagent", "reflect")
    builder.add_conditional_edges(
        "reflect",
        _make_route_after_reflection(max_routes),
        {"model_call": "model_call", END: END},
    )

    graph = builder.compile()
    logger.info(
        "Multi-Agent Supervisor 图构建完成: nodes=model_call/supervisor/subagent/reflect, "
        "subagents=%d, recursion_limit 由运行时 config 控制",
        len(registry),
    )
    return graph
