from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from my_df.agents.thread_state import ThreadState
from my_df.models.factory import create_chat_model
from my_df.agents.config.app_config import AppConfig, get_app_config
from my_df.agents.middlewares.todo_middleware import TodoMiddleware
from my_df.agents.middlewares.runtime_middlewares import (
    build_lead_runtime_middlewares,
)


def _get_runtime_config(config: RunnableConfig) -> dict:
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, dict):
        cfg.update(context)
    return cfg


# NOTE: 中间件链构建函数 — 以下注释规定了各中间件的注册顺序约束
# ThreadDataMiddleware must be before SandboxMiddleware to ensure thread_id is available
# UploadsMiddleware should be after ThreadDataMiddleware to access thread_id
# DanglingToolCallMiddleware patches missing ToolMessages before model sees the history
# SummarizationMiddleware should be early to reduce context before other processing
# TodoListMiddleware should be before ClarificationMiddleware to allow todo management
# TitleMiddleware generates title after first exchange
# MemoryMiddleware queues conversation for memory update (after TitleMiddleware)
# ViewImageMiddleware should be before ClarificationMiddleware to inject image details before LLM
# ToolErrorHandlingMiddleware should be before ClarificationMiddleware to convert tool exceptions to ToolMessages
# ClarificationMiddleware should be last to intercept clarification requests after model calls
# NOTE: 构建完整的中间件链，按依赖顺序注入：动态上下文 → 摘要 → 待办事项 → Token 统计 → 标题生成 → 记忆 → 图像查看 → 子代理限制 → 循环检测 → 自定义 → 安全结束 → 澄清
def _build_middlewares(
    config: RunnableConfig,
    model_name: str | None,
    agent_name: str | None,
    custom_middlewares: list[AgentMiddleware] | None = None,
    *,
    app_config: AppConfig | None = None,
):
    """Build middleware chain based on runtime configuration.

    Args:
        config: Runtime configuration containing configurable options like is_plan_mode.
        agent_name: If provided, MemoryMiddleware will use per-agent memory storage.
        custom_middlewares: Optional list of custom middlewares to inject into the chain.

    Returns:
        List of middleware instances.
    """
    # resolved_app_config = app_config  # or get_app_config()
    middlewares = build_lead_runtime_middlewares(lazy_init=True)

    # # Always inject current date (and optionally memory) as <system-reminder> into the
    # # first HumanMessage to keep the system prompt fully static for prefix-cache reuse.
    # from my_df.agents.middlewares.dynamic_context_middleware import (
    #     DynamicContextMiddleware,
    # )

    # middlewares.append(
    #     DynamicContextMiddleware(agent_name=agent_name, app_config=resolved_app_config)
    # )

    # # add summarization middleware if enabled
    # summmary_middleware = _create_summarization_middleware(
    #     app_config=resolved_app_config
    # )
    # if summmary_middleware is not None:
    #     middlewares.append(summmary_middleware)

    # add todo list middleware if plan mode is enabled
    cfg = _get_runtime_config(config)
    is_plan_mode = cfg.get("is_plan_mode", False)
    todo_list_middleware = _create_todo_list_middleware(is_plan_mode)
    if todo_list_middleware is not None:
        middlewares.append(todo_list_middleware)

    # # add tokenUsageMiddleware when token_usage tracking is enabled
    # if resolved_app_config.token_usage_tracking_enabled:
    #     middlewares.append(TokenUsageMiddleware())

    # # Add TitleMiddleware
    # middlewares.append(TitleMiddleware(app_config=resolved_app_config))

    # # Add MemoryMiddleware (after TitleMiddleware)
    # middlewares.append(
    #     MemoryMiddleware(
    #         agent_name=agent_name, memory_config=resolved_app_config.memory
    #     )
    # )

    # # Add ViewImageMiddleware only if the current model supports vision.
    # # Use the resolved runtime model_name from make_lead_agent to avoid stale config values.
    # model_config = (
    #     resolved_app_config.get_model_config(model_name) if model_name else None
    # )
    # if model_config is not None and model_config.supports_vision:
    #     middlewares.append(ViewImageMiddleware())

    # # Add DeferredToolFilterMiddleware to hide deferred tool schemas from model binding
    # if resolved_app_config.tool_search.enabled:
    #     from deerflow.agents.middlewares.deferred_tool_filter_middleware import (
    #         DeferredToolFilterMiddleware,
    #     )

    #     middlewares.append(DeferredToolFilterMiddleware())

    # # Add SubagentLimitMiddleware to truncate excess parallel task calls
    # subagent_enabled = cfg.get("subagent_enabled", False)
    # if subagent_enabled:
    #     max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
    #     middlewares.append(
    #         SubagentLimitMiddleware(max_concurrent=max_concurrent_subagents)
    #     )

    # # LoopDetectionMiddleware — detect and break repetitive tool call loops
    # loop_detection_config = resolved_app_config.loop_detection
    # if loop_detection_config.enabled:
    #     middlewares.append(LoopDetectionMiddleware.from_config(loop_detection_config))

    # # Inject custom middlewares before ClarificationMiddleware
    # if custom_middlewares:
    #     middlewares.extend(custom_middlewares)

    # # SafetyFinishReasonMiddleware — suppress tool execution when the provider
    # # safety-terminated the response. Registered after custom middlewares so
    # # that LangChain's reverse-order after_model dispatch runs Safety first;
    # # cleared tool_calls then flow through Loop/Subagent accounting without
    # # firing extra alarms. See safety_finish_reason_middleware.py docstring.
    # safety_config = resolved_app_config.safety_finish_reason
    # if safety_config.enabled:
    #     middlewares.append(SafetyFinishReasonMiddleware.from_config(safety_config))

    # # ClarificationMiddleware should always be last
    # middlewares.append(ClarificationMiddleware())
    return middlewares


# NOTE: 工厂函数：根据 is_plan_mode 标志创建 TodoMiddleware 待办事项中间件
def _create_todo_list_middleware(is_plan_mode: bool) -> TodoMiddleware | None:
    """Create and configure the TodoList middleware.

    Args:
        is_plan_mode: Whether to enable plan mode with TodoList middleware.

    Returns:
        TodoMiddleware instance if plan mode is enabled, None otherwise.
    """
    if not is_plan_mode:
        return None

    # Custom prompts matching DeerFlow's style
    system_prompt = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly

**When to Use:**
This tool is designed for complex objectives that require systematic tracking:
- Complex multi-step tasks requiring 3+ distinct steps
- Non-trivial tasks needing careful planning and execution
- User explicitly requests a todo list
- User provides multiple tasks (numbered or comma-separated list)
- The plan may need revisions based on intermediate results

**When NOT to Use:**
- Single, straightforward tasks
- Trivial tasks (< 3 steps)
- Purely conversational or informational requests
- Simple tool calls where the approach is obvious

**Best Practices:**
- Break down complex tasks into smaller, actionable steps
- Use clear, descriptive task names
- Remove tasks that become irrelevant
- Add new tasks discovered during implementation
- Don't be afraid to revise the todo list as you learn more

**Task Management:**
Writing todos takes time and tokens - use it when helpful for managing complex problems, not for simple requests.
</todo_list_system>
"""

    tool_description = """Use this tool to create and manage a structured task list for complex work sessions.

**IMPORTANT: Only use this tool for complex tasks (3+ steps). For simple requests, just do the work directly.**

## When to Use

Use this tool in these scenarios:
1. **Complex multi-step tasks**: When a task requires 3 or more distinct steps or actions
2. **Non-trivial tasks**: Tasks requiring careful planning or multiple operations
3. **User explicitly requests todo list**: When the user directly asks you to track tasks
4. **Multiple tasks**: When users provide a list of things to be done
5. **Dynamic planning**: When the plan may need updates based on intermediate results

## When NOT to Use

Skip this tool when:
1. The task is straightforward and takes less than 3 steps
2. The task is trivial and tracking provides no benefit
3. The task is purely conversational or informational
4. It's clear what needs to be done and you can just do it

## How to Use

1. **Starting a task**: Mark it as `in_progress` BEFORE beginning work
2. **Completing a task**: Mark it as `completed` IMMEDIATELY after finishing
3. **Updating the list**: Add new tasks, remove irrelevant ones, or update descriptions as needed
4. **Multiple updates**: You can make several updates at once (e.g., complete one task and start the next)

## Task States

- `pending`: Task not yet started
- `in_progress`: Currently working on (can have multiple if tasks run in parallel)
- `completed`: Task finished successfully

## Task Completion Requirements

**CRITICAL: Only mark a task as completed when you have FULLY accomplished it.**

Never mark a task as completed if:
- There are unresolved issues or errors
- Work is partial or incomplete
- You encountered blockers preventing completion
- You couldn't find necessary resources or dependencies
- Quality standards haven't been met

If blocked, keep the task as `in_progress` and create a new task describing what needs to be resolved.

## Best Practices

- Create specific, actionable items
- Break complex tasks into smaller, manageable steps
- Use clear, descriptive task names
- Update task status in real-time as you work
- Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
- Remove tasks that are no longer relevant
- **IMPORTANT**: When you write the todo list, mark your first task(s) as `in_progress` immediately
- **IMPORTANT**: Unless all tasks are completed, always have at least one task `in_progress` to show progress

Being proactive with task management demonstrates thoroughness and ensures all requirements are completed successfully.

**Remember**: If you only need a few tool calls to complete a task and it's clear what to do, it's better to just do the task directly and NOT use this tool at all.
"""

    return TodoMiddleware(
        system_prompt=system_prompt, tool_description=tool_description
    )


# NOTE: LangGraph 图工厂入口函数，保持与 LangGraph Server 兼容的签名
def make_lead_agent(config: RunnableConfig):
    runtime_config = _get_runtime_config(config)
    runtime_app_config = runtime_config.get("app_config")
    return _make_lead_agent(config, app_config=runtime_app_config or get_app_config())


# NOTE: Lead Agent 核心工厂：解析运行时配置 → 模型名称 → 注入追踪回调 → 组装工具链 → 应用提示词模板 → 创建 Agent
def _make_lead_agent(config: RunnableConfig, *, app_config: AppConfig):
    model_name = "deepseek-v4-flash"
    agent_name = "lead_agent"

    tools: list[BaseTool] = []

    return create_agent(
        model=create_chat_model(
            name=model_name,
            thinking_enable=False,
            app_config=AppConfig(),
            attach_tracing=False,
        ),
        tools=tools,
        middleware=_build_middlewares(
            config,
            model_name=model_name,
            agent_name=agent_name,
            app_config=app_config,
        ),
        system_prompt="",
        state_schema=ThreadState,
    )
