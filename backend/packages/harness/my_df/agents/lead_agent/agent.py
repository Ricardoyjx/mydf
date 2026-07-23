"""Lead Agent 工厂：构建具备完整中间件链的默认 Agent。"""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from my_df.agents.thread_state import ThreadState
from my_df.models.factory import create_chat_model
from my_df.agents.config.app_config import AppConfig, get_app_config
from my_df.agents.middlewares.todo_middleware import TodoMiddleware
from my_df.agents.middlewares.memory_middleware import MemoryMiddleware
from my_df.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware
from my_df.agents.middlewares.runtime_middlewares import (
    build_lead_runtime_middlewares,
)

logger = logging.getLogger(__name__)


def _get_runtime_config(config: RunnableConfig) -> dict:
    """从 RunnableConfig 中提取运行时配置（合并 configurable 与 context）。"""
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, dict):
        cfg.update(context)
    return cfg


def _build_middlewares(
    config: RunnableConfig,
    model_name: str | None,
    agent_name: str | None,
    custom_middlewares: list[AgentMiddleware] | None = None,
    *,
    app_config: AppConfig | None = None,
):
    """根据运行时配置构建中间件链。

    中间件注册顺序（按执行先后）：
    1. 运行时基础中间件（预留扩展）
    2. MemoryMiddleware：加载持久化记忆，注入到首条 HumanMessage（调用后回写）
    3. DynamicContextMiddleware：每次模型调用前注入当前日期时间
    4. TodoMiddleware：仅 ``is_plan_mode=True`` 时注册

    参数：
        config:            运行配置，包含 user_id 等。
        model_name:        当前模型名称。
        agent_name:        代理名称。
        custom_middlewares: 可选的自定义中间件列表，注入到链中。
        app_config:        应用配置（用于读取 is_plan_mode 等）。

    返回：
        中间件实例列表。
    """
    middlewares = build_lead_runtime_middlewares(lazy_init=True)

    # 从 config 中提取 user_id，供 MemoryMiddleware 按用户隔离记忆
    cfg = _get_runtime_config(config)
    user_id = cfg.get("user_id", "default")

    # MemoryMiddleware：加载持久化记忆，注入到首条 HumanMessage
    middlewares.append(MemoryMiddleware(agent_name=agent_name, user_id=user_id))

    # DynamicContextMiddleware：每次模型调用前注入当前日期时间
    middlewares.append(
        DynamicContextMiddleware(agent_name=agent_name, app_config=app_config)
    )

    # TodoMiddleware：仅 is_plan_mode=True 时注册
    is_plan_mode = cfg.get(
        "is_plan_mode",
        app_config.is_plan_mode if app_config is not None else False,
    )
    todo_list_middleware = _create_todo_list_middleware(is_plan_mode)
    if todo_list_middleware is not None:
        middlewares.append(todo_list_middleware)  # type: ignore

    return middlewares


def _create_todo_list_middleware(is_plan_mode: bool) -> TodoMiddleware | None:
    """工厂函数：根据 is_plan_mode 创建 TodoMiddleware 实例。"""
    if not is_plan_mode:
        return None

    system_prompt = """
<todo_list_system>
你有 `write_todos` 工具，用于管理和跟踪复杂多步骤目标。

**关键规则：**
- 每完成一步后**立即**标记为 completed——不要批量完成
- 任何时候只保留**恰好一个**任务为 `in_progress`（除非可并行执行）
- 实时更新待办列表——让用户了解你的进度
- 简单任务（< 3 步）不要使用此工具，直接完成即可

**使用时机：**
- 需要 3 步以上的复杂多步骤任务
- 需要仔细规划和执行的非平凡任务
- 用户明确要求任务列表
- 用户提供多项任务（编号列表或逗号分隔列表）
- 计划可能需要根据中间结果进行调整

**不使用时机：**
- 单步简单任务
- 纯对话或信息查询
- 方案显而易见的单次工具调用

**最佳实践：**
- 将复杂任务分解为可操作的步骤
- 使用清晰、描述性的任务名称
- 移除已无关的任务
- 发现新任务时添加
- 随着工作进展随时修改待办列表
</todo_list_system>
"""

    tool_description = """创建和管理结构化任务列表，用于复杂工作会话。

**重要：仅适用于复杂任务（3 步以上）。简单请求直接执行即可。**

## 使用时机

1. **复杂多步骤任务**：需要 3 步或更多操作
2. **非平凡任务**：需要仔细规划或多步操作
3. **用户明确要求任务列表**
4. **多项任务**：用户提供了任务列表
5. **动态规划**：计划可能根据中间结果调整

## 不使用时机

1. 任务简单直接，少于 3 步
2. 任务微不足道，追踪没有意义
3. 纯对话或信息查询
4. 方案显而易见

## 使用方法

1. **开始任务**：开始工作前标记为 `in_progress`
2. **完成任务**：完成后立即标记为 `completed`
3. **更新列表**：根据需要添加、移除或更新任务
4. **批量更新**：可一次执行多项更新

## 任务状态

- `pending`：尚未开始
- `in_progress`：正在执行
- `completed`：成功完成
"""

    return TodoMiddleware(
        system_prompt=system_prompt, tool_description=tool_description
    )


def make_lead_agent(config: RunnableConfig):
    """Lead Agent 工厂入口函数。"""
    runtime_config = _get_runtime_config(config)
    runtime_app_config = runtime_config.get("app_config")
    return _make_lead_agent(config, app_config=runtime_app_config or get_app_config())


def _make_lead_agent(config: RunnableConfig, *, app_config: AppConfig):
    """Lead Agent 核心工厂。"""
    agent_name = "lead_agent"

    if not app_config.models:
        logger.warning(
            "未配置模型（app_config.models 为空），"
            "请检查 .env 中的 %s 或 DEEPSEEK_API_KEY 环境变量。",
            "MYDF_LLM_API_KEY",
        )

    tools: list[BaseTool] = []

    return create_agent(
        model=create_chat_model(
            name=None,
            thinking_enable=False,
            app_config=app_config,
            attach_tracing=False,
        ),
        tools=tools,
        middleware=_build_middlewares(
            config,
            model_name=app_config.models[0].name if app_config.models else None,
            agent_name=agent_name,
            app_config=app_config,
        ),
        system_prompt="",
        state_schema=ThreadState,
    )
