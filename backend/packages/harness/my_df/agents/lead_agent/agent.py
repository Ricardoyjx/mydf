"""Lead Agent 工厂：构建具备完整中间件链的默认 Agent。"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from my_df.agents.config.model_config import ModelConfig
from my_df.agents.thread_state import ThreadState
from my_df.models.factory import create_chat_model
from my_df.agents.config.app_config import AppConfig, get_app_config
from my_df.agents.middlewares.todo_middleware import TodoMiddleware
from my_df.agents.middlewares.runtime_middlewares import (
    build_lead_runtime_middlewares,
)


def _get_runtime_config(config: RunnableConfig) -> dict:
    """从 RunnableConfig 中提取运行时配置（合并 configurable 与 context）。"""
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, dict):
        cfg.update(context)
    return cfg


# 中间件链构建函数 — 以下注释规定了各中间件的注册顺序约束
# ThreadDataMiddleware 必须在 SandboxMiddleware 之前，确保 thread_id 可用
# UploadsMiddleware 应在 ThreadDataMiddleware 之后，以获得 thread_id
# DanglingToolCallMiddleware 在模型看到历史前修补缺失的 ToolMessages
# SummarizationMiddleware 应尽早执行以减少上下文长度
# TodoListMiddleware 应在 ClarificationMiddleware 之前，允许待办管理
# TitleMiddleware 在首次对话后生成标题
# MemoryMiddleware 在 TitleMiddleware 之后加入对话记忆队列
# ViewImageMiddleware 应在 ClarificationMiddleware 之前注入图像详情
# ToolErrorHandlingMiddleware 应在 ClarificationMiddleware 之前转换工具异常
# ClarificationMiddleware 应始终在最后，截获模型调用后的澄清请求
def _build_middlewares(
    config: RunnableConfig,
    model_name: str | None,
    agent_name: str | None,
    custom_middlewares: list[AgentMiddleware] | None = None,
    *,
    app_config: AppConfig | None = None,
):
    """根据运行时配置构建中间件链。

    参数：
        config:            运行配置，包含 is_plan_mode 等可配置选项。
        model_name:        当前模型名称（用于中间件中的模型感知逻辑）。
        agent_name:        代理名称（MemoryMiddleware 用于按代理隔离记忆）。
        custom_middlewares: 可选的自定义中间件列表，注入到链中。

    返回：
        中间件实例列表。
    """
    middlewares = build_lead_runtime_middlewares(lazy_init=True)

    # 若启用计划模式，添加 TodoMiddleware
    cfg = _get_runtime_config(config)
    is_plan_mode = cfg.get("is_plan_mode", False)
    todo_list_middleware = _create_todo_list_middleware(is_plan_mode)
    if todo_list_middleware is not None:
        middlewares.append(todo_list_middleware)  # type: ignore

    return middlewares


def _create_todo_list_middleware(is_plan_mode: bool) -> TodoMiddleware | None:
    """工厂函数：根据 is_plan_mode 创建 TodoMiddleware 实例。

    参数：
        is_plan_mode: 是否启用计划模式。

    返回：
        TodoMiddleware 实例（启用时）或 None（禁用时）。
    """
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
    """Lead Agent 工厂入口函数，保持与 LangGraph Server 兼容的签名。"""
    runtime_config = _get_runtime_config(config)
    runtime_app_config = runtime_config.get("app_config")
    return _make_lead_agent(config, app_config=runtime_app_config or get_app_config())


def _make_lead_agent(config: RunnableConfig, *, app_config: AppConfig):
    """Lead Agent 核心工厂：解析运行时配置 → 创建模型 → 组装工具链 → 创建 Agent。"""
    model_name = "deepseek-v4-flash"
    agent_name = "lead_agent"

    tools: list[BaseTool] = []

    return create_agent(
        model=create_chat_model(
            name=model_name,
            thinking_enable=False,
            app_config=AppConfig(
                models=[
                    ModelConfig(
                        name="deepseek-v4-flash",
                        model="deepseek-v4-flash",
                        use="langchain_deepseek.ChatDeepSeek",
                    ),
                ]
            ),
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
