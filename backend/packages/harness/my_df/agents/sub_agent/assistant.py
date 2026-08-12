import logging

from langchain.agents import create_agent
from langchain.tools import BaseTool
from my_df.agents.thread_state import ThreadState
from my_df.config.app_config import AppConfig
from my_df.config.subagent_config import SubagentConfig
from my_df.models.factory import create_chat_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是通用助手（sub agent）。
职责：回答 supervisor 委派给你的具体任务。
注意：你不需要注入记忆或检索知识库，那些由 supervisor 负责。
完成任务后直接输出最终回答。"""


GENERAL_PURPOSE_CONFIG = SubagentConfig(
    name="general-purpose",
    description="""A capable agent for complex, multi-step tasks.
    
Use this subagent when:
- The task requires both exploration and modification
- Complex reasoning is needed to interpret results
- Multiple dependent steps must be executed
- The task would benefit from isolated context management

Do NOT use for simple, single-step operations.""",
    system_prompt="""You are a general-purpose subagent working on a delegated task.

<guidelines>
- Focus on completing the delegated task efficiently
- Use available tools as needed to accomplish the goal
- Think step by step but act decisively
- Return a concise summary of what you accomplished
- Do NOT ask for clarification - work with the information provided
</guidelines>

<output_format>
When you complete the task, provide:
1. A brief summary of what was accomplished
2. Key findings or results
3. Any relevant file paths, data, or artifacts created
4. Issues encountered (if any)
5. Citations: Use `[citation:Title](URL)` format for external sources
</output_format>
""",
    tools=None,  # 继承所有工具
    disallowed_tools=["task", "ask_clarification", "present_files"],
    model="inherit",
    max_turns=100,
)


def filter_tools(
    all_tools: list[BaseTool],
    allowed_tools: list[str] | None,
    disallowed_tools: list[str] | None,
) -> list[BaseTool]:
    """按子代理配置过滤工具：先白名单（allowed_tools），再黑名单（disallowed_tools）。"""
    filtered = all_tools

    if allowed_tools is not None:
        allowed_set = set(allowed_tools)
        filtered = [t for t in filtered if t.name in allowed_set]

    if disallowed_tools is not None:
        disallowed_set = set(disallowed_tools)
        filtered = [t for t in filtered if t.name not in disallowed_set]

    return filtered


def make_assistant_subagent(
    app_config: AppConfig,
    *,
    config: SubagentConfig | None = None,
    tools: list[BaseTool] | None = None,
):
    """根据 SubagentConfig 构建子代理图。

    - model：配置为 ``"inherit"``（或 None）时使用默认模型；否则按名称查找模型配置。
    - system_prompt：优先使用 ``config.system_prompt``，为空时回退模块级 SYSTEM_PROMPT。
    """
    sub_config = config or GENERAL_PURPOSE_CONFIG
    model_name = sub_config.model if sub_config.model != "inherit" else None
    return create_agent(
        model=create_chat_model(
            name=model_name,
            thinking_enable=False,
            app_config=app_config,
            attach_tracing=False,
        ),
        tools=filter_tools(
            tools or [], sub_config.tools, sub_config.disallowed_tools
        ),
        system_prompt=sub_config.system_prompt or SYSTEM_PROMPT,
        state_schema=ThreadState,
        name=sub_config.name,
    )
