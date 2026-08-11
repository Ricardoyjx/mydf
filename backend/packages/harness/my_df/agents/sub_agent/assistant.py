import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain.agents import create_agent
from langchain.tools import BaseTool
from my_df.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from my_df.config.app_config import AppConfig
from my_df.config.subagent_config import SubagentConfig
from my_df.models.factory import create_chat_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是通用助手（sub agent）。
职责：回答 supervisor 委派给你的具体任务。
注意：你不需要注入记忆或检索知识库，那些由 supervisor 负责。
完成任务后直接输出最终回答。"""


class SubagentStauts(Enum):
    """Status of a subagent execution."""

    PENDING = "pending"  # 待执行
    RUNNING = "running"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 执行失败
    CANCELLED = "cancelled"  # 已取消
    TIMED_OUT = "timed_out"  # 执行超时


@dataclass
class SubagentResult:
    """Result of a subagent execution."""

    task_id: str  # 任务唯一标识
    trace_id: str  # 分布式追踪 ID
    status: SubagentStauts  # 执行状态
    result: str | None = None  # 执行结果（成功时）
    error: str | None = None  # 错误信息（失败时）
    started_at: datetime | None = None  # 开始时间
    completed_at: datetime | None = None  # 完成时间
    ai_messages: list[dict] | None = None  # AI 消息记录（用于实时流）
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )  # 协作取消信号


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


class SubagentExecutor:
    """Executor for running subagent"""

    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        parent_model: str | None = None,
        sandbox_state: SandboxState | None = None,
        thread_data: ThreadDataState | None = None,  # 线程数据
        thread_id: str | None = None,  # 线程 ID
        trace_id: str | None = None,  # 追踪 ID（用于分布式追踪）
    ):
        self.config = config
        self.parent_model = parent_model
        self.sandbox_state = sandbox_state
        self.thread_data = thread_data
        self.thread_id = thread_id
        self.trace_id = trace_id or str(uuid.uuid4())[:8]

        # 根据配置过滤工具
        self.tools = _filter_tools(
            tools,
            config.tools,  # 白名单
            config.disallowed_tools,  # 黑名单
        )


def _filter_tools(
    all_tools: list[BaseTool],
    allowed_tools: list[str] | None,
    disallowed_tools: list[str] | None,
) -> list[BaseTool]:
    """根据配置过滤工具"""
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
    tools: list[BaseTool] | None = None,
):
    return create_agent(
        model=create_chat_model(
            name="deepseek-v4-flash",
            thinking_enable=False,
            app_config=app_config,
            attach_tracing=False,
        ),
        tools=tools or [],
        system_prompt=SYSTEM_PROMPT,
        state_schema=ThreadState,
        name="assistant",
    )
