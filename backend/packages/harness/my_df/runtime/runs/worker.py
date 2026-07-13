"""Agent 运行器：在后台执行 LangGraph agent 并处理输出。"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from my_df.agents.config.app_config import AppConfig
from my_df.runtime.runs.manager import RunManager, RunRecord
from my_df.runtime.stream_bridge.base import StreamBridge


@dataclass(frozen=True)
class RunContext:
    """单次 agent 运行的基础设施依赖。

    将 checkpointer、store 等持久化单例分组，
    使 ``run_agent`` 不必接收不断增长的参数列表。
    """

    checkpointer: Any
    store: Any | None = field(default=None)
    event_store: Any | None = field(default=None)
    run_events_config: Any | None = field(default=None)
    thread_store: Any | None = field(default=None)
    app_config: AppConfig | None = field(default=None)


async def run_agent_mini(
    agent_factory: Any,
    graph_input: dict,
    config: dict,
) -> None:
    """简化版 Agent 运行器：遍历 astream 输出并处理每个 chunk。"""

    async for chunk in agent_factory.astream(graph_input, config=config):
        process_chunk(chunk)


def process_chunk(chunk: dict):
    """处理单个 agent 输出 chunk（当前实现仅打印）。"""
    print(chunk)
