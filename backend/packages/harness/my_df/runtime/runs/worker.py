from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from my_df.agents.config.app_config import AppConfig
from my_df.runtime.runs.manager import RunManager, RunRecord
from my_df.runtime.stream_bridge.base import StreamBridge


@dataclass(frozen=True)
class RunContext:
    """Infrastructure dependencies for a single agent run.

    Groups checkpointer, store, and persistence-related singletons so that
    ``run_agent`` (and any future callers) receive one object instead of a
    growing list of keyword arguments.
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

    agent = agent_factory(config=config)

    async for chunk in agent.astream(graph_input, config=config):
        process_chunk(chunk)


def process_chunk(chunk: dict):
    print(chunk)
