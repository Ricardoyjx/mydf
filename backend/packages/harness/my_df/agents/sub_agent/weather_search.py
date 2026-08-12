from langchain.agents import create_agent
from langchain.tools import BaseTool
from my_df.agents.thread_state import ThreadState
from my_df.config.app_config import AppConfig
from my_df.config.subagent_config import WEATHER_SEARCH_CONFIG, SubagentConfig
from my_df.models.factory import create_chat_model


def make_node_weather_search(
    app_config: AppConfig,
    *,
    config: SubagentConfig = WEATHER_SEARCH_CONFIG,
    tools: list[BaseTool] | None = None,
):
    return create_agent(
        model=create_chat_model(
            name=None if config.model == "inherit" else config.model,
            thinking_enable=False,
            app_config=app_config,
            attach_tracing=False,
        ),
        tools=tools,
        system_prompt=config.system_prompt,
        state_schema=ThreadState,
        name=config.name,
    )
