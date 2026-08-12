from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from my_df.agents.sub_agent.assistant import filter_tools
from my_df.agents.thread_state import ThreadState
from my_df.config.app_config import AppConfig
from my_df.config.subagent_config import WEATHER_SEARCH_CONFIG, SubagentConfig
from my_df.models.factory import create_chat_model


def make_node_weather_search(
    app_config: AppConfig,
    *,
    config: SubagentConfig = WEATHER_SEARCH_CONFIG,
    tools: list[BaseTool] | None = None,
    model: BaseChatModel | None = None,
):
    """天气查询子代理工厂；model 为可选共享实例（None 时按配置创建）。"""
    if model is None:
        model = create_chat_model(
            name=None if config.model == "inherit" else config.model,
            thinking_enable=False,
            app_config=app_config,
            attach_tracing=False,
        )
    return create_agent(
        model=model,
        tools=filter_tools(tools or [], config.tools, config.disallowed_tools),
        system_prompt=config.system_prompt,
        state_schema=ThreadState,
        name=config.name,
    )
