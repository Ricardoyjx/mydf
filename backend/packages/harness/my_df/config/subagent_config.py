from dataclasses import dataclass, field


@dataclass
class SubagentConfig:
    """configuration for a subagent"""

    name: str
    description: str
    system_prompt: str
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900


WEATHER_SEARCH_CONFIG = SubagentConfig(
    name="weather_search",
    description="A subagent for weather search",
    system_prompt="""
    You are a weather research assistant.
    You are given a task to research the weather in a city.
    You will be given a city name and a date.
    You will need to research the weather in the city on the date.""",
    tools=["search_weather"],
    model="inherit",
    max_turns=5,
    timeout_seconds=300,
)
