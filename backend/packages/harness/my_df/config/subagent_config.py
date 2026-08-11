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
