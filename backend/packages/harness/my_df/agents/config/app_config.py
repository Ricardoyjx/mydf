from pydantic import BaseModel, Field

from my_df.agents.config.model_config import ModelConfig


class AppConfig(BaseModel):
    """Config for the DeerFlow application"""

    log_level: str = Field(
        default="info",
        description="Logging level for deerflow and app modules (debug/info/warning/error); third-party libraries are not affected",
    )
    # token_usage: TokenUsageConfig = Field(
    #     default_factory=TokenUsageConfig,
    #     description="Token usage tracking configuration",
    # )
    models: list[ModelConfig] = Field(
        default_factory=list, description="Available models"
    )
    # # sandbox: SandboxConfig = Field(description="Sandbox configuration")
    # tools: list[ToolConfig] = Field(default_factory=list, description="Available tools")
    # tool_groups: list[ToolGroupConfig] = Field(
    #     default_factory=list, description="Available tool groups"
    # )
    # skills: SkillsConfig = Field(
    #     default_factory=SkillsConfig, description="Skills configuration"
    # )
    # skill_evolution: SkillEvolutionConfig = Field(
    #     default_factory=SkillEvolutionConfig,
    #     description="Agent-managed skill evolution configuration",
    # )
    # extensions: ExtensionsConfig = Field(
    #     default_factory=ExtensionsConfig,
    #     description="Extensions configuration (MCP servers and skills state)",
    # )
    # tool_search: ToolSearchConfig = Field(
    #     default_factory=ToolSearchConfig,
    #     description="Tool search / deferred loading configuration",
    # )
    # title: TitleConfig = Field(
    #     default_factory=TitleConfig,
    #     description="Automatic title generation configuration",
    # )
    # summarization: SummarizationConfig = Field(
    #     default_factory=SummarizationConfig,
    #     description="Conversation summarization configuration",
    # )
    # memory: MemoryConfig = Field(
    #     default_factory=MemoryConfig, description="Memory subsystem configuration"
    # )
    # agents_api: AgentsApiConfig = Field(
    #     default_factory=AgentsApiConfig,
    #     description="Custom-agent management API configuration",
    # )
    # acp_agents: dict[str, ACPAgentConfig] = Field(
    #     default_factory=dict, description="ACP-compatible agent configuration"
    # )
    # subagents: SubagentsAppConfig = Field(
    #     default_factory=SubagentsAppConfig, description="Subagent runtime configuration"
    # )
    # guardrails: GuardrailsConfig = Field(
    #     default_factory=GuardrailsConfig,
    #     description="Guardrail middleware configuration",
    # )
    # circuit_breaker: CircuitBreakerConfig = Field(
    #     default_factory=CircuitBreakerConfig,
    #     description="LLM circuit breaker configuration",
    # )
    # loop_detection: LoopDetectionConfig = Field(
    #     default_factory=LoopDetectionConfig,
    #     description="Loop detection middleware configuration",
    # )
    # safety_finish_reason: SafetyFinishReasonConfig = Field(
    #     default_factory=SafetyFinishReasonConfig,
    #     description="Provider safety-filter finish_reason interception middleware configuration",
    # )
    # model_config = ConfigDict(extra="allow")
    # database: DatabaseConfig = Field(
    #     default_factory=DatabaseConfig,
    #     description="Unified database backend configuration",
    # )
    # run_events: RunEventsConfig = Field(
    #     default_factory=RunEventsConfig, description="Run event storage configuration"
    # )
    # checkpointer: CheckpointerConfig | None = Field(
    #     default=None, description="Checkpointer configuration"
    # )
    # stream_bridge: StreamBridgeConfig | None = Field(
    #     default=None, description="Stream bridge configuration"
    # )


def get_app_config() -> AppConfig:
    """Get the DeerFlow config instance.

    Returns a cached singleton instance and automatically reloads it when the
    underlying config file path or modification time changes. Use
    `reload_app_config()` to force a reload, or `reset_app_config()` to clear
    the cache.
    """
    return AppConfig()
