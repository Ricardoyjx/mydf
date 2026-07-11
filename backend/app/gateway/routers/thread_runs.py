from typing import Any, Literal

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    assistant_id: str | None = Field(
        default=None, description="Agent / assistant to use"
    )
    input: dict[str, Any] | None = Field(
        default=None, description="Graph input (e.g. {messages: [...]})"
    )
    command: dict[str, Any] | None = Field(
        default=None, description="LangGraph Command"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Run metadata")
    config: dict[str, Any] | None = Field(
        default=None, description="RunnableConfig overrides"
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="DeerFlow context overrides (model_name, thinking_enabled, etc.)",
    )
    webhook: str | None = Field(default=None, description="Completion callback URL")
    checkpoint_id: str | None = Field(
        default=None, description="Resume from checkpoint"
    )
    checkpoint: dict[str, Any] | None = Field(
        default=None, description="Full checkpoint object"
    )
    interrupt_before: list[str] | Literal["*"] | None = Field(
        default=None, description="Nodes to interrupt before"
    )
    interrupt_after: list[str] | Literal["*"] | None = Field(
        default=None, description="Nodes to interrupt after"
    )
    stream_mode: list[str] | str | None = Field(
        default=None, description="Stream mode(s)"
    )
    stream_subgraphs: bool = Field(default=False, description="Include subgraph events")
    stream_resumable: bool | None = Field(
        default=None, description="SSE resumable mode"
    )
    on_disconnect: Literal["cancel", "continue"] = Field(
        default="cancel", description="Behaviour on SSE disconnect"
    )
    on_completion: Literal["delete", "keep"] = Field(
        default="keep", description="Delete temp thread on completion"
    )
    multitask_strategy: Literal["reject", "rollback", "interrupt", "enqueue"] = Field(
        default="reject", description="Concurrency strategy"
    )
    after_seconds: float | None = Field(default=None, description="Delayed execution")
    if_not_exists: Literal["reject", "create"] = Field(
        default="create", description="Thread creation policy"
    )
    feedback_keys: list[str] | None = Field(
        default=None, description="LangSmith feedback keys"
    )
