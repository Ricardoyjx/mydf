"""请求体 Pydantic 模型：定义 ``/api/runs/stream`` 接受的参数。"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    """创建一次运行的请求体，所有字段均为可选（含默认值）。"""

    assistant_id: str | None = Field(
        default=None, description="使用的 Agent / assistant 名称"
    )
    input: dict[str, Any] | None = Field(
        default=None, description='LangGraph 图输入（如 {"messages": [...]}）'
    )
    command: dict[str, Any] | None = Field(
        default=None, description="LangGraph Command（直接指令模式）"
    )
    metadata: dict[str, Any] | None = Field(default=None, description="运行元数据")
    config: dict[str, Any] | None = Field(
        default=None, description="RunnableConfig 覆盖"
    )
    context: dict[str, Any] | None = Field(
        default=None, description="运行时上下文覆盖（model_name、thinking_enabled 等）"
    )
    webhook: str | None = Field(default=None, description="运行完成后的回调 URL")
    checkpoint_id: str | None = Field(default=None, description="从指定检查点恢复运行")
    checkpoint: dict[str, Any] | None = Field(
        default=None, description="完整的检查点对象"
    )
    interrupt_before: list[str] | Literal["*"] | None = Field(
        default=None, description="在此节点列表之前中断"
    )
    interrupt_after: list[str] | Literal["*"] | None = Field(
        default=None, description="在此节点列表之后中断"
    )
    stream_mode: list[str] | str | None = Field(
        default=None, description='流模式（如 "values"、"updates"）'
    )
    stream_subgraphs: bool = Field(default=False, description="是否包含子图事件")
    stream_resumable: bool | None = Field(
        default=None, description="SSE 是否支持断线重连"
    )
    on_disconnect: Literal["cancel", "continue"] = Field(
        default="continue",
        description="SSE 断开时的行为：取消运行或继续执行（默认继续，前端无取消按钮）",
    )
    on_completion: Literal["delete", "keep"] = Field(
        default="keep", description="完成后是否删除临时线程"
    )
    multitask_strategy: Literal["reject", "rollback", "interrupt", "enqueue"] = Field(
        default="reject", description="并发运行策略"
    )
    after_seconds: float | None = Field(default=None, description="延迟执行的秒数")
    if_not_exists: Literal["reject", "create"] = Field(
        default="create", description="线程不存在时的行为"
    )
    feedback_keys: list[str] | None = Field(
        default=None, description="LangSmith 反馈键列表"
    )
