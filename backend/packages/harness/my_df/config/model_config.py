"""单个模型的配置定义。"""

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """模型配置段。"""

    name: str = Field(..., description="模型唯一标识名称")
    display_name: str | None = Field(
        default=None, description="模型显示名称"
    )
    description: str | None = Field(
        default=None, description="模型描述"
    )
    use: str = Field(
        ...,
        description="模型提供者的类路径（如 langchain_openai.ChatOpenAI）",
    )
    model: str = Field(..., description="模型名称（传递给提供者的参数）")
    model_config = ConfigDict(extra="allow")

    use_responses_api: bool | None = Field(
        default=None,
        description="是否将 OpenAI ChatOpenAI 调用路由到 /v1/responses API",
    )
    output_version: str | None = Field(
        default=None,
        description="OpenAI responses 内容的结构化输出版本，如 responses/v1",
    )
    supports_thinking: bool = Field(
        default_factory=lambda: False, description="是否支持思考模式"
    )
    supports_reasoning_effort: bool = Field(
        default_factory=lambda: False, description="是否支持推理力度控制"
    )
    when_thinking_enabled: dict | None = Field(
        default_factory=lambda: None,
        description="启用思考时传递给模型的额外设置",
    )
    when_thinking_disabled: dict | None = Field(
        default_factory=lambda: None,
        description="禁用思考时传递给模型的额外设置",
    )
    supports_vision: bool = Field(
        default_factory=lambda: False, description="是否支持视觉/图像输入"
    )
    thinking: dict | None = Field(
        default_factory=lambda: None,
        description=(
            "模型的思考设置。若提供，启用思考时传递给模型。"
            "这是 ``when_thinking_enabled`` 的快捷方式，两者同时提供时会合并。"
        ),
    )
