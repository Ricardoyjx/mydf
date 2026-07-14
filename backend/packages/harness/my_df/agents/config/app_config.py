"""应用级配置模型。"""

from my_df.agents.config.checkpointer_config import CheckpointerConfig
from my_df.agents.config.stream_bridge_config import StreamBridgeConfig
from pydantic import BaseModel, Field

from my_df.agents.config.model_config import ModelConfig


class AppConfig(BaseModel):
    """my-df 应用配置。"""

    log_level: str = Field(
        default="info",
        description="日志级别（debug/info/warning/error）；不影响第三方库的日志",
    )
    models: list[ModelConfig] = Field(
        default_factory=list, description="可用的模型配置列表"
    )
    # run_events: RunEventsConfig = Field(
    #     default_factory=RunEventsConfig, description="Run event storage configuration"
    # )
    checkpointer: CheckpointerConfig | None = Field(
        default=None, description="Checkpointer configuration"
    )
    stream_bridge: StreamBridgeConfig | None = Field(
        default=None, description="Stream bridge configuration"
    )


def get_app_config() -> AppConfig:
    """获取应用配置实例。

    返回缓存的单例实例。当底层配置文件路径或修改时间变化时自动重载。
    """
    return AppConfig()
