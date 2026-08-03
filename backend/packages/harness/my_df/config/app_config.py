"""应用级配置模型。

通过环境变量加载运行时配置，所有敏感信息（API Key）来自 .env 文件。
"""

import logging
import os
from typing import cast

from pydantic import BaseModel, Field

from my_df.config.checkpointer_config import CheckpointerConfig, CheckpointerType
from my_df.config.embedding_config import EmbeddingConfig
from my_df.config.milvus_config import MilvusConfig, load_milvus_config_from_env
from my_df.config.model_config import ModelConfig
from my_df.config.stream_bridge_config import StreamBridgeConfig

logger = logging.getLogger(__name__)

# 环境变量常量
ENV_LOG_LEVEL = "MYDF_LOG_LEVEL"
ENV_DEBUG = "MYDF_DEBUG"
ENV_PLAN_MODE = "MYDF_IS_PLAN_MODE"
ENV_LLM_MODEL = "MYDF_LLM_MODEL"
ENV_LLM_API_KEY = "MYDF_LLM_API_KEY"
ENV_CHECKPOINTER_TYPE = "MYDF_CHECKPOINTER_TYPE"
ENV_CHECKPOINTER_PATH = "MYDF_CHECKPOINTER_PATH"


class AppConfig(BaseModel):
    """my-df 应用配置。"""

    log_level: str = Field(
        default="info",
        description="日志级别（debug/info/warning/error）；不影响第三方库的日志",
    )
    models: list[ModelConfig] = Field(
        default_factory=list, description="可用的模型配置列表"
    )
    checkpointer: CheckpointerConfig | None = Field(
        default=None, description="Checkpointer configuration"
    )
    stream_bridge: StreamBridgeConfig | None = Field(
        default=None, description="Stream bridge configuration"
    )
    is_plan_mode: bool = Field(
        default=False, description="是否启用计划模式（TodoMiddleware）"
    )
    milvus: MilvusConfig | None = Field(
        default=None,
        description="Milvus 配置",
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding 模型配置",
    )


def _build_default_model_config() -> ModelConfig | None:
    """从环境变量构建默认模型配置。

    如果配置了 ``MYDF_LLM_API_KEY``（或 ``DEEPSEEK_API_KEY`` 回退），
    则返回一个 ModelConfig 实例；否则返回 None 表示未配置模型。
    """
    api_key = os.getenv(ENV_LLM_API_KEY) or os.getenv("DEEPSEEK_API_KEY")
    model_name = os.getenv(ENV_LLM_MODEL, "deepseek-v4-flash")

    if not api_key:
        logger.warning(
            "未检测到 LLM API Key（%s 或 DEEPSEEK_API_KEY）。"
            "如需使用 LLM，请在 .env 中配置。当前以调试模式运行。",
            ENV_LLM_API_KEY,
        )
        return None

    # langchain_deepseek.ChatDeepSeek 会自动读取 DEEPSEEK_API_KEY 环境变量
    return ModelConfig(
        name=model_name,
        model=model_name,
        use="langchain_deepseek.ChatDeepSeek",
    )


def _build_checkpointer_config() -> CheckpointerConfig | None:
    """从环境变量构建 checkpointer 配置。"""
    raw = os.getenv(ENV_CHECKPOINTER_TYPE)
    if raw not in ("memory", "sqlite", "postgres"):
        return None  # 未配置或值无效，使用 InMemorySaver
    return CheckpointerConfig(
        type=cast("CheckpointerType", raw),
        connection_string=os.getenv(ENV_CHECKPOINTER_PATH)
        or ".deer-flow/checkpoints.db",
    )


_app_config: AppConfig | None = None


def get_app_config() -> AppConfig:
    """获取应用配置实例。

    从环境变量中读取配置并缓存单例。当 ``get_app_config()`` 首次调用时，
    ``.env`` 应已被 ``app.py`` 的 lifespan 加载。
    """
    global _app_config
    if _app_config is not None:
        return _app_config

    log_level = os.getenv(ENV_LOG_LEVEL, "info").lower()
    is_debug = os.getenv(ENV_DEBUG, "0") == "1"
    is_plan_mode = os.getenv(ENV_PLAN_MODE, "0") == "1"

    models: list[ModelConfig] = []
    default_model = _build_default_model_config()
    if default_model is not None:
        models.append(default_model)

    checkpointer = _build_checkpointer_config()

    from my_df.config.embedding_config import load_embedding_config_from_env

    _app_config = AppConfig(
        log_level=log_level if not is_debug else "debug",
        models=models,
        checkpointer=checkpointer,
        is_plan_mode=is_plan_mode,
        embedding=load_embedding_config_from_env(),
        milvus=load_milvus_config_from_env(),
    )

    logger.info(
        "应用配置已加载: models=%d, plan_mode=%s, log_level=%s, checkpointer=%s",
        len(_app_config.models),
        _app_config.is_plan_mode,
        _app_config.log_level,
        _app_config.checkpointer.type if _app_config.checkpointer else "memory",  # type: ignore[reportOptionalMemberAccess]
    )
    return _app_config
