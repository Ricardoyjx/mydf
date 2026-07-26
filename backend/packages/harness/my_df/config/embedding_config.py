"""Embedding 模型配置。

环境变量：
    MYDF_EMBEDDING_MODEL    SentenceTransformer 模型名称（默认 all-MiniLM-L6-v2）
"""

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置。"""

    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer 模型名称。支持 HuggingFace 上的任意 SentenceTransformer 模型。",
    )


# ── 全局单例 ──

_embedding_config: EmbeddingConfig = EmbeddingConfig()


def get_embedding_config() -> EmbeddingConfig:
    return _embedding_config


def set_embedding_config(config: EmbeddingConfig) -> None:
    global _embedding_config
    _embedding_config = config


def load_embedding_config_from_env() -> EmbeddingConfig:
    import os

    return EmbeddingConfig(
        model=os.getenv("MYDF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )
