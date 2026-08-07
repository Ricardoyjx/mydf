from pydantic import BaseModel, Field


class RerankConfig(BaseModel):
    model: str = Field(
        default="BAAI/bge-reranker-base",
        description="SentenceTransformer 模型名称。支持 HuggingFace 上的任意 SentenceTransformer 模型。",
    )

    enable: bool = Field(
        default=False,
        description="是否启用 Rerank 模型（需显式配置 MYDF_RERANK_ENABLED=true）。",
    )

    candidate_k: int = Field(default=20, description="Rerank 模型使用的候选数量。")

    top_n: int = Field(default=5, description="Rerank 模型返回精排保留数。")


def load_rerank_config_from_env() -> RerankConfig:
    import os

    return RerankConfig(
        model=os.getenv("MYDF_RERANK_MODEL", "BAAI/bge-reranker-base"),
        enable=os.getenv("MYDF_RERANK_ENABLED", "false").lower() == "true",
    )
