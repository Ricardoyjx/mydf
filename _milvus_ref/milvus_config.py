"""Milvus 向量数据库配置模型。

与环境变量或 .env 配合使用（遵循项目规范，敏感信息不硬编码）。

环境变量：
    MYDF_MILVUS_HOST      Milvus 服务地址（默认 localhost）
    MYDF_MILVUS_PORT      Milvus gRPC 端口（默认 19530）
    MYDF_MILVUS_ALIAS     连接别名（默认 "default"）
    MYDF_MILVUS_DIM       向量维度（默认 384，对应 all-MiniLM-L6-v2）
    MYDF_MILVUS_INDEX_TYPE 索引类型（默认 IVF_FLAT，可选 HNSW）
"""

from typing import Literal

from pydantic import BaseModel, Field

MilvusIndexType = Literal["IVF_FLAT", "IVF_SQ8", "HNSW"]


class MilvusConfig(BaseModel):
    """Milvus 连接与索引配置。"""

    host: str = Field(
        default="localhost",
        description="Milvus 服务地址。生产环境应配置为内网 IP。",
    )
    port: int = Field(
        default=19530,
        description="Milvus gRPC 端口。HTTP 端口为 9091（用于 Attu 管理界面）。",
    )
    alias: str = Field(
        default="default",
        description="连接别名，用于区分多个 Milvus 连接。",
    )
    vector_dim: int = Field(
        default=384,
        description="向量维度。需与 Embedding 模型输出维度一致。"
        "sentence-transformers/all-MiniLM-L6-v2 = 384 维；"
        "OpenAI text-embedding-ada-002 = 1536 维。",
    )
    index_type: MilvusIndexType = Field(
        default="IVF_FLAT",
        description="索引类型。IVF_FLAT 精度最高（暴力搜索），"
        "HNSW 性能更好但构建更慢，适合大规模数据。",
    )
    collection_name_prefix: str = Field(
        default="my_df_memory",
        description="集合名称前缀。完整集合名为 {prefix}_{user_id}，实现多用户隔离。",
    )
    nlist: int = Field(
        default=128,
        description="IVF 索引的聚类中心数。越大精度越高但构建越慢。",
    )


# ── 全局单例 ──

_milvus_config: MilvusConfig = MilvusConfig()


def get_milvus_config() -> MilvusConfig:
    """获取当前的 Milvus 配置实例。"""
    return _milvus_config


def set_milvus_config(config: MilvusConfig) -> None:
    """设置 Milvus 配置（在 lifespan 初始化时调用）。"""
    global _milvus_config
    _milvus_config = config


def load_milvus_config_from_env() -> MilvusConfig:
    """从环境变量加载 Milvus 配置。

    在 ``app.py`` 的 lifespan 中调用，此时 .env 已加载。
    """
    import os

    return MilvusConfig(
        host=os.getenv("MYDF_MILVUS_HOST", "localhost"),
        port=int(os.getenv("MYDF_MILVUS_PORT", "19530")),
        alias=os.getenv("MYDF_MILVUS_ALIAS", "default"),
        vector_dim=int(os.getenv("MYDF_MILVUS_DIM", "384")),
        index_type=os.getenv("MYDF_MILVUS_INDEX_TYPE", "IVF_FLAT"),  # type: ignore
        collection_name_prefix=os.getenv(
            "MYDF_MILVUS_COLLECTION_PREFIX", "my_df_memory"
        ),
        nlist=int(os.getenv("MYDF_MILVUS_NLIST", "128")),
    )
