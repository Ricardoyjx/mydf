from pydantic import BaseModel, ConfigDict, Field


class SandboxConfig(BaseModel):
    """沙箱配置：仅保留 Provider 类路径（注册与发现入口）。"""

    use: str = Field(
        default="my_df.sandbox.local_sandbox_provider.LocalSandboxProvider",
        description="Class path of the sandbox provider (e.g. deerflow.sandbox.local:LocalSandboxProvider)",
    )

    model_config = ConfigDict(extra="allow")


def load_sandbox_config_from_env() -> SandboxConfig:
    import os

    # 默认值必须与注册表/实际模块路径一致（解析器按类路径或别名查找）
    return SandboxConfig(
        use=os.getenv(
            "MYDF_SANDBOX_PROVIDER",
            "my_df.sandbox.local_sandbox_provider.LocalSandboxProvider",
        )
    )
