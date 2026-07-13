"""应用配置：通过 os.getenv 从环境变量或 .env 加载。

不依赖 pydantic-settings，使用 Python 内置 os.getenv + dataclass。
如需 .env 文件自动加载，安装 python-dotenv 后取消下方 dotenv 注释。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# 尝试自动加载 .env 文件（安装 python-dotenv 后生效）
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    """my-df 网关配置项，从环境变量（前缀 MYDF_）读取。"""

    # 调试模式: 1 启用 mock 依赖
    debug: bool = field(default_factory=lambda: os.getenv("MYDF_DEBUG", "0") == "1")

    # 日志级别
    log_level: str = field(
        default_factory=lambda: os.getenv("MYDF_LOG_LEVEL", "INFO")
    )

    # LLM 模型名称
    llm_model: str = field(
        default_factory=lambda: os.getenv("MYDF_LLM_MODEL", "deepseek-v4-flash")
    )

    # LLM API Key
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("MYDF_LLM_API_KEY", "")
    )

    # 服务端口
    port: int = field(
        default_factory=lambda: int(os.getenv("MYDF_PORT", "8000"))
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局单例配置（惰性初始化）。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
