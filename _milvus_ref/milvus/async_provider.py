"""Async Milvus storage factory — 生命周期上下文管理器。

对齐 :func:`my_df.runtime.checkpointer.async_provider.make_checkpointer`
的 async context manager 模式，用于 FastAPI lifespan。

用法（在 app.py 的 lifespan 中）::

    from my_df.runtime.milvus.async_provider import make_milvus_storage

    async with make_milvus_storage(app_config) as milvus:
        await milvus.ensure_collection("default")
        app.state.milvus = milvus
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_df.runtime.milvus.base import MilvusStorage

from my_df.config.app_config import AppConfig
from my_df.config.milvus_config import MilvusConfig, get_milvus_config
from my_df.runtime.milvus.client import PymilvusStorage

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def make_milvus_storage(
    app_config: AppConfig | None = None,
) -> AsyncIterator[MilvusStorage]:
    """Async context manager: 创建并管理 MilvusStorage 生命周期。

    1. 从配置构建 PymilvusStorage 实例
    2. 调用 connect() 建立连接
    3. yield 给调用方（FastAPI lifespan）
    4. 退出时调用 close() 断开连接

    参数：
        app_config: 应用配置。为 None 时从全局单例获取。

    Yields:
        MilvusStorage 实例（已连接状态）。
    """
    if app_config is not None:
        config = app_config.milvus if hasattr(app_config, 'milvus') and app_config.milvus else None
    else:
        config = get_milvus_config()

    if config is None:
        config = MilvusConfig()

    storage = PymilvusStorage(config=config)

    try:
        await storage.connect()
        logger.info("MilvusStorage 已就绪: host=%s:%s", config.host, config.port)
        yield storage
    finally:
        await storage.close()
        logger.info("MilvusStorage 已关闭")
