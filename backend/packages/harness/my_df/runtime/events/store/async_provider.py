"""事件存储工厂：根据 checkpointer 配置选择 PG 或内存实现。"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from my_df.config.app_config import AppConfig, get_app_config
from my_df.runtime.events.store.base import RunEventStore
from my_df.runtime.events.store.memory import MemoryRunEventStore
from my_df.runtime.events.store.postgres import PostgresRunEventStore

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def make_event_store(
    app_config: AppConfig | None = None,
) -> AsyncIterator[RunEventStore]:
    """根据 checkpointer 配置构建事件存储（PG 持久化，其余回退内存）。"""
    if app_config is None:
        app_config = get_app_config()

    cp = app_config.checkpointer
    if cp is not None and cp.type == "postgres" and cp.connection_string:
        async with PostgresRunEventStore(cp.connection_string) as store:
            yield store
        return

    logger.warning(
        "事件存储使用内存实现（数据重启即失）；配置 postgres checkpointer 可持久化"
    )
    yield MemoryRunEventStore()
