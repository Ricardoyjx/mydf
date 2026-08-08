"""FastAPI 依赖注入工具，从 app.state 中获取共享实例。"""

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TypeVar, cast

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.base import BaseCheckpointSaver
from my_df.config.app_config import AppConfig, get_app_config
from my_df.runtime.checkpointer.async_provider import make_checkpointer
from my_df.runtime.runs.manager import RunManager
from my_df.runtime.runs.worker import RunContext
from my_df.runtime.store.async_provider import make_store
from my_df.runtime.store.base import RunStore
from my_df.runtime.store.memory import MemoryRunStore
from my_df.runtime.stream_bridge.async_provider import make_stream_bridge
from my_df.runtime.stream_bridge.base import StreamBridge

T = TypeVar("T")


logger = logging.getLogger(__name__)


@asynccontextmanager
async def langgraph_runtime(
    app: FastAPI, startup_config: AppConfig
) -> AsyncGenerator[None, None]:
    """引导并拆除所有 LangGraph 运行时单例。

    lifespan 启动时在此处构建流桥、checkpointer、store 等基础设施，
    它们持有连接、文件句柄等资源，必须与启动时的配置快照绑定，
    因此存活于整个应用生命周期，不随配置热重载变化。

    ``get_run_context()`` 会将新加载的 ``AppConfig`` 与启动时冻结的
    运行事件配置配对使用，避免“新配置 + 旧存储后端”的错配。

    用法（app.py 的 lifespan 中）::

        async with langgraph_runtime(app, startup_config):
            yield
    """
    async with AsyncExitStack() as stack:
        config = startup_config
        app.state.stream_bridge = await stack.enter_async_context(
            make_stream_bridge(config)
        )

        app.state.checkpointer = await stack.enter_async_context(
            make_checkpointer(config)
        )

        app.state.store = await stack.enter_async_context(make_store(config))

        app.state.run_store = MemoryRunStore()

        app.state.run_manager = RunManager(store=app.state.run_store)

        yield


def _require(attr: str, label: str) -> Callable[[Request], T]:  # type: ignore
    """工厂函数：生成一个 FastAPI 依赖，从 ``app.state.<attr>`` 取值。

    如果该属性未设置，返回 503 Service Unavailable。
    """

    def dep(request: Request) -> T:
        val = getattr(request.app.state, attr, None)
        if val is None:
            raise HTTPException(status_code=503, detail=f"{label} not available")
        return cast(T, val)

    dep.__name__ = dep.__qualname__ = f"get_{attr}"
    return dep


Checkpointer = None | bool | BaseCheckpointSaver
"""用于子图的检查指针的类型 

``True` 启用该子图的持久检查点 
 “False”禁用检查点，即使父图有检查点 
 `None` 从父图继承检查指针
"""

# 预定义的依赖注入器
get_stream_bridge: Callable[[Request], StreamBridge] = _require(
    "stream_bridge", "Stream bridge"
)
get_run_manager: Callable[[Request], RunManager] = _require(
    "run_manager", "Run manager"
)
get_checkpointer: Callable[[Request], BaseCheckpointSaver | None] = _require(
    "checkpointer", "Checkpointer"
)
# get_run_event_store: Callable[[Request], RunEventStore] = _require(
#     "run_event_store", "Run event store"
# )
# get_feedback_repo: Callable[[Request], FeedbackRepository] = _require(
#     "feedback_repo", "Feedback"
# )
get_run_store: Callable[[Request], RunStore] = _require("run_store", "Run store")


def get_store(request: Request):
    """Return the global store (may be ``None`` if not configured)."""
    return getattr(request.app.state, "store", None)


def get_run_context(request: Request) -> RunContext:
    """从 app.state 单例构建一个 :class:`RunContext`。

    基础上下文持有 checkpointer、store 等基础设施依赖；
    ``app_config`` 字段在每次请求时实时解析，因此模型等配置
    修改可以热生效。事件存储与运行事件配置保持启动时的配对，
    避免存储后端与配置指向不一致。
    """

    return RunContext(
        # checkpointer=get_checkpointer(request),
        checkpointer=getattr(request.app.state, "checkpointer", None),
        # store=get_store(request),
        store=getattr(request.app.state, "store", None),
        # event_store=get_run_event_store(request),
        event_store=getattr(request.app.state, "event_store", None),
        run_events_config=getattr(request.app.state, "run_events_config", None),
        # thread_store=get_thread_store(request),
        app_config=get_app_config(),
    )
