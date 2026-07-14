"""FastAPI 依赖注入工具，从 app.state 中获取共享实例。"""

from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
import logging
from my_df.agents.config.app_config import AppConfig, get_app_config
from my_df.agents.config.stream_bridge_config import get_stream_bridge_config
from my_df.runtime.checkpointer.async_provider import make_checkpointer
from my_df.runtime.events.store.base import RunEventStore
from my_df.runtime.runs.manager import RunManager
from fastapi import FastAPI, HTTPException, Request
from typing import AsyncGenerator, AsyncIterator, TypeVar, cast
from langgraph.checkpoint.base import BaseCheckpointSaver
from my_df.runtime.runs.store.base import RunStore
from my_df.runtime.runs.store.memory import MemoryRunStore
from my_df.runtime.runs.worker import RunContext
from my_df.runtime.store.async_provider import make_store
from my_df.runtime.stream_bridge.base import StreamBridge

T = TypeVar("T")


logger = logging.getLogger(__name__)


@asynccontextmanager
async def langgraph_runtime(
    app: FastAPI, startup_config: AppConfig
) -> AsyncGenerator[None, None]:
    """引导并拆除所有 Lang Graph 运行时单例

    “启动配置”是“应用程序配置”期间拍摄的快照
    一次性基础设施引导程序的“lifespan ()” 引擎和
    此处构建的存储（流桥、持久化引擎、检查点、
    store 、 run event store ）是设计所要求的重新启动 - 它们保持活动状态
    连接、文件句柄或单例提供程序 - 因此它们绑定到此
    快照并在“config yaml”编辑中生存请求时间消费者
    仍然必须通过 :func :`get config ` 对于任何应该是的字段
    可热重载 请参阅``backend /CLAUDE md ``“配置热重载边界”

    匹配的“运行事件配置”被冻结到“应用程序状态”，因此
    :func :`get run context ` 将新加载的 ``App Config `` 与
    *启动时间*运行事件配置底层``事件存储``
    是从构建的 - 否则运行时可能最终会结合实时
    新的“运行事件配置”，事件存储仍然绑定到
    以前的后端

    在``app py``中的用法::

        与 langgraph 运行时异步（应用程序，启动配置）：
            产量 = yield
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


@asynccontextmanager
async def make_stream_bridge(
    app_config: AppConfig | None = None,
) -> AsyncIterator[StreamBridge]:
    """Async context manager that yields a :class:`StreamBridge`.

    Falls back to :class:`MemoryStreamBridge` when no configuration is
    provided and nothing is set globally.
    """
    if app_config is None:
        config = get_stream_bridge_config()
    else:
        config = app_config.stream_bridge

    if config is None or config.type == "memory":
        from my_df.runtime.stream_bridge.memory import InMemoryStreamBridge

        maxsize = config.queue_maxsize if config is not None else 256
        bridge = InMemoryStreamBridge(queue_maxsize=maxsize)
        logger.info("Stream bridge initialised: memory (queue_maxsize=%d)", maxsize)
        try:
            yield bridge
        finally:
            await bridge.close()
        return

    if config.type == "redis":
        raise NotImplementedError("Redis stream bridge planned for Phase 2")

    raise ValueError(f"Unknown stream bridge type: {config.type!r}")


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
get_checkpointer: Callable[[Request], Checkpointer] = _require(
    "checkpointer", "Checkpointer"
)
get_run_event_store: Callable[[Request], RunEventStore] = _require(
    "run_event_store", "Run event store"
)
# get_feedback_repo: Callable[[Request], FeedbackRepository] = _require(
#     "feedback_repo", "Feedback"
# )
get_run_store: Callable[[Request], RunStore] = _require("run_store", "Run store")


def get_store(request: Request):
    """Return the global store (may be ``None`` if not configured)."""
    return getattr(request.app.state, "store", None)


def get_run_context(request: Request) -> RunContext:
    """从 ``app state `` 单例构建一个 :class :`Run Context `

     返回具有基础设施依赖项的 *base * 上下文
     ``app config `` 字段是实时解析的，因此每个运行字段（例如
    ``models [*] max tokens ``) 遵循 ``config yaml `` 编辑；的
     “事件存储”/“运行事件配置”对保持冻结到快照
     在 :func :`langgraph Runtime` 中捕获，因此调用者永远不会看到存储绑定
     一个后端与指向另一个后端的配置配对
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
