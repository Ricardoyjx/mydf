"""FastAPI 依赖注入工具，从 app.state 中获取共享实例。"""

from collections.abc import Callable
from my_df.runtime.runs.manager import RunManager
from fastapi import HTTPException, Request
from typing import TypeVar, cast
from my_df.runtime.stream_bridge.base import StreamBridge

T = TypeVar("T")


def _require(attr: str, label: str) -> Callable[[Request], T]:
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


# 预定义的依赖注入器
get_stream_bridge: Callable[[Request], StreamBridge] = _require(
    "stream_bridge", "Stream bridge"
)
get_run_manager: Callable[[Request], RunManager] = _require(
    "run_manager", "Run manager"
)
