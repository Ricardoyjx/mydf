"""流桥接器抽象协议。

StreamBridge 将 Agent 工作线程（生产者）与 SSE 端点（消费者）解耦，
与 LangGraph Platform 的 Queue + StreamManager 架构一致。
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    """单条流事件。

    属性：
        id:    单调递增的事件 ID（用作 SSE ``id:`` 字段，支持 ``Last-Event-ID`` 重连）。
        event: SSE 事件名称，如 ``"metadata"``、``"updates"``、``"events"``、``"error"``、``"end"``。
        data:  可 JSON 序列化的负载。
    """

    id: str
    event: str
    data: Any


# 预定义的哨兵事件
HEARTBEAT_SENTINEL = StreamEvent(id="", event="__heartbeat__", data=None)
END_SENTINEL = StreamEvent(id="", event="__end__", data=None)


class StreamBridge(abc.ABC):
    """流桥接器抽象基类。"""

    @abc.abstractmethod
    async def publish(self, run_id: str, event: str, data: Any) -> None:
        """生产者：为指定 run_id 入队一条事件。"""

    @abc.abstractmethod
    async def publish_end(self, run_id: str) -> None:
        """生产者：通知消费者不会再有新事件。"""

    @abc.abstractmethod
    def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamEvent]:
        """消费者：返回指定 run_id 的异步迭代器。

        当 *heartbeat_interval* 秒内无事件时 yield ``HEARTBEAT_SENTINEL``。
        当生产者调用 ``publish_end`` 后 yield ``END_SENTINEL``。
        """

    @abc.abstractmethod
    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        """释放与 run_id 相关的资源。

        若 *delay* > 0，实现应等待一段时间再释放，
        以便迟到的消费者有机会排空剩余事件。
        """

    async def close(self) -> None:
        """释放后端资源。默认无操作。"""
