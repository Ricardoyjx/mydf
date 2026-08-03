"""基于 ``asyncio.Queue`` 的内存版 ``StreamBridge`` 实现。

生产者调用 ``publish`` / ``publish_end`` 写入事件；
SSE 消费者通过 ``subscribe`` 异步迭代读取。
适用于单进程开发模式和测试场景。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from my_df.runtime.stream_bridge.base import (
    END_SENTINEL,  # 流结束哨兵
    HEARTBEAT_SENTINEL,  # 心跳哨兵（超时时发出）
    StreamBridge,
    StreamEvent,
)


class InMemoryStreamBridge(StreamBridge):
    """基于内存队列的 StreamBridge 实现。

    每个 ``run_id`` 对应一个独立的 ``asyncio.Queue``，
    生产者与消费者通过队列解耦。
    """

    def __init__(self, queue_maxsize: int = 256) -> None:
        # run_id -> asyncio.Queue[StreamEvent]
        self._queues: dict[str, asyncio.Queue[StreamEvent]] = {}
        self._maxsize = queue_maxsize

    async def publish(self, run_id: str, event: str, data: Any) -> None:
        """生产者：向指定 run_id 的队列写入一条事件。"""
        q = self._queues.get(run_id)
        if q is None:
            # 懒创建队列
            q = asyncio.Queue(maxsize=self._maxsize)
            self._queues[run_id] = q
        await q.put(StreamEvent(id=str(uuid.uuid4()), event=event, data=data))

    async def publish_end(self, run_id: str) -> None:
        """生产者：向队列写入结束哨兵，通知消费者流已终止。"""
        q = self._queues.get(run_id)
        if q is not None:
            await q.put(END_SENTINEL)

    async def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamEvent]:
        """消费者：异步迭代指定 run_id 的事件流。

        参数：
            run_id:           目标运行 ID。
            last_event_id:    断线重连时使用（当前实现忽略历史，
                              仅保证新队列可读）。
            heartbeat_interval: 无事件时的最大等待秒数，超时则发出心跳。

        产出：
            StreamEvent — 正常事件；
            HEARTBEAT_SENTINEL — 心跳；
            END_SENTINEL — 流终止信号（产出后迭代结束）。
        """
        q = self._queues.get(run_id)
        if q is None:
            q = asyncio.Queue(maxsize=self._maxsize)
            self._queues[run_id] = q
        while True:
            try:
                # 阻塞等待事件，超时则发送心跳
                entry = await asyncio.wait_for(q.get(), timeout=heartbeat_interval)
            except TimeoutError:
                yield HEARTBEAT_SENTINEL
                continue
            yield entry
            if entry is END_SENTINEL:
                return

    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        """清理指定 run_id 的队列资源。"""
        self._queues.pop(run_id, None)

    async def close(self) -> None:
        """释放所有队列资源。

        清空所有未消费的事件队列，防止内存泄漏。
        由 make_stream_bridge 的 finally 块调用。
        """
        self._queues.clear()
