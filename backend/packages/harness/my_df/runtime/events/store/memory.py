"""内存版 RunEventStore：开发/测试环境使用，进程退出即丢失。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from my_df.runtime.events.store.base import RunEventStore

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """当前 UTC 时间 ISO 格式。"""
    return datetime.now(timezone.utc).isoformat()


class MemoryRunEventStore(RunEventStore):
    """基于内存 dict 的事件存储（按 run_id 分组，seq 单调递增）。"""

    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = {}
        self._seq = 0

    async def put(
        self,
        *,
        thread_id: str,
        run_id: str,
        event_type: str,
        category: str,
        content: str | dict = "",
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> dict:
        self._seq += 1
        record = {
            "seq": self._seq,
            "thread_id": thread_id,
            "run_id": run_id,
            "event_type": event_type,
            "category": category,
            "content": content,
            "metadata": metadata or {},
            "created_at": created_at or _utc_now(),
        }
        self._events.setdefault(run_id, []).append(record)
        return record

    async def put_batch(self, events: list[dict]) -> list[dict]:
        """批量写入（逐条 put，保证 seq 递增）。"""
        return [await self.put(**e) for e in events]

    async def list_messages(
        self,
        thread_id: str,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> list[dict]:
        """返回线程内 category="message" 的事件（按 seq 升序）。"""
        all_events = [
            e
            for events in self._events.values()
            for e in events
            if e["thread_id"] == thread_id and e["category"] == "message"
        ]
        return self._paginate(all_events, limit, before_seq, after_seq)

    async def list_messages_by_run(
        self,
        thread_id: str,
        run_id: str,
        *,
        limit: int = 50,
        before_seq: int | None = None,
        after_seq: int | None = None,
    ) -> list[dict]:
        """返回指定 run 内的 message 事件（按 seq 升序，支持游标分页）。"""
        events = [
            e
            for e in self._events.get(run_id, [])
            if e["category"] == "message" and e["thread_id"] == thread_id
        ]
        return self._paginate(events, limit, before_seq, after_seq)

    async def count_messages(self, thread_id: str) -> int:
        """统计线程内 message 事件数。"""
        return sum(
            1
            for events in self._events.values()
            for e in events
            if e["thread_id"] == thread_id and e["category"] == "message"
        )

    async def delete_by_thread(self, thread_id: str) -> int:
        """删除线程全部事件，返回删除数量。"""
        deleted = 0
        for run_id, events in list(self._events.items()):
            kept = [e for e in events if e["thread_id"] != thread_id]
            deleted += len(events) - len(kept)
            if kept:
                self._events[run_id] = kept
            else:
                self._events.pop(run_id, None)
        return deleted

    async def delete_by_run(self, thread_id: str, run_id: str) -> int:
        """删除指定 run 的事件，返回删除数量。"""
        events = self._events.get(run_id, [])
        kept = [e for e in events if e["thread_id"] != thread_id]
        deleted = len(events) - len(kept)
        if kept:
            self._events[run_id] = kept
        else:
            self._events.pop(run_id, None)
        return deleted

    async def list_events(
        self,
        thread_id: str,
        run_id: str,
        *,
        event_types: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """返回指定 run 的全部事件（按 seq 升序，可选类型过滤）。"""
        events = self._events.get(run_id, [])
        if event_types:
            events = [e for e in events if e["event_type"] in event_types]
        return events[-limit:]

    @staticmethod
    def _paginate(
        events: list[dict],
        limit: int,
        before_seq: int | None,
        after_seq: int | None,
    ) -> list[dict]:
        """游标分页：按 seq 过滤并取最近 limit 条（升序返回）。"""
        if before_seq is not None:
            events = [e for e in events if e["seq"] < before_seq]
        if after_seq is not None:
            events = [e for e in events if e["seq"] > after_seq]
        return events[-limit:]
