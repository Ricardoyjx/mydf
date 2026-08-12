"""PostgreSQL 版 RunEventStore：基于 psycopg 连接池的持久化事件存储。"""

from __future__ import annotations

import json
import logging
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from my_df.runtime.events.store.base import RunEventStore

logger = logging.getLogger(__name__)


class PostgresRunEventStore(RunEventStore):
    """run_events 表的事件存储实现。

    表结构：
        seq BIGSERIAL（同线程内严格递增）
        thread_id / run_id / event_type / category / content
        metadata JSONB / created_at TIMESTAMPTZ
    """

    def __init__(self, conn_string: str, *, pool_size: int = 5) -> None:
        self._pool = AsyncConnectionPool(
            conninfo=conn_string,
            min_size=1,
            max_size=pool_size,
            open=False,
            kwargs={"row_factory": dict_row},
        )

    async def __aenter__(self) -> "PostgresRunEventStore":
        await self._pool.open()
        await self._ensure_schema()
        logger.info("PostgresRunEventStore 已就绪")
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._pool.close()
        logger.info("PostgresRunEventStore 已关闭")

    async def _ensure_schema(self) -> None:
        """建表（幂等）。"""
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    seq BIGSERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'trace',
                    content TEXT NOT NULL DEFAULT '',
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_run_events_run
                    ON run_events (run_id, seq);
                CREATE INDEX IF NOT EXISTS idx_run_events_thread
                    ON run_events (thread_id, seq);
                """
            )

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
        """写入单条事件并返回完整记录。"""
        content_json = (
            json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else content
        )
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO run_events
                    (thread_id, run_id, event_type, category, content, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, now()))
                RETURNING seq, thread_id, run_id, event_type, category, content,
                          metadata, created_at
                """,
                (
                    thread_id,
                    run_id,
                    event_type,
                    category,
                    content_json,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created_at,
                ),
            )
            row = await cursor.fetchone()
            record = dict(row)
            if isinstance(record.get("metadata"), str):
                record["metadata"] = json.loads(record["metadata"])
            record["created_at"] = record["created_at"].isoformat()
            return record

    async def put_batch(self, events: list[dict]) -> list[dict]:
        """批量写入（逐条 put，保证 seq 递增与错误隔离）。"""
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
        clauses = ["thread_id = %s", "category = 'message'"]
        params: list[Any] = [thread_id]
        if before_seq is not None:
            clauses.append("seq < %s")
            params.append(before_seq)
        if after_seq is not None:
            clauses.append("seq > %s")
            params.append(after_seq)
        params.append(limit)
        sql = (
            "SELECT * FROM run_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY seq DESC LIMIT %s"
        )
        async with self._pool.connection() as conn:
            rows = await conn.execute(sql, tuple(params))
            records = [self._normalize(r) for r in await rows.fetchall()]
        return list(reversed(records))

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
        clauses = ["thread_id = %s", "run_id = %s", "category = 'message'"]
        params: list[Any] = [thread_id, run_id]
        if before_seq is not None:
            clauses.append("seq < %s")
            params.append(before_seq)
        if after_seq is not None:
            clauses.append("seq > %s")
            params.append(after_seq)
        params.append(limit)
        sql = (
            "SELECT * FROM run_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY seq DESC LIMIT %s"
        )
        async with self._pool.connection() as conn:
            rows = await conn.execute(sql, tuple(params))
            records = [self._normalize(r) for r in await rows.fetchall()]
        return list(reversed(records))

    async def count_messages(self, thread_id: str) -> int:
        """统计线程内 message 事件数。"""
        async with self._pool.connection() as conn:
            row = await conn.execute(
                "SELECT count(*) AS n FROM run_events "
                "WHERE thread_id = %s AND category = 'message'",
                (thread_id,),
            )
            result = await row.fetchone()
        return int(result["n"])

    async def delete_by_thread(self, thread_id: str) -> int:
        """删除线程全部事件，返回删除数量。"""
        async with self._pool.connection() as conn:
            row = await conn.execute(
                "DELETE FROM run_events WHERE thread_id = %s RETURNING seq",
                (thread_id,),
            )
            rows = await row.fetchall()
        return len(rows)

    async def delete_by_run(self, thread_id: str, run_id: str) -> int:
        """删除指定 run 的事件，返回删除数量。"""
        async with self._pool.connection() as conn:
            row = await conn.execute(
                "DELETE FROM run_events WHERE thread_id = %s AND run_id = %s "
                "RETURNING seq",
                (thread_id, run_id),
            )
            rows = await row.fetchall()
        return len(rows)

    async def list_events(
        self,
        thread_id: str,
        run_id: str,
        *,
        event_types: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """返回指定 run 的全部事件（按 seq 升序，可选类型过滤）。"""
        clauses = ["run_id = %s"]
        params: list[Any] = [run_id]
        if event_types:
            clauses.append("event_type = ANY(%s)")
            params.append(event_types)
        params.append(limit)
        sql = (
            "SELECT * FROM run_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY seq ASC LIMIT %s"
        )
        async with self._pool.connection() as conn:
            rows = await conn.execute(sql, tuple(params))
            return [self._normalize(r) for r in await rows.fetchall()]

    @staticmethod
    def _normalize(row: dict) -> dict:
        """规范化返回记录（JSONB / 时间戳转可序列化类型）。"""
        record = dict(row)
        if isinstance(record.get("metadata"), str):
            record["metadata"] = json.loads(record["metadata"])
        created = record.get("created_at")
        if created is not None:
            record["created_at"] = created.isoformat()
        return record
