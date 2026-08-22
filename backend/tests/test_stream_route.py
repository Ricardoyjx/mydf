"""Tests for ``/api/runs/stream`` SSE endpoint.

使用 ``asyncio.run()`` 内嵌异步逻辑，避免 ``pytest-asyncio`` 版本兼容性问题。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from app.gateway.app import create_app
from my_df.runtime.runs.schema import DisconnectMode, RunStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeRunRecord:
    """RunRecord 的最小替代品，用于 SSE 消费者测试。"""

    run_id: str
    thread_id: str
    assistant_id: str | None
    status: RunStatus
    on_disconnect: DisconnectMode
    multitask_strategy: str = "reject"
    metadata: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    task: None = None
    abort_event: None = None
    abort_action: str = "interrupt"
    error: str | None = None
    model_name: str | None = None
    store_only: bool = False
    first_human_message: str | None = None


def _sse(event: str, data: Any = None, *, event_id: str | None = None) -> str:
    """格式化单条 SSE 帧（与 app.gateway.services.format_sse 保持一致）。"""
    payload = json.dumps(data, default=str, ensure_ascii=False)
    parts = [f"event: {event}", f"data: {payload}"]
    if event_id:
        parts.append(f"id: {event_id}")
    parts.append("")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """FastAPI app with mock singletons on ``app.state``。"""
    application = create_app()
    application.state.stream_bridge = AsyncMock()
    application.state.run_manager = AsyncMock()
    return application


# ---------------------------------------------------------------------------
# Tests — 使用 asyncio.run() 内嵌异步
# ---------------------------------------------------------------------------


class TestStreamRoute:
    def test_returns_sse_stream(self, app):
        """应返回 text/event-stream，包含 metadata/updates/end 事件。"""

        async def _test():
            with (
                patch("app.gateway.routers.runs.start_run") as mock_start,
                patch("app.gateway.routers.runs.see_consumer") as mock_cons,
            ):
                fake_run = FakeRunRecord(
                    run_id=str(uuid.uuid4()),
                    thread_id="t1",
                    assistant_id="test-agent",
                    status=RunStatus.pending,
                    on_disconnect=DisconnectMode.cancel,
                    created_at=datetime.now().isoformat(),  # noqa: DTZ005
                )
                mock_start.return_value = fake_run

                async def _gen(*_a, **_k):
                    yield _sse("metadata", {"run_id": fake_run.run_id})
                    yield _sse(
                        "updates",
                        {"messages": [{"role": "assistant", "content": "hi"}]},
                    )
                    yield _sse("end", None)

                mock_cons.return_value = _gen()

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/runs/stream",
                        json={
                            "assistant_id": "test-agent",
                            "input": {
                                "messages": [{"role": "user", "content": "hello"}]
                            },
                        },
                    )

            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
            assert resp.headers["cache-control"] == "no-cache"
            assert "event: metadata" in resp.text
            assert "event: updates" in resp.text
            assert "event: end" in resp.text

        asyncio.run(_test())

    def test_heartbeat_skipped(self, app):
        """心跳哨兵应只产出 ``: heartbeat`` 注释行。"""

        async def _test():
            with (
                patch("app.gateway.routers.runs.start_run") as mock_start,
                patch("app.gateway.routers.runs.see_consumer") as mock_cons,
            ):
                fake_run = FakeRunRecord(
                    run_id=str(uuid.uuid4()),
                    thread_id="t1",
                    assistant_id="test-agent",
                    status=RunStatus.pending,
                    on_disconnect=DisconnectMode.cancel,
                )
                mock_start.return_value = fake_run

                async def _gen(*_a, **_k):
                    yield ": heartbeat\n\n"
                    yield _sse("end", None)

                mock_cons.return_value = _gen()

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/runs/stream",
                        json={"assistant_id": "test-agent"},
                    )

            assert resp.status_code == 200
            assert ": heartbeat" in resp.text

        asyncio.run(_test())

    def test_missing_state_returns_503(self):
        """未设置 app.state 时应返回 503。"""

        async def _test():
            app_no_state = create_app()
            transport = httpx.ASGITransport(app=app_no_state)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/runs/stream",
                    json={"assistant_id": "test-agent"},
                )
            assert resp.status_code == 503

        asyncio.run(_test())

    def test_non_dict_input_returns_422(self, app):
        """非 dict 的 JSON 载荷应被拒绝并返回 422。"""

        async def _test():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/runs/stream",
                    content="not-json",
                    headers={"content-type": "application/json"},
                )
            assert resp.status_code == 422

        asyncio.run(_test())
