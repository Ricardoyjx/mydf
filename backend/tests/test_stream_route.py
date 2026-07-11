"""Tests for ``/api/runs/stream`` SSE endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch
import json
import uuid

import httpx
import pytest

from app.gateway.app import create_app
from my_df.runtime.runs.schema import DisconnectMode, RunStatus

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeRunRecord:
    """Minimal stand-in for ``RunRecord`` used by the SSE consumer."""

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
    """Format a single SSE frame (mirrors ``app.gateway.services.format_sse``)."""
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
    """FastAPI app with mock singletons on ``app.state``."""
    application = create_app()
    application.state.stream_bridge = AsyncMock()
    application.state.run_manager = AsyncMock()
    return application


@pytest.fixture
async def client(app):
    """Async HTTP client bound to the test app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamRoute:
    async def test_returns_sse_stream(self, client: httpx.AsyncClient):
        """Should return ``text/event-stream`` with metadata/updates/end."""
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
                created_at=datetime.now().isoformat(),
            )
            mock_start.return_value = fake_run

            async def _gen(*_a, **_k):
                yield _sse("metadata", {"run_id": fake_run.run_id})
                yield _sse(
                    "updates", {"messages": [{"role": "assistant", "content": "hi"}]}
                )
                yield _sse("end", None)

            mock_cons.return_value = _gen()

            resp = await client.post(
                "/api/runs/stream",
                json={
                    "assistant_id": "test-agent",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                },
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert resp.headers["cache-control"] == "no-cache"
        assert "event: metadata" in resp.text
        assert "event: updates" in resp.text
        assert "event: end" in resp.text

    async def test_heartbeat_skipped(self, client: httpx.AsyncClient):
        """Heartbeat sentinel yields ``: heartbeat`` comment only."""
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

            resp = await client.post(
                "/api/runs/stream",
                json={"assistant_id": "test-agent"},
            )

        assert resp.status_code == 200
        assert ": heartbeat" in resp.text

    async def test_missing_state_returns_503(self):
        """Without app.state, should return 503."""
        app_no_state = create_app()
        transport = httpx.ASGITransport(app=app_no_state)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/runs/stream",
                json={"assistant_id": "test-agent"},
            )
        assert resp.status_code == 503

    async def test_non_dict_input_returns_422(self, client: httpx.AsyncClient):
        """Non-dict JSON payload should be rejected with 422."""
        resp = await client.post(
            "/api/runs/stream",
            content="not-json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
