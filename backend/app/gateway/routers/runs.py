"""SSE 流式运行端点：接收请求、启动 agent、以 Server-Sent Events 返回结果。"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.gateway.routers.thread_runs import RunCreateRequest
from app.gateway.services import see_consumer, start_run
from app.gateway.deps import get_stream_bridge, get_run_manager, get_run_context

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("/stream")
async def stateless_stream(
    body: RunCreateRequest, request: Request
) -> StreamingResponse:
    """创建一次运行并通过 SSE 流式返回事件。

    如果 ``config.configurable.thread_id`` 提供了线程 ID，则在该线程上创建运行以保留对话历史；
    否则创建一个临时线程。
    """

    thread_id = "1"
    bridge = get_stream_bridge(request)
    run_mgr = get_run_manager(request)
    run_context = get_run_context(request)
    record = await start_run(body, thread_id, request, run_context, bridge)

    return StreamingResponse(
        see_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
