from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.gateway.routers.thread_runs import RunCreateRequest
from app.gateway.services import see_consumer, start_run
from app.gateway.deps import get_stream_bridge, get_run_manager

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("/stream")
async def stateless_stream(
    body: RunCreateRequest, request: Request
) -> StreamingResponse:
    """Create a run and stream events via SSE.

    If ``config.configurable.thread_id`` is provided, the run is created
    on the given thread so that conversation history is preserved.
    Otherwise a new temporary thread is created.
    """

    thread_id = "1"
    bridge = get_stream_bridge(request)
    run_mgr = get_run_manager(request)
    record = await start_run(body, thread_id, request)

    return StreamingResponse(
        see_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
