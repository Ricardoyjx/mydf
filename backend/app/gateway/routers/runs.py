"""SSE 流式运行端点：接收请求、启动 agent、以 Server-Sent Events 返回结果。"""

import uuid

from app.gateway.deps import (
    get_run_context,
    get_run_event_store,
    get_run_manager,
    get_stream_bridge,
)
from app.gateway.routers.thread_runs import RunCreateRequest
from app.gateway.services import see_consumer, start_run
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("/{run_id}/events", summary="运行事件流水（可观测性）")
async def get_run_events(
    run_id: str,
    request: Request,
    event_type: str | None = Query(default=None, description="按事件类型过滤"),
    limit: int = Query(default=500, ge=1, le=5000),
):
    """返回一次运行的可观测性事件流（run_start/route/subagent/reflect/token/run_end）。"""
    event_store = get_run_event_store(request)
    event_types = [event_type] if event_type else None
    events = await event_store.list_events(
        thread_id="",
        run_id=run_id,
        event_types=event_types,
        limit=limit,
    )
    return {"run_id": run_id, "count": len(events), "events": events}


@router.post("/stream")
async def stateless_stream(
    body: RunCreateRequest, request: Request
) -> StreamingResponse:
    """创建一次运行并通过 SSE 流式返回事件。

    如果 ``config.configurable.thread_id`` 提供了线程 ID，则在该线程上创建运行以保留对话历史；
    否则创建一个临时线程。
    """
    # get thread_ id
    thread_id = str(
        (body.config or {}).get("configurable", {}).get("thread_id") or uuid.uuid4()
    )

    bridge = get_stream_bridge(request)
    run_mgr = get_run_manager(request)
    run_context = get_run_context(request)
    record = await start_run(
        body=body,
        thread_id=thread_id,
        request=request,
        context=run_context,
        bridge=bridge,
    )

    return StreamingResponse(
        see_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("", summary="获取运行列表")
async def list_runs(
    request: Request,
    user_id: str | None = Query(default=None, description="用户ID"),
    status: str | None = Query(default=None, description="运行状态"),
    thread_id: str | None = Query(default=None, description="主题ID"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """列出运行记录，支持按用户/状态/线程过滤与分页。"""
    run_mgr = get_run_manager(request)
    runs = await run_mgr.list_runs(
        user_id=user_id,
        status=status,
        thread_id=thread_id,
        limit=limit,
        offset=offset,
    )
    return {"runs": runs, "count": len(runs), "limit": limit, "offset": offset}


@router.delete("/{runs_id}")
async def delete_run(
    runs_id: str,
    request: Request,
):
    """删除运行记录。"""
    run_mgr = get_run_manager(request)
    await run_mgr.delete_run(runs_id)
    return {"status": "ok"}


@router.post("/{run_id}/cancel")
async def cancel(run_id: str, request: Request):
    run_mgr = get_run_manager(request)
    ok = await run_mgr.cancel(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="运行不存在或已结束")
    return {"run_id": run_id, "status": "interrupted"}
