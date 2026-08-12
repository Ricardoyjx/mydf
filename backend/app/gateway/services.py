"""运行管理服务：创建运行、消费 SSE 流、格式化 SSE 帧。"""

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from app.gateway.deps import get_run_manager
from fastapi import HTTPException, Request
from my_df.agents.supervisor_graph import build_supervisor_graph
from my_df.config.app_config import get_app_config
from my_df.runtime.runs.manager import RunManager, RunRecord
from my_df.runtime.runs.schema import DisconnectMode, RunStatus
from my_df.runtime.runs.worker import RunContext, run_agent
from my_df.runtime.stream_bridge.base import StreamBridge, StreamEvent

logger = logging.getLogger(__name__)


async def start_run(
    body: Any,
    thread_id: str,
    request: Request,
    context: RunContext,
    bridge: StreamBridge,
) -> RunRecord:
    """通过 RunManager 注册运行记录并启动后台 agent 任务。

    状态流转：``pending -> running -> success / error / interrupted``；
    终态由 task 完成回调统一收尾并同步持久化存储。
    """
    run_mgr = get_run_manager(request)
    model_name = (
        context.app_config.models[0].name
        if context.app_config and context.app_config.models
        else None
    )
    disconnect = (
        DisconnectMode.cancel
        if body.on_disconnect == "cancel"
        else DisconnectMode.continue_
    )

    try:
        record = await run_mgr.create(
            thread_id=thread_id,
            assistant_id=body.assistant_id,
            on_disconnect=disconnect,
            metadata=body.metadata or {},
            kwargs=body.config or {},
            multitask_strategy=body.multitask_strategy,
            model_name=model_name,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    # # 构造 agent 配置并启动后台运行
    # agent_config: RunnableConfig = {
    #     "recursion_limit": 100,
    #     "configurable": {"user_id": "default"},
    # }
    # if body.assistant_id:
    #     agent_config["configurable"]["assistant_id"] = body.assistant_id
    # agent_factory = make_lead_agent(
    #     agent_config,
    #     store=context.store,
    #     milvus=getattr(request.app.state, "milvus", None),
    #     embedding_model=getattr(request.app.state, "embedding_model", None),
    # )  # type: ignore

    # 优先复用启动时预热的 Supervisor 编排图；未预热/失败时按需构建
    agent_factory = getattr(request.app.state, "agent_factory", None)
    if agent_factory is None:
        app_config = context.app_config or get_app_config()
        agent_factory = build_supervisor_graph(
            app_config,
            store=context.store,
            milvus=getattr(request.app.state, "milvus", None),
            embedding_model=getattr(request.app.state, "embedding_model", None),
        )
    graph_input = body.input
    config = build_run_config(
        thread_id=thread_id,
        request_config=body.config,
        metadata=record.metadata,
        assistant_id=body.assistant_id,
    )
    task = asyncio.create_task(
        run_agent(
            agent_factory=agent_factory,
            graph_input=graph_input,
            config=config,
            run_id=record.run_id,
            bridge=bridge,
            context=context,
        )
    )

    record.task = task
    # 标记运行中；终态由 _finalize_run 回调统一收尾
    await run_mgr.update_status(record.run_id, RunStatus.running)
    task.add_done_callback(
        lambda done: asyncio.create_task(_finalize_run(run_mgr, record, done))
    )
    return record


async def _finalize_run(
    run_mgr: RunManager,
    record: RunRecord,
    task: asyncio.Task,
) -> None:
    """task 完成回调：把运行终态写入 RunManager 与持久化存储。"""
    if task.cancelled():
        await run_mgr.update_status(
            record.run_id, RunStatus.interrupted, error="运行已被取消"
        )
        return
    try:
        result = task.result()
    except Exception as e:
        logger.error("运行 %s 异常退出: %s", record.run_id, e, exc_info=True)  # noqa: G201
        await run_mgr.update_status(record.run_id, RunStatus.error, error=str(e))
        return
    if isinstance(result, tuple) and len(result) == 2:
        status, stats = result
    else:
        status, stats = result, None
    if isinstance(status, RunStatus):
        await run_mgr.update_status(record.run_id, status)
        if stats is not None:
            await run_mgr.update_run_completion(
                record.run_id, status=status, stats=stats
            )


# 流结束哨兵与心跳哨兵（与 stream_bridge/base.py 中的定义保持一致）
HEARTBEAT_SENTINEL = StreamEvent(id="", event="__heartbeat__", data=None)
END_SENTINEL = StreamEvent(id="", event="__end__", data=None)


async def see_consumer(
    bridge: StreamBridge,
    record: RunRecord,
    request: Request,
    run_mgr: RunManager,
):
    """异步生成器：从 bridge 读取事件并产出 SSE 格式字符串。

    ``finally`` 块实现了 ``on_disconnect`` 语义：
    - ``cancel``：客户端断开时取消后台任务。
    - ``continue``：让任务继续执行，事件被丢弃。
    """
    last_event_id = request.headers.get("Last-Event-ID")
    try:
        async for entry in bridge.subscribe(record.run_id, last_event_id=last_event_id):
            # 客户端已断开则停止消费
            if await request.is_disconnected():
                break

            # 心跳哨兵 → 产出 SSE 注释行
            if entry is HEARTBEAT_SENTINEL:
                yield ": heartbeat\n\n"
                continue

            # 结束哨兵 → 产出 end 事件并终止
            if entry is END_SENTINEL:
                yield format_sse("end", None, event_id=entry.id or None)
                return

            # 普通事件 → 格式化后产出
            yield format_sse(entry.event, entry.data, event_id=entry.id or None)

    finally:
        # 若运行仍在进行中且配置为取消，则终止任务
        if (
            record.status in (RunStatus.pending, RunStatus.running)
            and record.on_disconnect == DisconnectMode.cancel
        ):
            await run_mgr.cancel(record.run_id)


_DEFAULT_ASSISTANT_ID = "lead_agent"


def build_run_config(
    thread_id: str,
    request_config: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    assistant_id: str | None = None,
) -> dict[str, Any]:
    """构造 RunnableConfig 字典。

    当 ``assistant_id`` 指向自定义 agent（非 ``"lead_agent"`` / ``None``）时，
    将名称以 ``agent_name`` 形式传入运行配置，供 ``make_lead_agent`` 加载对应的
    ``agents/<name>/SOUL.md`` 和 per-agent 配置。
    """
    config: dict[str, Any] = {"recursion_limit": 100}
    if request_config:
        if "context" in request_config:
            context_value = request_config["context"]
            if context_value is None:
                context = {}
            elif isinstance(context_value, Mapping):
                context = dict(context_value)
            else:
                raise ValueError("Invalid context value")
            config["context"] = context
        else:
            configurable = {"thread_id": thread_id}
            configurable.update(request_config.get("configurable", {}))
            config["configurable"] = configurable
        for k, v in request_config.items():
            if k not in ("configurable", "context"):
                config[k] = v
    else:
        config["configurable"] = {"thread_id": thread_id}

    # Inject custom agent name when the caller specified a non-default assistant.
    # Honour an explicit agent_name in the active runtime options container.
    if assistant_id and assistant_id != _DEFAULT_ASSISTANT_ID:
        normalized = assistant_id.strip().lower().replace("_", "-")
        if not normalized or not re.fullmatch(r"[a-z0-9-]+", normalized):
            raise ValueError(
                f"Invalid assistant_id {assistant_id!r}: must contain only letters, digits, and hyphens after normalization."
            )
        if "configurable" in config:
            target = config["configurable"]
        elif "context" in config:
            target = config["context"]
        else:
            target = config.setdefault("configurable", {})
        if target is not None and "agent_name" not in target:
            target["agent_name"] = normalized
        config.setdefault("run_name", resolve_root_run_name(config, normalized))
    if metadata:
        config.setdefault("metadata", {}).update(metadata)
    return config


def resolve_root_run_name(config: Mapping[str, Any], assistant_id: str | None) -> str:
    for container_name in ("context", "configurable"):
        container = config.get(container_name)
        if isinstance(container, Mapping):
            agent_name = container.get("agent_name")
            if isinstance(agent_name, str) and agent_name.strip():
                return agent_name
    return assistant_id or "lead_agent"


def format_sse(event: str, data: Any, *, event_id: str | None = None) -> str:
    """格式化单条 SSE 帧。

    字段顺序：``event:`` → ``data:`` → ``id:``（可选）→ 空行。
    与 LangGraph Platform 的 ``useStream`` React hook 和 Python ``langgraph-sdk``
    SSE 解码器兼容。
    """
    payload = json.dumps(data, default=str, ensure_ascii=False)
    parts = [f"event: {event}", f"data: {payload}"]
    if event_id:
        parts.append(f"id: {event_id}")
    parts.append("")
    parts.append("")
    return "\n".join(parts)
