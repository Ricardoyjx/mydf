"""运行管理服务：创建运行、消费 SSE 流、格式化 SSE 帧。"""

import asyncio
from datetime import datetime
import json
import re
from typing import Any, Mapping
import uuid

from my_df.agents.lead_agent.agent import make_lead_agent
from my_df.runtime.runs.manager import RunManager, RunRecord
from fastapi import HTTPException, Request
from my_df.runtime.stream_bridge.base import StreamBridge, StreamEvent
from my_df.runtime.runs.schema import DisconnectMode, RunStatus
from my_df.runtime.runs.worker import RunContext, run_agent_mini


async def start_run(
    body: Any,
    thread_id: str,
    request: Request,
    context: RunContext,
    bridge: StreamBridge,
) -> RunRecord:
    """创建 RunRecord 并启动后台 agent 任务。

    参数：
        body:      验证后的请求体（类型标注为 Any 以避免与定义 Pydantic 模型的 router 模块产生循环导入）。
        thread_id: 目标线程 ID。
        request:   FastAPI 请求对象，用于从 ``app.state`` 获取单例。
    """
    # run_mgr = get_run_manager(request)  # 暂未启用持久化管理

    now = datetime.now().isoformat()
    # 根据请求决定断开行为
    disconnect = (
        DisconnectMode.cancel
        if body.on_disconnect == "cancel"
        else DisconnectMode.continue_
    )
    run_id = str(uuid.uuid4())

    try:
        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=body.assistant_id,
            status=RunStatus.pending,
            on_disconnect=disconnect,
            multitask_strategy="rollback",
            metadata={},
            kwargs={},
            created_at=now,
            updated_at=now,
            model_name="deepseek-v4-flash",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 构造 agent 配置并启动后台运行
    agent_config: dict[str, Any] = {"recursion_limit": 100}
    if body.assistant_id:
        agent_config.setdefault("configurable", {})["assistant_id"] = body.assistant_id
    agent_factory = make_lead_agent(agent_config)  # type: ignore
    graph_input = body.input
    config = build_run_config(
        thread_id=thread_id,
        request_config=body.config,
        metadata=record.metadata,
        assistant_id=body.assistant_id,
    )
    task = asyncio.create_task(
        run_agent_mini(
            agent_factory=agent_factory,
            graph_input=graph_input,
            config=config,
            run_id=run_id,
            bridge=bridge,
            context=context,
        )
    )

    record.task = task
    return record


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
        if record.status in (RunStatus.pending, RunStatus.running):
            if record.on_disconnect == DisconnectMode.cancel:
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
