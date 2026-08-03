"""Agent 运行器：在后台执行 LangGraph agent 并处理输出。

负责：
- 注入 checkpointer（图编译后附着）
- 遍历 ``astream`` 输出并逐块发布到 bridge
- 异常处理：所有错误通过 bridge 发布 error 事件
- 确保结束信号：finally 块中发布 ``__end__``
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from my_df.config.app_config import AppConfig
from my_df.runtime.stream_bridge.base import StreamBridge

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    """单次 agent 运行的基础设施依赖。

    将 checkpointer、store 等持久化单例分组，
    使 ``run_agent_mini`` 不必接收不断增长的参数列表。
    """

    checkpointer: Any
    store: Any | None = field(default=None)
    event_store: Any | None = field(default=None)
    run_events_config: Any | None = field(default=None)
    app_config: AppConfig | None = field(default=None)


async def run_agent_mini(
    agent_factory: Any,
    graph_input: dict | None,
    config: dict,
    bridge: StreamBridge,
    run_id: str,
    context: RunContext,
) -> None:
    """简化版 Agent 运行器：遍历 astream 输出并处理每个 chunk。

    参数：
        agent_factory: 已编译的 LangGraph agent（CompiledStateGraph）。
        graph_input:   图输入字典。为 None 时发布错误并立即返回。
        config:        RunnableConfig，含 thread_id、recursion_limit 等。
        bridge:        流桥接器，用于发布事件。
        run_id:        当前运行 ID。
        context:       基础设施依赖（checkpointer、store 等）。
    """
    # 1. 注入 checkpointer（图编译后附着，模仿 LangGraph Platform 方案）
    if context.checkpointer is not None:
        agent_factory.checkpointer = context.checkpointer

    # 2. 保护：graph_input 为 None
    if graph_input is None:
        await bridge.publish(
            run_id,
            "error",
            {"message": "请求体缺少 input 字段", "code": "MISSING_INPUT"},
        )
        await bridge.publish_end(run_id)
        return

    # 3. 遍历 astream，捕获所有异常
    try:
        async for chunk in agent_factory.astream(graph_input, config=config):
            try:
                await process_chunk(bridge, run_id, chunk)
            except Exception as e:
                logger.exception("处理 chunk 时出错 (run_id=%s)", run_id)
                await bridge.publish(
                    run_id,
                    "error",
                    {"message": f"处理事件时出错: {e}", "code": "CHUNK_ERROR"},
                )
    except asyncio.CancelledError:
        logger.info("Agent 运行被取消 (run_id=%s)", run_id)
        await bridge.publish(
            run_id, "error", {"message": "运行已被取消", "code": "CANCELLED"}
        )
    except Exception as e:
        logger.exception("Agent 运行失败 (run_id=%s)", run_id)
        await bridge.publish(
            run_id,
            "error",
            {"message": f"Agent 执行出错: {e}", "code": "AGENT_ERROR"},
        )
    finally:
        # 确保结束哨兵一定会发送，前端才能知道流已结束
        await bridge.publish_end(run_id)


async def process_chunk(bridge: StreamBridge, run_id: str, chunk: dict):
    """处理单个 agent 输出 chunk。

    astream stream_mode="updates" 格式为 ``{node_name: output}``。
    只关注名为 ``"model"`` 的节点（真正的 LLM 调用输出），
    跳过中间件注入（MemoryMiddleware、DynamicContextMiddleware）。
    """
    try:
        for node_name, output in chunk.items():
            # 只处理 model 节点的输出（真正的 AI 回复）
            if node_name != "model":
                continue
            if not isinstance(output, dict):
                continue

            messages = output.get("messages")
            if not messages or len(messages) == 0:
                continue

            last = messages[-1]
            # 优先 .content 属性
            text = getattr(last, "content", None)
            if text:
                await bridge.publish(run_id, "updates", str(text))
                logger.debug("published from %s: %s", node_name, str(text)[:80])
                return
            # 回退：str() 后正则提取
            raw = str(last)
            m = re.search(r"content='([^']*)'", raw)
            if m:
                await bridge.publish(run_id, "updates", m.group(1))
                return
            # 最终回退：发原始字符串
            await bridge.publish(run_id, "updates", raw[:200])
    except Exception as e:
        logger.error("process_chunk error: %s", e, exc_info=True)  # noqa: G201
        await bridge.publish(run_id, "error", {"message": f"process_chunk: {e}"})
