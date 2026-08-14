import asyncio
import logging
from typing import Any, NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from my_df.agents.thread_state import SandboxState, ThreadDataState
from my_df.sandbox.sandbox_provider import get_sandbox_provider

logger = logging.getLogger(__name__)


class SandboxMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    sandbox: NotRequired[SandboxState | None]
    thread_data: NotRequired[ThreadDataState | None]


class SandboxMiddleware(AgentMiddleware[SandboxMiddlewareState]):
    state_schema = SandboxMiddlewareState

    def __init__(self, lazy_init: bool = True):
        super().__init__()
        self._lazy_init = lazy_init

    def _acquire_sandbox(self, thread_id: str) -> str:
        provider = get_sandbox_provider()
        sandbox_id = provider.acquire(thread_id)
        logger.info("Acquiring sandbox for thread_id=%s ")
        return sandbox_id

    async def _release_sandbox_async(self, sandbox_id: str) -> None:
        provider = get_sandbox_provider()
        releaser = provider.release

        await asyncio.to_thread(releaser, sandbox_id)

    def before_agent(
        self, state: SandboxMiddlewareState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Skip acquisition if lazy_init is enabled
        if self._lazy_init:
            return super().before_agent(state, runtime)

        # Eager initialization (original behavior)
        if "state" not in state or state["sandbox"] is None:
            thread_id = (runtime.context or {}).get("thread_id")
            if thread_id is None:
                return super().before_agent(state, runtime)
            sandbox_id = self._acquire_sandbox(thread_id)
            logger.info(
                "Acquired sandbox: sandbox_id=%s thread_id=%s", sandbox_id, thread_id
            )
            return {"sandbox": sandbox_id}
        return super().before_agent(state, runtime)

    def after_agent(
        self, state: SandboxMiddlewareState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # 从状态中释放沙箱
        sandbox = state.get("sandbox")
        if sandbox is not None:
            sandbox_id = sandbox.get("sandbox_id")
            logger.info("Releasing sandbox: sandbox_id=%s", sandbox_id)
            if sandbox_id is None:
                return super().after_agent(state, runtime)
            get_sandbox_provider().release(sandbox_id)
            return None
        # 从 runtime context 释放沙箱
        if runtime.context and (runtime.context or {}).get("sandbox_id") is not None:
            sandbox_id = runtime.context.get("sandbox_id")
            logger.info("Releasing sandbox: sandbox_id=%s", sandbox_id)
            get_sandbox_provider().release(sandbox_id)
            return None
        # No sandbox to release
        return super().after_agent(state, runtime)

    @override
    async def aafter_agent(
        self, state: SandboxMiddlewareState, runtime: Runtime
    ) -> dict | None:
        sandbox = state.get("sandbox")
        if sandbox is not None:
            sandbox_id = sandbox.get("sandbox_id")
            logger.info(f"Releasing sandbox {sandbox_id}")
            if sandbox_id is None:
                return super().after_agent(state, runtime)
            await self._release_sandbox_async(sandbox_id)
            return None

        if runtime.context and (runtime.context or {}).get("sandbox_id") is not None:
            sandbox_id = runtime.context.get("sandbox_id")
            logger.info(f"Releasing sandbox {sandbox_id} from context")
            await self._release_sandbox_async(sandbox_id)
            return None

        # No sandbox to release
        return await super().aafter_agent(state, runtime)
