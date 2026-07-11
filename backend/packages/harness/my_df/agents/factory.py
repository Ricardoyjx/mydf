from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def create_mydf_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: list[AgentMiddleware] | None = None,
    # features
    # extra_middleware
    plan_mode: bool = False,
    state_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    name: str = "default",
) -> CompiledStateGraph:
    """Parameters
    ----------
    model:
        Chat model instance.
    tools:
        User-provided tools.  Feature-injected tools are appended automatically.
    system_prompt:
        System message.  ``None`` uses a minimal default.
    middleware:
        **Full takeover** — if provided, this exact list is used.
        Cannot be combined with *features* or *extra_middleware*.
    features:
        Declarative feature flags.  Cannot be combined with *middleware*.
    extra_middleware:
        Additional middlewares inserted into the auto-assembled chain via
        ``@Next``/``@Prev`` positioning.  Cannot be used with *middleware*.
    plan_mode:
        Enable TodoMiddleware for task tracking.
    state_schema:
        LangGraph state type.  Defaults to ``ThreadState``.
    checkpointer:
        Optional persistence backend.
    name:
        Agent name (passed to middleware that cares, e.g. ``MemoryMiddleware``).

    Raises
    ------
    ValueError
        If both *middleware* and *features*/*extra_middleware* are provided.
    """

    effective_tools: list[BaseTool] = list(tools or [])
    effective_state = state_schema

    if middleware is not None:
        effective_middleware = list(middleware)

    return create_agent(
        model=model,
        tools=effective_tools or None,
        middleware=effective_middleware,
        system_prompt=system_prompt,
        state_schema=effective_state,
        checkpointer=checkpointer,
        name=name,
    )
