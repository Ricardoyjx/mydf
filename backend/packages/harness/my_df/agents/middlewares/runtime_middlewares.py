from langchain.agents.middleware import AgentMiddleware


def build_lead_runtime_middlewares(lazy_init: bool = False) -> list[AgentMiddleware]:
    """Build middleware chain for LeadAgent.

    Args:
        lazy_init: If True, middlewares will be initialized lazily.

    Returns:
        List of middleware instances.
    """

    middlewares: list[AgentMiddleware] = []

    return middlewares
