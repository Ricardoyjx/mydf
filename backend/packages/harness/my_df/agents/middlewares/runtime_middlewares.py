"""运行时中间件链构造：为 Lead Agent 构建默认中间件列表。

TodoMiddleware 不由此模块管理，而是在 ``lead_agent/agent.py`` 的
``_build_middlewares()`` 中根据 ``is_plan_mode`` 条件注册。
"""

from langchain.agents.middleware import AgentMiddleware


def build_lead_runtime_middlewares(lazy_init: bool = False) -> list[AgentMiddleware]:
    """构建 Lead Agent 的运行时中间件链。

    参数：
        lazy_init: 若为 True，中间件将被延迟初始化（当前未使用，预留扩展）。

    返回：
        中间件实例列表。当前为空（所有中间件由 ``_build_middlewares`` 统一注册）。
    """
    # 此处预留将来加入与 is_plan_mode 无关的通用中间件
    middlewares: list[AgentMiddleware] = []

    return middlewares
