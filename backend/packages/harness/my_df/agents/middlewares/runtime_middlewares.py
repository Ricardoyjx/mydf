"""运行时中间件链构造：为 Lead Agent 构建默认中间件列表。"""

from langchain.agents.middleware import AgentMiddleware
from my_df.agents.middlewares.todo_middleware import TodoMiddleware


def build_lead_runtime_middlewares(lazy_init: bool = False) -> list[AgentMiddleware]:
    """构建 Lead Agent 的运行时中间件链。

    参数：
        lazy_init: 若为 True，中间件将被延迟初始化。

    返回：
        中间件实例列表（当前为空，预留扩展）。
    """

    middlewares: list[AgentMiddleware] = [TodoMiddleware()]  # type: ignore

    return middlewares
