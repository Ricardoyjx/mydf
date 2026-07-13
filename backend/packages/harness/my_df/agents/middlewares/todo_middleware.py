"""TodoMiddleware：在上下文丢失时自动恢复待办事项提醒。"""

from typing import Any, override

from langchain.agents.middleware import Runtime, TodoListMiddleware, hook_config
from langchain.agents.middleware.todo import Todo
from langchain.messages import AIMessage, HumanMessage
from my_df.agents.thread_state import ThreadState


class TodoMiddleware(TodoListMiddleware):
    """扩展 TodoListMiddleware，增加 `write_todos` 上下文丢失检测。

    当原始的 `write_todos` 工具调用因上下文窗口限制被截断（如摘要后）时，
    模型会丢失当前待办列表的认知。此中间件在 ``before_model`` / ``abefore_model``
    中检测到该空隙，并注入一条提醒消息，使模型能继续追踪进度。
    """

    state_schema = ThreadState

    @override
    def before_model(
        self,
        state: ThreadState,
        runtime: Runtime,
    ):
        """模型调用前：若有 active 待办但上下文已丢失，注入提醒。"""
        todos: list[Todo] = state.get("todos") or []
        if not todos:
            return None
        # 格式化当前待办状态
        formatted: list[str] = []
        for todo in todos:
            status = todo.get("status", "pending")
            content = todo.get("content", "")
            formatted.append(f"-[{status}]{content}")

        reminder = HumanMessage(
            name="todo_reminder",
            additional_kwargs={
                "hide_from_ui": True,  # 不向用户展示此消息
            },
            content=(
                "<system_reminder>\n"
                "你的待办列表在之前的上下文中已不可见，但它仍然活跃。当前状态：\n\n"
                f"{formatted}\n\n"
                "请继续追踪和更新此待办列表。任何项目状态变化时请调用 `write_todos`。\n"
                "</system_reminder>"
            ),
        )

        return reminder

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(
        self,
        state: ThreadState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """模型调用后：防止 Agent 在待办未完成时过早退出。

        除了基类对并行 write_todos 调用的检查外，此方法还会拦截那些没有任何
        工具调用的模型响应——前提是仍有未完成的待办事项。此时强制跳转回模型节点，
        使 Agent 能继续处理待办列表中的任务。
        """
        # 先让基类处理它的逻辑
        base_result = super().after_model(state, runtime)  # type: ignore
        if base_result is not None:
            return base_result

        messages = state.get("messages") or []
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if not last_ai:
            return None

        # 所有待办已完成 → 允许退出
        todos: list[Todo] = state.get("todos") or []
        if not todos or all(t.get("status") == "completed" for t in todos):
            return None

        # 仍有未完成的待办 → 跳回模型继续处理
        return {"jump_to": "model"}

    @override
    def before_agent(self, state, runtime) -> None:
        """Agent 调用前 hook（当前无操作）。"""
        pass

    @override
    def after_agent(self, state, runtime) -> None:
        """Agent 调用后 hook（当前无操作）。"""
        pass
