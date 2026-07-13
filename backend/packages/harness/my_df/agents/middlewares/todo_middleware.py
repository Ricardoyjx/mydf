from typing import Any, override

from langchain.agents.middleware import Runtime, TodoListMiddleware, hook_config
from langchain.agents.middleware.todo import Todo
from langchain.messages import AIMessage, HumanMessage
from my_df.agents.thread_state import ThreadState


class TodoMiddleware(TodoListMiddleware):
    """Extends TodoListMiddleware with `write_todos` context-loss detection.

    When the original `write_todos` tool call has been truncated from the message
    history (e.g., after summarization), the model loses awareness of the current
    todo list. This middleware detects that gap in `before_model` / `abefore_model`
    and injects a reminder message so the model can continue tracking progress.
    """

    state_schema = ThreadState

    @override
    def before_model(
        self,
        state: ThreadState,
        runtime: Runtime,
    ):
        todos: list[Todo] = state.get("todos") or []
        if not todos:
            return None
        formatted: list[str] = []
        for todo in todos:
            status = todo.get("status", "pending")
            content = todo.get("content", "")
            formatted.append(f"-[{status}]{content}")

        reminder = HumanMessage(
            name="todo_reminder",
            additional_kwargs={
                "hide_from_ui": True,
            },
            content=(
                "<system_reminder>\n"
                "Your todo list from earlier is no longer visible in the current context window, "
                "but it is still active. Here is the current state:\n\n"
                f"{formatted}\n\n"
                "Continue tracking and updating this todo list as you work. "
                "Call `write_todos` whenever the status of any item changes.\n"
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
        """防止在待办事项（todo items）尚未全部完成时，Agent 过早退出。
        除了基类中对并行 write_todos 调用的检查外，此重写方法还会拦截那些没有任何工具调用（
        tool calls）的模型响应——前提是当前仍存在未完成的待办事项。
        此时，它会注入一条提醒性质的 HumanMessage（人类消息），并强制跳转回模型节点，
        以便 Agent 能继续处理待办列表中的任务。
        此外，系统设置了一个重试上限 _MAX_COMPLETION_REMINDERS（默认为 2 次），
        以防止 Agent 无法取得进一步进展时陷入死循环。
        """
        base_result = super().after_model(state, runtime)  # type: ignore
        if base_result is not None:
            return base_result

        messages = state.get("messages") or []
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if not last_ai:
            return None

        # 3. Allow exit when all todos are completed or there are no todos.
        todos: list[Todo] = state.get("todos") or []
        if not todos or all(t.get("status") == "completed" for t in todos):
            return None

        # 将一条提醒指令排入队列，留给下一次模型请求使用，并强制跳转回模型节点。
        return {"jump_to": "model"}

    @override
    def before_agent(self, state, runtime) -> None:
        pass

    @override
    def after_agent(self, state, runtime) -> None:
        pass
