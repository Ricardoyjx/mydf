"""线程状态类型定义：LangGraph Agent 的核心 State schema。"""

from typing import Annotated, NotRequired, TypedDict

from langchain.agents import AgentState
from langchain.agents.middleware.todo import Todo


class SandboxState(TypedDict):
    """沙箱环境状态。"""

    sandbox_id: NotRequired[str | None]


class ThreadDataState(TypedDict):
    """线程数据目录状态。"""

    workspace_path: NotRequired[str | None]
    uploads_path: NotRequired[str | None]
    outputs_path: NotRequired[str | None]


class ViewedImageData(TypedDict):
    """已查看图像的缓存数据。"""

    base64: str
    mime_type: str


def merge_artifacts(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Reducer：合并并去重 artifacts 列表（保留顺序）。"""
    if existing is None:
        return new or []
    if new is None:
        return existing
    # 用 dict.fromkeys 去重同时保留顺序
    return list(dict.fromkeys(existing + new))


def merge_viewed_images(
    existing: dict[str, ViewedImageData] | None, new: dict[str, ViewedImageData] | None
) -> dict[str, ViewedImageData]:
    """Reducer：合并 viewed_images 字典。

    特殊规则：若 new 为空字典 {}，则清空现有的 viewed_images。
    这允许中间件在处理完图像后清除状态。
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    # 空字典 = 清除信号
    if len(new) == 0:
        return {}
    # 合并，新值覆盖旧值
    return {**existing, **new}


def merge_todos(
    existing: list[Todo] | None, new: list[Todo] | None
) -> list[Todo] | None:
    """Reducer：todos 列表取最后一个非 None 值。

    语义：
    - 若 new 为 None（节点未操作 todos），保留 existing。
    - 若 new 有值（即使为空列表），表示显式更新，覆盖 existing。
    """
    if new is None:
        return existing
    return new


class ThreadState(AgentState):
    """线程级 Agent 状态，包含沙箱、文件、待办事项和已查看图像等信息。"""

    sandbox: NotRequired[SandboxState | None]
    thread_data: NotRequired[ThreadDataState | None]
    title: NotRequired[str | None]
    artifacts: Annotated[list[str], merge_artifacts]
    todos: Annotated[list[Todo] | None, merge_todos]
    uploaded_files: NotRequired[list[dict] | None]
    viewed_images: Annotated[
        dict[str, ViewedImageData], merge_viewed_images
    ]  # image_path -> {base64, mime_type}
