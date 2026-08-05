from fastapi import APIRouter, HTTPException, Request
from my_df.agents.memory.updater import get_memory_data_async
from my_df.runtime.user_context import get_effective_user_id
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api", tags=["memory"])


class ContextSection(BaseModel):
    summary: str = Field(default="", description="摘要")
    updatedAt: str = Field(default="", description="更新时间")


class UserContext(BaseModel):
    workContext: ContextSection = Field(default_factory=ContextSection)
    personalContext: ContextSection = Field(default_factory=ContextSection)
    topOfMind: ContextSection = Field(default_factory=ContextSection)


class HistoryContext(BaseModel):
    recentMonths: ContextSection = Field(default_factory=ContextSection)
    earlierContext: ContextSection = Field(default_factory=ContextSection)
    longTermBackground: ContextSection = Field(default_factory=ContextSection)


class Fact(BaseModel):
    id: str = Field(..., description="Unique identifier for the fact")
    content: str = Field(..., description="Fact content")


class MemoryResponse(BaseModel):
    version: str = Field(default="1.0", description="Memory response version")
    lastUpdated: str = Field(default="", description="Last updated time")
    user: UserContext = Field(default_factory=UserContext)
    history: HistoryContext = Field(default_factory=HistoryContext)
    facts: list[Fact] = Field(default_factory=list)


@router.get(
    "/{thread_id}/memory",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="Get memory Data",
    description="Get memory Data",
)
async def get_memory(request: Request) -> MemoryResponse:
    """Get the current user's memory data from LangGraph Store.

    与 MemoryMiddleware 共用 Store（namespace=("user", user_id)）中的数据，
    保证 API 与对话记忆读写一致。``thread_id`` 仅保留路径兼容，
    记忆按用户（user_id）而非线程隔离。

    Returns:
        The current memory data with user context, history, and facts.

    Example Response:
        ```json
        {
            "version": "1.0",
            "lastUpdated": "2024-01-15T10:30:00Z",
            "user": {
                "workContext": {"summary": "Working on my-df project", "updatedAt": "..."},
                "personalContext": {"summary": "Prefers concise responses", "updatedAt": "..."},
                "topOfMind": {"summary": "Building memory API", "updatedAt": "..."}
            },
            "history": {
                "recentMonths": {"summary": "Recent development activities", "updatedAt": "..."},
                "earlierContext": {"summary": "", "updatedAt": ""},
                "longTermBackground": {"summary": "", "updatedAt": ""}
            },
            "facts": [
                {
                    "id": "fact_abc123",
                    "content": "User prefers TypeScript over JavaScript",
                    "category": "preference",
                    "confidence": 0.9,
                    "createdAt": "2024-01-15T10:30:00Z",
                    "source": "thread_xyz"
                }
            ]
        }
        ```
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Store not available")
    memory_data = await get_memory_data_async(store, user_id=get_effective_user_id())
    return MemoryResponse(**memory_data)
