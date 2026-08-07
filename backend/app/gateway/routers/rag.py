"""RAG 知识库 API：文档入库、列表、语义检索与删除。"""

import base64
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from my_df.rag.service import KnowledgeService, extract_text_from_file
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/rag", tags=["rag"])


class RagDocumentCreate(BaseModel):
    """文本方式创建知识库文档。"""

    title: str = Field(default="未命名文档", description="文档标题")
    content: str = Field(..., min_length=1, description="文档正文")
    user_id: str = Field(default="default", description="用户 ID")
    source: str = Field(default="manual", description="来源标识")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="附加元数据（可选）"
    )
    chunk_size: int | None = Field(
        default=None, ge=100, le=8000, description="分块大小，覆盖默认值"
    )
    chunk_overlap: int | None = Field(
        default=None, ge=0, le=1000, description="分块重叠，覆盖默认值"
    )


class RagSearchRequest(BaseModel):
    """语义检索请求。"""

    query: str = Field(..., min_length=1, description="检索问题")
    user_id: str = Field(default="default", description="用户 ID")
    top_k: int = Field(default=5, ge=1, le=20, description="返回条数")
    min_score: float | None = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="最低相似度阈值，低于该分的片段会被过滤",
    )


class RagFileUploadRequest(BaseModel):
    """JSON 方式上传文本文件内容。"""

    filename: str = Field(..., min_length=1, description="文件名，将作为默认标题")
    content: str = Field(..., min_length=1, description="文件正文（UTF-8 文本）")
    user_id: str = Field(default="default", description="用户 ID")
    source: str = Field(default="upload", description="来源标识")
    chunk_size: int | None = Field(
        default=None, ge=100, le=8000, description="分块大小，覆盖默认值"
    )
    chunk_overlap: int | None = Field(
        default=None, ge=0, le=1000, description="分块重叠，覆盖默认值"
    )


class RagBinaryUploadRequest(BaseModel):
    """二进制文件上传（base64 编码），后端按扩展名解析。"""

    filename: str = Field(..., min_length=1, description="文件名（含扩展名）")
    content_base64: str = Field(..., min_length=1, description="文件内容的 base64 编码")
    user_id: str = Field(default="default", description="用户 ID")
    source: str = Field(default="upload", description="来源标识")
    chunk_size: int | None = Field(
        default=None, ge=100, le=8000, description="分块大小，覆盖默认值"
    )
    chunk_overlap: int | None = Field(
        default=None, ge=0, le=1000, description="分块重叠，覆盖默认值"
    )


def _get_knowledge_service(request: Request) -> KnowledgeService:
    """从 app.state 组装 RAG 服务；Milvus 或 Embedding 缺失时返回 503。"""
    milvus = getattr(request.app.state, "milvus", None)
    embedding = getattr(request.app.state, "embedding_model", None)
    reranker = getattr(request.app.state, "reranker", None)
    if milvus is None or embedding is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 不可用：Milvus 或 Embedding 模型未初始化",
        )
    return KnowledgeService(milvus, embedding, reranker)


@router.post("/documents", summary="文本方式入库文档")
async def add_document(body: RagDocumentCreate, request: Request):
    """接收纯文本，切分并向量化后写入知识库。"""
    service = _get_knowledge_service(request)
    chunks = await service.add_text(
        user_id=body.user_id,
        title=body.title,
        content=body.content,
        source=body.source,
        metadata=body.metadata,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    return {
        "document_id": chunks[0].document_id,
        "title": body.title,
        "chunk_count": len(chunks),
        "user_id": body.user_id,
    }


@router.post("/documents/upload", summary="文本文件内容入库（JSON）")
async def upload_document(body: RagFileUploadRequest, request: Request):
    """接收 JSON 文本内容入库；前端可读取本地文件后按此格式发送。"""
    service = _get_knowledge_service(request)
    chunks = await service.add_text(
        user_id=body.user_id,
        title=body.filename,
        content=body.content,
        source=body.source,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    return {
        "document_id": chunks[0].document_id,
        "title": body.filename,
        "chunk_count": len(chunks),
        "user_id": body.user_id,
    }


@router.post("/files", summary="二进制文件入库（自动解析 docx/txt/md）")
async def upload_binary_file(body: RagBinaryUploadRequest, request: Request):
    """接收 base64 编码的文件内容，按扩展名解析为纯文本后入库。"""
    service = _get_knowledge_service(request)
    try:
        raw = base64.b64decode(body.content_base64, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"base64 解码失败: {exc}") from exc

    try:
        text = extract_text_from_file(body.filename, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="文件内容为空或无法提取文本")

    chunks = await service.add_text(
        user_id=body.user_id,
        title=body.filename,
        content=text,
        source=body.source,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    return {
        "document_id": chunks[0].document_id,
        "title": body.filename,
        "chunk_count": len(chunks),
        "user_id": body.user_id,
    }


@router.get("/documents", summary="知识库文档列表")
async def list_documents(
    request: Request,
    user_id: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """按文档聚合返回知识库目录。"""
    service = _get_knowledge_service(request)
    return {
        "user_id": user_id,
        "documents": await service.list_documents(
            user_id=user_id, limit=limit, offset=offset
        ),
    }


@router.get("/search", summary="知识库语义检索")
async def search_documents(
    request: Request,
    q: str = Query(..., min_length=1, description="检索问题"),
    user_id: str = Query(default="default"),
    top_k: int = Query(default=5, ge=1, le=20),
    min_score: float | None = Query(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="最低相似度阈值，低于该分的片段会被过滤",
    ),
):
    """对知识库执行语义检索，返回相关片段与相似度。"""
    service = _get_knowledge_service(request)
    results = await service.search(
        user_id=user_id,
        query=q,
        top_k=top_k,
        min_score=min_score,
    )
    return {
        "query": q,
        "user_id": user_id,
        "results": [
            {
                "id": r.id,
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }


@router.post("/search", summary="知识库语义检索（POST 版）")
async def search_documents_post(body: RagSearchRequest, request: Request):
    """POST 版语义检索，便于传递较长查询。"""
    service = _get_knowledge_service(request)
    results = await service.search(
        user_id=body.user_id,
        query=body.query,
        top_k=body.top_k,
        min_score=body.min_score,
    )
    return {
        "query": body.query,
        "user_id": body.user_id,
        "results": [
            {
                "id": r.id,
                "score": r.score,
                "text": r.text,
                "metadata": r.metadata,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }


@router.delete("/documents/{document_id}", summary="删除知识库文档")
async def delete_document(
    document_id: str,
    request: Request,
    user_id: str = Query(default="default"),
):
    """删除指定文档的全部知识块。"""
    service = _get_knowledge_service(request)
    deleted = await service.delete_document(user_id=user_id, document_id=document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"document_id": document_id, "deleted": deleted}
