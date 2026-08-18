"""FastAPI 网关入口：应用生命周期管理 + 路由注册。"""

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import ClassVar

from app.gateway.config import get_gateway_config
from app.gateway.deps import langgraph_runtime
from app.gateway.routers.memory import router as memory_router
from app.gateway.routers.rag import router as rag_router
from app.gateway.routers.runs import router as runs_router
from app.gateway.routers.threads import router as threads_router
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from my_df.agents.supervisor_graph import build_supervisor_graph
from my_df.agents.tools.weather import search_weather
from my_df.config.app_config import get_app_config
from my_df.rag.parent_doc import make_parent_docstore
from my_df.rag.service import KnowledgeService
from my_df.runtime.embeddings.sentence import SentenceEmbeddings
from my_df.runtime.milvus.async_provider import make_milvus_storage
from my_df.runtime.reranker.sentence import SentenceRerank

_BACKEND_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用程序生命周期处理器。"""

    # 0. 配置日志级别 + 彩色输出
    log_level = (os.getenv("MYDF_LOG_LEVEL") or "info").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    if not logging.getLogger().hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(_ColoredFormatter())
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    # 第三方库噪音日志降级：httpx / huggingface_hub 等仅保留 WARNING 以上
    for _noisy_logger in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "filelock",
        "urllib3",
        "transformers",
        "sentence_transformers",
    ):
        logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

    # HF 未认证请求等常规提示一并屏蔽（真实错误仍以异常形式暴露）
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

    # 禁用 tqdm 进度条输出（模型权重加载时的 Loading weights: ...）
    os.environ.setdefault("TQDM_DISABLE", "1")

    # 1. 加载 .env 文件，使 DEEPSEEK_API_KEY 等环境变量可用
    env_path = _BACKEND_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logger.info("已加载环境变量: %s", env_path)

    # 2. 加载应用配置（含模型、日志级别等）
    try:
        startup_config = get_app_config()
        logger.info("应用配置加载成功")
    except Exception as e:
        error_msg = f"网关启动时加载配置失败: {e}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    config = get_gateway_config()
    logger.info("API 网关启动于 %s:%s", config.host, config.port)

    # 3. 初始化 Milvus 向量存储（可选，失败不影响应用启动）
    _milvus_stack = AsyncExitStack()
    try:
        milvus = await _milvus_stack.enter_async_context(
            make_milvus_storage(startup_config)
        )
        await milvus.ensure_collection("default")
        app.state.milvus = milvus
        logger.info("Milvus 向量存储已就绪")
    except Exception:  # noqa: BLE001
        logger.warning("Milvus 未就绪（向量存储不可用，其他功能正常）")
        app.state.milvus = None

    # 4. 注册 Embedding 模型（懒加载：启动不占内存，首次检索/入库时加载）
    app.state.embedding_model = SentenceEmbeddings(
        model_name=startup_config.embedding.model
    )
    logger.info(
        "Embedding 已注册（懒加载，首次检索时加载）: model=%s",
        startup_config.embedding.model,
    )

    # 5. 注册 reranker 模型（懒加载：启动不占内存，首次搜索时加载）
    if startup_config.reranker.enable:
        app.state.reranker = SentenceRerank(model_name=startup_config.reranker.model)
        logger.info(
            "Reranker 已注册（懒加载，首次搜索时加载）: model=%s",
            app.state.reranker._model_name,
        )
    else:
        logger.info("Reranker 未启用（MYDF_RERANK_ENABLED=true 可开启精排）")
        app.state.reranker = None

    # 5.5 small_to_big 父块 docstore + 共享 knowledgeService
    try:
        rag_docstore = await make_parent_docstore(startup_config.checkpointer)
        app.state.rag_docstore = rag_docstore
        app.state.knowledge_service = KnowledgeService(
            milvus=milvus,
            embedding=app.state.embedding_model,
            reranker=app.state.reranker,
            small_to_big=os.getenv("MYDF_RAG_SMALL_TO_BIG") == "true",
            docstore=rag_docstore,
            rrf_enabled=os.getenv("MYDF_RAG_RRF_ENABLED") == "true",
        )
        logger.info(
            "RAG KnowledgeService 已就绪: small_to_big=%s, rrf=%s",
            app.state.knowledge_service.small_to_big_enabled,
            app.state.knowledge_service.rrf_enabled,
        )
    except Exception:
        logger.exception("RAG docstore 初始化失败，small_to_big 不可用")
        app.state.rag_docstore = None
        app.state.knowledge_service = None

    # 6. 初始化 LangGraph 运行时（stream bridge、checkpointer 等）
    async with langgraph_runtime(app, startup_config):
        logger.info("LangGraph 运行时初始化完成")

        # 7. 预热 Multi-Agent Supervisor 编排图（启动构建一次，请求时复用）
        try:
            app.state.agent_factory = build_supervisor_graph(
                startup_config,
                store=app.state.store,
                milvus=app.state.milvus,
                embedding_model=app.state.embedding_model,
                tools=[search_weather],
                event_store=app.state.event_store,
            )
            logger.info("Multi-Agent Supervisor 图已预热，请求时将复用该实例")
        except Exception:
            logger.exception("Supervisor 图预热失败，请求时将按需构建")
            app.state.agent_factory = None

        yield

    # 清理 Milvus 连接（在 langgraph_runtime 关闭后）
    await _milvus_stack.aclose()

    rag_docstore = getattr(app.state, "rag_docstore", None)
    if rag_docstore is not None and hasattr(rag_docstore, "aclose"):
        await rag_docstore.aclose()

    # 释放本地模型（embedding / reranker），避免 reload 旧进程残留
    embedder = getattr(app.state, "embedding_model", None)
    if embedder is not None:
        await embedder.close()
    reranker = getattr(app.state, "reranker", None)
    if reranker is not None:
        await reranker.close()
    logger.info("API 网关关闭中")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    app = FastAPI(title="my-df Gateway", version="0.1.0", lifespan=lifespan)

    # API 路由
    app.include_router(runs_router)
    app.include_router(memory_router)
    app.include_router(threads_router)
    app.include_router(rag_router)

    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 前端页面（SPA 单页入口）
    frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"

    @app.get("/")
    def get_root():
        return FileResponse(frontend_dir / "index.html")

    @app.get("/health", tags=["health"])
    def get_health():
        return {"status": "healthy", "service": "my-df-gateway"}

    # ── 请求日志中间件 ──
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """记录每个 HTTP 请求的方法、路径、状态码和处理耗时。"""
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        query = request.url.query
        full_path = f"{path}?{query}" if query else path

        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "请求异常 | %s %s | %.0fms",
                method,
                full_path,
                elapsed * 1000,
                # exc_info=True,
            )
            raise

        elapsed = time.perf_counter() - start
        # SSE 流式请求降级到 DEBUG，避免刷屏
        content_type = response.headers.get("content-type", "")
        log_fn = logger.debug if "text/event-stream" in content_type else logger.info
        log_fn(
            "%s %s | %s | %.0fms",
            method,
            full_path,
            response.status_code,
            elapsed * 1000,
        )
        return response

    return app


class _ColoredFormatter(logging.Formatter):
    """带 ANSI 颜色的日志格式化器。"""

    _LEVEL_COLORS: ClassVar[dict] = {
        "DEBUG": "\033[36m",  # 青色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[41m",  # 红底
    }
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        level_color = self._LEVEL_COLORS.get(record.levelname, "")
        level = f"{level_color}{record.levelname}{self._RESET}"
        name = f"{self._DIM}{record.name}{self._RESET}"
        return f"{level} {name} | {record.getMessage()}"


app = create_app()
