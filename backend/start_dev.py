"""开发启动脚本：启动 FastAPI 网关，关闭 uvicorn 默认 access log。

先加载 .env，确保子进程能继承环境变量，避免 checkpointer 等配置在
get_app_config() 首次调用时因 .env 未加载而丢失。
"""

import os
from pathlib import Path

# 屏蔽 tqdm 进度条输出（模型权重加载），子进程继承
os.environ.setdefault("TQDM_DISABLE", "1")

# 在导入任何业务模块之前加载 .env，使环境变量对子进程可见
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=False)
    # 检查关键配置是否已加载
    cp_type = os.getenv("MYDF_CHECKPOINTER_TYPE")
    if cp_type:
        print(f"[start_dev] 已加载环境变量: MYDF_CHECKPOINTER_TYPE={cp_type}")

import uvicorn
from app.gateway.config import get_gateway_config

if __name__ == "__main__":
    config = get_gateway_config()
    uvicorn.run(
        "app.gateway.app:app",
        host=config.host,
        port=config.port,
        reload=True,
        log_level="info",
        access_log=False,
    )
