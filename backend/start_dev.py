"""开发启动脚本：启动 FastAPI 网关，关闭 uvicorn 默认 access log。"""

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
        # 关闭 uvicorn 自带的 access log，用我们自定义的中间件替代
        access_log=False,
    )
