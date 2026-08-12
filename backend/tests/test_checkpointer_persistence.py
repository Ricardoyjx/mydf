"""测试 SQLite Checkpointer 持久化。

验证步骤：
1. 设置环境变量启用 SQLite checkpointer
2. 创建 checkpointer 并写入一个 checkpoint
3. 重新创建 checkpointer 实例（模拟重启）
4. 验证之前写入的 checkpoint 仍然可读
"""

import os
import tempfile

# 必须在导入 app_config 之前设置环境变量
os.environ["MYDF_CHECKPOINTER_TYPE"] = "sqlite"
os.environ["MYDF_CHECKPOINTER_PATH"] = tempfile.mktemp(suffix=".db")

import asyncio

from my_df.config.app_config import get_app_config
from my_df.runtime.checkpointer.async_provider import make_checkpointer


async def test_sqlite_persistence():
    """写入 checkpoint → 重建连接 → 读取验证。"""
    db_path = os.environ["MYDF_CHECKPOINTER_PATH"]
    print(f"测试数据库: {db_path}")

    # ── 第一步：写入 checkpoint ──
    config = get_app_config()
    # 使用最新的 app_config（已经包含 checkpointer 配置）
    # 强制重新加载，确保使用我们的环境变量
    import importlib

    import my_df.config.app_config as cfg_mod

    importlib.reload(cfg_mod)
    config = cfg_mod.get_app_config()

    print(
        f"Checkpointer 配置: type={config.checkpointer.type}, path={config.checkpointer.connection_string}"
    )

    async with make_checkpointer(config) as cp:
        # 写入一个简单的 checkpoint
        test_config = {
            "configurable": {
                "thread_id": "test-thread-001",
                "checkpoint_ns": "",
            }
        }
        checkpoint = {
            "v": 1,
            "id": "test-cp-001",
            "ts": "2026-07-24T00:00:00Z",
            "channel_values": {
                "messages": [
                    {"role": "human", "content": "你好"},
                    {"role": "ai", "content": "你好！有什么可以帮你的？"},
                ]
            },
        }
        await cp.aput(test_config, checkpoint, {}, {})

        # 立即读取验证
        result = await cp.aget_tuple(test_config)
        assert result is not None, "第一次读取 checkpoint 失败"
        print("✅ 第一次写入并读取成功")

    # ── 第二步：重建连接（模拟服务重启） ──
    async with make_checkpointer(config) as cp2:
        result2 = await cp2.aget_tuple(test_config)
        assert result2 is not None, "重启后读取 checkpoint 失败"
        print("✅ 重启后读取成功 — 数据已持久化到磁盘")

    # ── 清理 ──
    os.unlink(db_path)
    print(f"✅ 测试全部通过，临时数据库已清理: {db_path}")


if __name__ == "__main__":
    asyncio.run(test_sqlite_persistence())
