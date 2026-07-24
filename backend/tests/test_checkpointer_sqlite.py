"""测试 SQLite Checkpointer 持久化。

运行方式：
    cd backend && .venv/bin/python3 -m pytest tests/test_checkpointer_sqlite.py -v
"""

import os
import tempfile


def test_sqlite_checkpointer_persistence():
    """写入 checkpoint → 重建连接 → 验证数据仍可读取。"""
    # 用临时文件模拟 SQLite 数据库
    db_path = tempfile.mktemp(suffix=".db")

    from langgraph.checkpoint.sqlite import SqliteSaver

    # ── 第一步：写入 ──
    with SqliteSaver.from_conn_string(db_path) as saver:
        saver.setup()
        config = {"configurable": {"thread_id": "test-thread-001", "checkpoint_ns": ""}}
        saver.put(
            config,
            {"v": 1, "id": "test-cp-001", "ts": "2026-07-24T00:00:00Z", "channel_values": {"messages": ["hello"]}},
            {"source": "test", "step": 1, "writes": {}},
            {},
        )
        result = saver.get_tuple(config)
        assert result is not None, "第一次写入后读取失败"
        print("✅ 写入并读取成功")

    # ── 第二步：重建连接（模拟重启） ──
    with SqliteSaver.from_conn_string(db_path) as saver2:
        result2 = saver2.get_tuple({"configurable": {"thread_id": "test-thread-001", "checkpoint_ns": ""}})
        assert result2 is not None, "重启后读取失败 — 数据未持久化"
        print("✅ 重启后读取成功 — 数据已持久化到磁盘")

    # 清理
    os.unlink(db_path)
    print("✅ 测试全部通过")


def test_checkpointer_via_app_config():
    """测试 AppConfig 能正确从环境变量加载 checkpointer 配置。"""
    os.environ["MYDF_CHECKPOINTER_TYPE"] = "sqlite"
    os.environ["MYDF_CHECKPOINTER_PATH"] = tempfile.mktemp(suffix=".db")

    from my_df.config.app_config import get_app_config

    config = get_app_config()
    assert config.checkpointer is not None, "checkpointer 配置不应为空"
    assert config.checkpointer.type == "sqlite", f"期望 sqlite, 得到 {config.checkpointer.type}"
    assert config.checkpointer.connection_string is not None, "connection_string 不应为空"
    print(f"✅ AppConfig 加载 checkpointer 成功: type={config.checkpointer.type}")
