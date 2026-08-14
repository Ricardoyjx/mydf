"""MemoryMiddleware 单元测试。

覆盖：
- 工具函数：_has_content / _format_memory_block / _extract_conversation_summary
- abefore_model：空 memory 跳过 / 有 memory 注入 / 无 HumanMessage 跳过
- aafter_model：对话回写 / 无消息跳过
- 完整集成：给定有内容的 storage → abefore_model 注入 → aafter_model 更新
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain.messages import AIMessage, HumanMessage
from my_df.agents.memory.storage import create_empty_memory, utc_now_iso_z
from my_df.agents.middlewares.memory_middleware import (
    MemoryMiddleware,
    _extract_conversation_summary,
    _format_memory_block,
    _has_content,
)

# =============================================================
# 工具函数测试
# =============================================================


class TestHelpers:
    def test_has_content_empty(self):
        """空 memory → False"""
        assert not _has_content(create_empty_memory())

    def test_has_content_with_facts(self):
        """facts 有值 → True"""
        mem = create_empty_memory()
        mem["facts"] = [{"id": "f1", "content": "用户喜欢 Python"}]
        assert _has_content(mem)

    def test_has_content_with_user_summary(self):
        """user.workContext.summary 有值 → True"""
        mem = create_empty_memory()
        mem["user"]["workContext"]["summary"] = "Working on my-df"
        assert _has_content(mem)

    def test_has_content_with_history_summary(self):
        """history 字段有值 → True"""
        mem = create_empty_memory()
        mem["history"]["recentMonths"]["summary"] = "Recent dev work"
        assert _has_content(mem)

    def test_has_content_with_personal_context(self):
        """personalContext 原本不在空结构中，应该也能正确处理"""
        mem = create_empty_memory()
        mem["user"]["personalContext"] = {"summary": "Likes coffee", "updatedAt": ""}
        assert _has_content(mem)

    def test_format_memory_block_full(self):
        """完整 memory → 正确的 XML 格式"""
        mem = create_empty_memory()
        mem["user"]["workContext"] = {"summary": "项目开发", "updatedAt": ""}
        mem["user"]["personalContext"] = {"summary": "喜欢简洁回答", "updatedAt": ""}
        mem["user"]["topOfMind"] = {"summary": "实现 MemoryMiddleware", "updatedAt": ""}
        mem["history"]["recentMonths"] = {"summary": "最近在写 Python", "updatedAt": ""}
        mem["facts"] = [{"id": "f1", "content": "用户是开发者"}]

        result = _format_memory_block(mem)

        assert "<memory_context>" in result
        assert "</memory_context>" in result
        assert "<work_context>项目开发</work_context>" in result
        assert "<personal_context>喜欢简洁回答</personal_context>" in result
        assert "<top_of_mind>实现 MemoryMiddleware</top_of_mind>" in result
        assert "<recent_history>最近在写 Python</recent_history>" in result
        assert "<fact>用户是开发者</fact>" in result

    def test_format_memory_block_empty(self):
        """空 memory → 只有根标签"""
        result = _format_memory_block(create_empty_memory())
        assert result == "<memory_context>\n</memory_context>"

    def test_extract_conversation_summary(self):
        """提取最后 N 轮对话摘要"""
        messages = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮助你的吗？"),
            HumanMessage(content="帮我写个 Python 计算器"),
            AIMessage(content="好的，以下是代码：\n\ndef calculator():\n    ..."),
        ]
        result = _extract_conversation_summary(messages)
        assert "[human] 帮我写个 Python 计算器" in result
        assert "[ai] 好的，以下是代码：" in result
        assert "[human] 你好" in result

    def test_extract_conversation_summary_empty(self):
        """空消息列表 → 空字符串"""
        assert _extract_conversation_summary([]) == ""


# =============================================================
# MemoryMiddleware 单元测试
# =============================================================


class TestMemoryMiddlewareUnit:
    """使用 MagicMock 隔离 storage，不写真实文件。"""

    def _make_middleware(self, memory_data=None):
        """创建中间件实例，替换 storage 为 mock。"""
        mw = MemoryMiddleware(agent_name="test_agent", user_id="test_user")
        mw._storage = MagicMock()
        if memory_data is not None:
            mw._storage.load.return_value = memory_data
        else:
            mw._storage.load.return_value = create_empty_memory()
        return mw

    # ── abefore_model ──

    def test_before_model_empty_memory_skips(self):
        """空 memory → abefore_model 返回 None，不注入"""
        mw = self._make_middleware(create_empty_memory())
        state = {"messages": [HumanMessage(content="hello")]}
        result = asyncio.run(mw.abefore_model(state, MagicMock()))
        assert result is None
        # 原始消息不该被修改
        assert state["messages"][0].content == "hello"

    def test_before_model_injects_memory(self):
        """有内容的 memory → 注入到首条 HumanMessage 前"""
        memory = create_empty_memory()
        memory["user"]["workContext"] = {"summary": "用户是开发者", "updatedAt": ""}

        mw = self._make_middleware(memory)
        state = {"messages": [HumanMessage(content="帮我写代码")]}
        result = asyncio.run(mw.abefore_model(state, MagicMock()))

        assert result is not None
        messages = result["messages"]
        content = messages[0].content
        # 原始内容保留
        assert "帮我写代码" in content
        # memory 上下文已注入
        assert "用户是开发者" in content
        assert "<memory_context>" in content

    def test_before_model_no_human_message_skips(self):
        """没有 HumanMessage → abefore_model 返回 None"""
        mw = self._make_middleware()
        mw._storage.load.return_value = {
            "version": "1.0",
            "user": {"workContext": {"summary": "test", "updatedAt": ""}},
            "history": {},
            "facts": [],
            "lastUpdated": "",
        }
        state = {"messages": [AIMessage(content="I am AI")]}
        result = asyncio.run(mw.abefore_model(state, MagicMock()))
        assert result is None

    def test_before_model_empty_messages_skips(self):
        """空消息列表 → abefore_model 返回 None"""
        mw = self._make_middleware()
        result = asyncio.run(mw.abefore_model({"messages": []}, MagicMock()))
        assert result is None

    def test_before_model_storage_error_graceful(self):
        """storage.load 抛异常 → 优雅降级，返回 None"""
        mw = self._make_middleware()
        mw._storage.load.side_effect = Exception("磁盘错误")
        state = {"messages": [HumanMessage(content="hello")]}
        result = asyncio.run(mw.abefore_model(state, MagicMock()))
        assert result is None  # 不崩溃

    # ── aafter_model ──

    def test_after_model_saves_conversation(self):
        """有消息 → aafter_model 调用 storage.save()"""
        mw = self._make_middleware()
        state = {
            "messages": [
                HumanMessage(content="第一轮提问"),
                AIMessage(content="第一轮回答"),
            ]
        }
        result = asyncio.run(mw.aafter_model(state, MagicMock()))

        assert result is None  # aafter_model 不返回 state 更新
        mw._storage.save.assert_called_once()
        saved, kwargs = mw._storage.save.call_args
        saved_data = saved[0]
        assert "第一轮提问" in saved_data["history"]["recentMonths"]["summary"]

    def test_after_model_empty_messages_skips(self):
        """空消息 → 不调用 save()"""
        mw = self._make_middleware()
        result = asyncio.run(mw.aafter_model({"messages": []}, MagicMock()))
        assert result is None
        mw._storage.save.assert_not_called()

    def test_after_model_storage_error_graceful(self):
        """storage.save 抛异常 → 优雅降级，不崩溃"""
        mw = self._make_middleware()
        mw._storage.save.side_effect = Exception("写入失败")
        state = {"messages": [HumanMessage(content="hi")]}
        result = asyncio.run(mw.aafter_model(state, MagicMock()))
        assert result is None  # 不崩溃


# =============================================================
# 集成测试：使用临时文件
# =============================================================


class TestMemoryMiddlewareIntegration:
    """使用真实的 FileMemoryStorage + 临时目录测试完整流程。"""

    def test_full_flow_with_tmp_storage(self, tmp_path):
        """完整流程：写 memory → abefore_model 读到 → aafter_model 更新"""
        from my_df.agents.middlewares.memory_middleware import MemoryMiddleware as MW
        from my_df.agents.memory.storage import FileMemoryStorage
        from my_df.config.paths import get_paths

        storage = FileMemoryStorage()
        user_id = "intg_test_user"

        # 直接构造一个有内容的 memory dict
        memory = {
            "version": "1.0",
            "lastUpdated": utc_now_iso_z(),
            "user": {
                "workContext": {
                    "summary": "用户在做集成测试",
                    "updatedAt": utc_now_iso_z(),
                }
            },
            "history": {},
            "facts": [{"id": "f1", "content": "测试很重要"}],
        }
        storage.save(memory, agent_name="intg-agent", user_id=user_id)

        # 创建中间件，注入真实 storage
        mw = MW(agent_name="intg-agent", user_id=user_id)
        mw._storage = storage

        # abefore_model → 应能读到刚才写入的内容
        state = {"messages": [HumanMessage(content="你好")]}
        result = asyncio.run(mw.abefore_model(state, MagicMock()))

        assert result is not None
        injected = result["messages"][0].content
        assert "用户在做集成测试" in injected
        assert "测试很重要" in injected
        assert "你好" in injected  # 原始内容保留

        # aafter_model → 应把新对话写回去
        state2 = {
            "messages": [
                HumanMessage(content="新的一轮对话"),
                AIMessage(content="这是回答"),
            ]
        }
        mw2 = MW(agent_name="intg-agent", user_id=user_id)
        mw2._storage = storage
        asyncio.run(mw2.aafter_model(state2, MagicMock()))

        # 验证写回成功
        loaded = storage.load(agent_name="intg-agent", user_id=user_id)
        assert loaded["history"]["recentMonths"]["summary"] != ""
        assert "新的一轮对话" in loaded["history"]["recentMonths"]["summary"]
