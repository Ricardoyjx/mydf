"""一次性迁移脚本：把文件系统记忆（memory.json）迁移到 LangGraph Store。

用法（在 backend 目录下）：
    .venv/bin/python scripts/migrate_memory_to_store.py
    .venv/bin/python scripts/migrate_memory_to_store.py --delete-source

默认只迁移用户隔离布局下的文件：
- ``{base_dir}/users/{user_id}/agents/{agent_name}/memory.json``
- ``{base_dir}/users/{user_id}/memory.json``（用户级，key="default"）

迁移完成后源文件保留，加 ``--delete-source`` 才删除。
Legacy 布局（``{base_dir}/memory.json``、``{base_dir}/agents/*``）不含用户
隔离信息，不迁移，仅打印提示。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _collect_user_memory_files(base_dir: Path) -> list[tuple[str, str | None, Path]]:
    """收集用户隔离布局下的记忆文件。

    返回 [(user_id, agent_name, file_path)]；agent_name 为 None 表示用户级记忆。
    """
    files: list[tuple[str, str | None, Path]] = []
    users_dir = base_dir / "users"
    if not users_dir.is_dir():
        return files

    for user_dir in sorted(users_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        user_memory = user_dir / "memory.json"
        if user_memory.is_file():
            files.append((user_id, None, user_memory))
        agents_dir = user_dir / "agents"
        if agents_dir.is_dir():
            for agent_dir in sorted(agents_dir.iterdir()):
                if not agent_dir.is_dir():
                    continue
                memory_file = agent_dir / "memory.json"
                if memory_file.is_file():
                    files.append((user_id, agent_dir.name, memory_file))
    return files


def _memory_key(agent_name: str | None) -> str:
    """与 MemoryMiddleware 相同的 key 归一化规则。"""
    return (agent_name or "").replace("_", "-") or "default"


def _load_memory_file(file_path: Path) -> dict:
    """同步读取 memory.json（在线程池中执行，避免阻塞事件循环）。"""
    with open(file_path, encoding="utf-8") as fh:
        return json.load(fh)


async def _migrate(
    store,
    files: list[tuple[str, str | None, Path]],
    *,
    delete_source: bool,
) -> int:
    migrated = 0
    for user_id, agent_name, file_path in files:
        try:
            data = await asyncio.to_thread(_load_memory_file, file_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取失败，跳过 %s: %s", file_path, exc)
            continue

        key = _memory_key(agent_name)
        await store.aput(("user", user_id), key, data)
        migrated += 1
        logger.info("已迁移: user=%s key=%s <- %s", user_id, key, file_path)
        if delete_source:
            await asyncio.to_thread(file_path.unlink, missing_ok=True)
            logger.info("已删除源文件: %s", file_path)

    return migrated


async def run(*, delete_source: bool) -> int:
    from my_df.config.app_config import get_app_config
    from my_df.config.path import get_paths
    from my_df.runtime.store.async_provider import make_store

    env_path = _BACKEND_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    base_dir = get_paths().base_dir
    files = _collect_user_memory_files(base_dir)
    if not files:
        logger.info("未找到可迁移的记忆文件（base_dir=%s）", base_dir)
        return 0

    logger.info("发现 %d 个记忆文件，开始迁移...", len(files))
    async with make_store(get_app_config()) as store:
        migrated = await _migrate(store, files, delete_source=delete_source)
    logger.info("迁移完成: %d/%d 个文件写入 Store", migrated, len(files))

    legacy = [
        str(path)
        for path in (base_dir / "memory.json", base_dir / "agents")
        if path.exists()
    ]
    if legacy:
        logger.warning(
            "检测到 legacy 布局（%s），不含用户隔离信息，已跳过；如需处理请手动确认。",
            ", ".join(legacy),
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="迁移 memory.json 到 LangGraph Store")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="迁移成功后删除源 memory.json 文件（默认保留）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s | %(message)s",
    )
    asyncio.run(run(delete_source=args.delete_source))


if __name__ == "__main__":
    main()
