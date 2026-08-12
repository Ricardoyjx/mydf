import logging
import threading
from abc import ABC, abstractmethod

from my_df.config.app_config import get_app_config
from my_df.sandbox.loacl_sanbox_provider import LocalSandboxProvider
from my_df.sandbox.sandbox import Sandbox


class SandboxProvider(ABC):
    @abstractmethod
    def acquire(self, thread_id: str | None = None) -> str:
        """Acquire a sandbox environment and return its ID.

        Returns:
            The ID of the acquired sandbox environment.
        """

    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox environment by ID.

        Args:
            sandbox_id: The ID of the sandbox environment to retain.
        """

    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        """Release a sandbox environment.

        Args:
            sandbox_id: The ID of the sandbox environment to destroy.
        """


_default_sandbox_provider: SandboxProvider | None = None
_provider_lock = threading.Lock()  # 并发安全：多请求同时首次 get 只创建一个

logger = logging.getLogger(__name__)


def get_sandbox_provider(**kwargs) -> SandboxProvider:
    """Get the sandbox provider singleton.

    Returns a cached singleton instance. Use `reset_sandbox_provider()` to clear
    the cache, or `shutdown_sandbox_provider()` to properly shutdown and clear.
    """
    global _default_sandbox_provider
    if _default_sandbox_provider is None:
        with _provider_lock:
            if _default_sandbox_provider is None:
                _default_sandbox_provider = _create_sandbox_provider(**kwargs)
                logger.info(
                    "已创建沙箱环境提供者 %s",
                    type(_default_sandbox_provider).__name__,
                )
    return _default_sandbox_provider


def _create_sandbox_provider(**kwargs) -> SandboxProvider:
    """按配置创建 Provider 实现（沙箱后端：local/docker/provisioner）。"""
    # 从配置读取沙箱类型，当前只有 local 实现
    app_config = get_app_config()
    sandbox_type = getattr(app_config, "sandbox_type", "local")
    if sandbox_type == "local":
        pass
