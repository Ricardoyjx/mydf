import importlib
import logging
import threading
from abc import ABC, abstractmethod

from my_df.config.app_config import get_app_config
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


# ── Provider 注册与发现（动态类解析）────────────────────────────────

# 内置注册表：别名 -> 完整类路径（与 models.factory 动态导入风格一致）
_BUILTIN_PROVIDERS: dict[str, str] = {
    "local": "my_df.sandbox.local_sandbox_provider.LocalSandboxProvider",
    # 预留：docker / provisioner 实现就绪后在此注册
    # "docker": "my_df.sandbox.docker_sandbox_provider.DockerSandboxProvider",
}

# 运行时自定义注册表（register_provider 写入），优先级高于内置
_CUSTOM_PROVIDERS: dict[str, str] = {}


def register_provider(alias: str, class_path: str) -> None:
    """注册自定义 Provider：alias -> 完整类路径（如 'my_pkg.xxx.MyProvider'）。"""
    if not alias or not class_path:
        raise ValueError("alias 与 class_path 不能为空")
    if "." not in class_path:
        raise ValueError(f"class_path 必须是完整类路径: {class_path!r}")
    _CUSTOM_PROVIDERS[alias] = class_path
    logger.info("已注册沙箱 Provider: %s -> %s", alias, class_path)


def _import_class(class_path: str) -> type:
    """从 'module.path.ClassName' 动态导入类。"""
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(f"非法类路径: {class_path!r}（需为 module.path.ClassName）")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"导入模块失败: {module_path}") from exc
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        raise ImportError(f"模块 {module_path} 中不存在类 {class_name}")
    return cls


def resolve_provider_class(sandbox_type: str) -> type:
    """按 sandbox_type 解析 Provider 类（注册与发现入口）。

    解析优先级：
        1. 运行时自定义注册表（register_provider）；
        2. 内置注册表（local 等）；
        3. 把 sandbox_type 本身当作完整类路径解析。
    """
    class_path = _CUSTOM_PROVIDERS.get(sandbox_type)
    if class_path is None:
        class_path = _BUILTIN_PROVIDERS.get(sandbox_type)
    if class_path is None:
        if "." in sandbox_type:
            class_path = sandbox_type
        else:
            raise ValueError(
                f"未知沙箱类型: {sandbox_type!r}"
                "（可 register_provider 注册或使用完整类路径）"
            )
    return _import_class(class_path)


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
    """按配置创建 Provider 实现（注册表 + 动态类解析）。"""
    app_config = get_app_config()
    sandbox_type = app_config.sandbox.use if app_config.sandbox else "local"
    # kwargs 可显式覆盖（测试/扩展用）
    if "sandbox_type" in kwargs:
        sandbox_type = kwargs.pop("sandbox_type")
    provider_cls = resolve_provider_class(sandbox_type)
    logger.info("沙箱 Provider 解析: %s -> %s", sandbox_type, provider_cls.__name__)
    return provider_cls(**kwargs)


def reset_sandbox_provider() -> None:
    """清除单例缓存（同时调用 Provider.reset 释放缓存实例）。

    下次 ``get_sandbox_provider()`` 会按最新配置重建。
    """
    global _default_sandbox_provider
    with _provider_lock:
        provider = _default_sandbox_provider
        _default_sandbox_provider = None
    if provider is not None:
        reset = getattr(provider, "reset", None)
        if reset is not None:
            reset()
    logger.info("沙箱环境提供者缓存已重置")


async def shutdown_sandbox_provider() -> None:
    """优雅关闭并清除单例（释放全部已获取的沙箱，兼容 async shutdown）。"""
    global _default_sandbox_provider
    with _provider_lock:
        provider = _default_sandbox_provider
        _default_sandbox_provider = None
    if provider is None:
        return
    shutdown = getattr(provider, "shutdown", None)
    if shutdown is not None:
        result = shutdown()
        if hasattr(result, "__await__"):
            await result
    logger.info("沙箱环境提供者已关闭并清除")
