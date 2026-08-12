import threading

from my_df.sandbox.local_sandbox import LocalSandbox
from my_df.sandbox.sandbox_provider import SandboxProvider

_singleton: SandboxProvider | None = None


class LocalSandboxProvider(SandboxProvider):
    def __init__(self, max_cached_threads: int = 256):
        """Initialize the local sandbox provider with static path mappings.

        Args:
            max_cached_threads: Upper bound on per-thread sandboxes retained in
                the LRU cache. When exceeded, the least-recently-used entry is
                evicted on the next ``acquire``.
        """
        self._path_mappings = []
        self._generic_sandbox: LocalSandbox | None = None
        self._thread_sandboxes: dict[str, LocalSandbox] = {}
        self._max_cached_threads = max_cached_threads
        self._lock = threading.Lock()

    def acquire(self, thread_id: str | None = None) -> str:
        """Return a sandbox id scoped to *thread_id* (or the generic singleton).

        - ``thread_id=None`` keeps the legacy singleton with id ``"local"`` for
          callers that have no thread context (e.g. legacy tests, scripts).
        - ``thread_id="abc"`` yields a per-thread ``LocalSandbox`` with id
          ``"local:abc"`` whose ``path_mappings`` resolve ``/mnt/user-data/...``
          to that thread's host directories.

        Thread-safe under concurrent invocation: the cache check + insert is
        guarded by ``self._lock`` so two callers racing on the same
        ``thread_id`` always observe the same LocalSandbox instance.
        """
        global _singleton

        return ""
