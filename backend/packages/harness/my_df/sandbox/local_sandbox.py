from dataclasses import dataclass

from my_df.sandbox.sandbox import Sandbox


@dataclass(frozen=True)
class PathMapping:
    """A path mapping from a container path to a local path with optional read-only flag."""

    container_path: str
    local_path: str
    read_only: bool = False


class LocalSandbox(Sandbox):
    def __init__(self, id: str, path_mappings: list[PathMapping] | None = None):
        """
        Initialize local sandbox with optional path mappings.

        Args:
            id: Sandbox identifier
            path_mappings: List of path mappings with optional read-only flag.
                          Skills directory is read-only by default.
        """
        super().__init__(id)
        self.path_mapping = path_mappings or []
