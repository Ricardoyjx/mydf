import errno
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from my_df.sandbox.sandbox import Sandbox


@dataclass(frozen=True)
class PathMapping:
    """A path mapping from a container path to a local path with optional read-only flag."""

    container_path: str
    local_path: str
    read_only: bool = False


class ResolvedPath(NamedTuple):
    path: str
    mapping: PathMapping | None


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
        self.path_mappings = path_mappings or []

    def _find_path_mapping(self, path: str) -> tuple[PathMapping, str] | None:
        path_str = str(path)

        for mapping in sorted(
            self.path_mappings,
            key=lambda m: len(m.container_path.rstrip("/") or "/"),
            reverse=True,
        ):
            container_path = mapping.container_path.rstrip("/") or "/"
            if container_path == "/":
                if path_str.startswith("/"):
                    return mapping, path_str.lstrip("/")
                continue

            if path_str == container_path or path_str.startswith(container_path + "/"):
                relative = path_str[len(container_path) :].lstrip("/")
                return mapping, relative

        return None

    def _resolve_path_with_mapping(self, path: str) -> ResolvedPath:
        """
        Resolve container path to actual local path using mappings.

        Args:
            path: Path that might be a container path

        Returns:
            Resolved local path and the matched mapping, if any
        """
        path_str = str(path)

        mapping_match = self._find_path_mapping(path_str)
        if mapping_match is None:
            return ResolvedPath(path_str, None)

        mapping, relative = mapping_match
        local_root = Path(mapping.local_path).resolve()
        resolved_path = (local_root / relative).resolve() if relative else local_root

        try:
            resolved_path.relative_to(local_root)
        except ValueError as exc:
            raise PermissionError(
                errno.EACCES, "Access denied: path escapes mounted directory", path_str
            ) from exc

        return ResolvedPath(str(resolved_path), mapping)

    def _resolve_path(self, path: str) -> str:
        return self._resolve_path_with_mapping(path).path

    def _resolve_paths_in_command(self, command: str) -> str:
        """
        Resolve container paths to local paths in a command string.

        Args:
            command: Command string that may contain container paths

        Returns:
            Command with container paths resolved to local paths
        """

        import re

        # Sort mappings by length (longest first) for correct prefix matching
        sorted_mappings = sorted(
            self.path_mappings, key=lambda m: len(m.container_path), reverse=True
        )

        # Build regex pattern to match all container paths
        # Match container path followed by optional path components
        if not sorted_mappings:
            return command

        # Create pattern that matches any of the container paths.
        # The lookahead (?=/|$|...) ensures we only match at a path-segment boundary,
        # preventing /mnt/skills from matching inside /mnt/skills-extra.
        patterns = [
            re.escape(m.container_path)
            + r"(?=/|$|[\s\"';&|<>()])(?:/[^\s\"';&|<>()]*)?"
            for m in sorted_mappings
        ]
        pattern = re.compile("|".join(f"({p})" for p in patterns))

        def replace_match(match: re.Match) -> str:
            matched_path = match.group(0)
            return self._resolve_path(matched_path)

        return pattern.sub(replace_match, command)

    @staticmethod
    def _find_first_available_shell(candidates: tuple[str, ...]) -> str | None:
        """Return the first executable shell path or command found from candidates."""
        for shell in candidates:
            if os.path.isabs(shell):
                if os.path.isfile(shell) and os.access(shell, os.X_OK):
                    return shell
                continue

            shell_from_path = shutil.which(shell)
            if shell_from_path is not None:
                return shell_from_path

        return None

    @staticmethod
    def _get_shell() -> str:
        """Detect available shell executable with fallback."""
        shell = LocalSandbox._find_first_available_shell(
            ("/bin/zsh", "/bin/bash", "/bin/sh", "sh")
        )
        if shell is not None:
            return shell
        raise RuntimeError(
            "No suitable shell executable found. Tried /bin/zsh, /bin/bash, /bin/sh, and `sh` on PATH."
        )

    def _reverse_resolve_path(self, path: str) -> str:
        """
        Reverse resolve local path back to container path using mappings.

        Args:
            path: Local path that might need to be mapped to container path

        Returns:
            Container path if mapping exists, otherwise original path
        """
        normalized_path = path.replace("\\", "/")
        path_str = str(Path(normalized_path).resolve())

        # Try each mapping (longest local path first for more specific matches)
        for mapping in sorted(
            self.path_mappings, key=lambda m: len(m.local_path), reverse=True
        ):
            local_path_resolved = str(Path(mapping.local_path).resolve())
            if path_str == local_path_resolved or path_str.startswith(
                local_path_resolved + "/"
            ):
                # Replace the local path prefix with container path
                relative = path_str[len(local_path_resolved) :].lstrip("/")
                resolved = (
                    f"{mapping.container_path}/{relative}"
                    if relative
                    else mapping.container_path
                )
                return resolved

        # No mapping found, return original path
        return path_str

    def type(self) -> str:
        return "local"

    def _reverse_resolve_paths_in_output(self, output: str) -> str:
        """
        Reverse resolve local paths back to container paths in output string.

        Args:
            output: Output string that may contain local paths

        Returns:
            Output with local paths resolved to container paths
        """
        import re

        # Sort mappings by local path length (longest first) for correct prefix matching
        sorted_mappings = sorted(
            self.path_mappings, key=lambda m: len(m.local_path), reverse=True
        )

        if not sorted_mappings:
            return output

        # Create pattern that matches absolute paths
        # Match paths like /Users/... or other absolute paths
        result = output
        for mapping in sorted_mappings:
            # Escape the local path for use in regex
            escaped_local = re.escape(str(Path(mapping.local_path).resolve()))
            # Match the local path followed by optional path components with either separator
            pattern = re.compile(escaped_local + r"(?:[/\\][^\s\"';&|<>()]*)?")

            def replace_match(match: re.Match) -> str:
                matched_path = match.group(0)
                return self._reverse_resolve_path(matched_path)

            result = pattern.sub(replace_match, result)

        return result

    def execute_command(self, command: str) -> str:
        # Resolve container paths in command before execution
        resolved_command = self._resolve_paths_in_command(command)
        shell = self._get_shell()

        args = [shell, "-c", resolved_command]
        result = subprocess.run(  # noqa: PLW1510
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=600,
        )

        output = result.stdout
        if result.stderr:
            output += f"\nStd Error:\n{result.stderr}" if output else result.stderr
        if result.returncode != 0:
            output += f"\nExit Code: {result.returncode}"

        final_output = output if output else "(no output)"
        # Reverse resolve local paths back to container paths in output
        return self._reverse_resolve_paths_in_output(final_output)

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """读取文件内容，支持行范围（1-based，含端点）。

        参数：
            path:       容器路径（经 path_mappings 映射到本地）。
            start_line: 起始行号（含），从 1 开始。
            end_line:   结束行号（含）；缺省读到文件末尾。
        """
        resolved = self._resolve_path(path)
        local_path = Path(resolved)

        if not local_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not local_path.is_file():
            raise IsADirectoryError(f"不是文件: {path}")

        with open(local_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if start_line is not None or end_line is not None:
            start = (start_line or 1) - 1
            end = end_line if end_line is not None else len(lines)
            start = max(0, start)
            end = min(len(lines), end)
            lines = lines[start:end] if start < end else []

        return "".join(lines)

    def write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> None:
        """写入文件；append=True 追加，否则覆盖。

        参数：
            path:    容器路径（经 path_mappings 映射到本地）。
            content: 要写入的内容。
            append:  True 追加到文件末尾，False 覆盖整个文件。
        """
        resolved = self._resolve_path_with_mapping(path)
        if resolved.mapping is not None and resolved.mapping.read_only:
            raise PermissionError(
                errno.EACCES, "Access denied: path is read-only", path
            )

        local_path = Path(resolved.path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(local_path, mode, encoding="utf-8") as f:
            f.write(content)

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        """递归列出目录内容，最多 max_depth 层，返回相对路径（排序）。

        参数：
            path:      容器路径（经 path_mappings 映射到本地）。
            max_depth: 递归深度（0 表示仅列出当前目录直接子项）。
        """
        resolved = self._resolve_path(path)
        root = Path(resolved)

        if not root.exists():
            raise FileNotFoundError(f"目录不存在: {path}")
        if not root.is_dir():
            raise NotADirectoryError(f"不是目录: {path}")

        max_depth = max(0, max_depth)
        root_depth = len(root.parts)
        results: list[str] = []

        for current, dirs, files in os.walk(root):
            current_depth = len(Path(current).parts) - root_depth
            # 达到深度上限时不再进入子目录
            if current_depth >= max_depth:
                dirs[:] = []
                if current_depth > max_depth:
                    continue

            rel = Path(current).relative_to(root)
            for name in sorted(dirs) + sorted(files):
                results.append(str(rel / name) if str(rel) != "." else name)

        return results
