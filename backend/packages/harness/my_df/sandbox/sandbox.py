from abc import ABC, abstractmethod


class Sandbox(ABC):
    """沙盒抽象基类"""

    _id: str

    def __init__(self, id: str):
        self._id = id

    @property
    def id(self) -> str:
        return self._id
        """沙盒标识"""

    @property
    @abstractmethod
    def type(self) -> str:
        """沙盒类型: local,docker,provisioner"""

    @abstractmethod
    def execute_command(self, command: str) -> str:
        """执行命令"""

    @abstractmethod
    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """读取文件"""

    @abstractmethod
    def write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> None:
        """写入文件"""

    @abstractmethod
    def list_dir(
        self,
        path: str,
        max_depth: int = 2,
    ) -> list[str]:
        """
        列出目录
        """
