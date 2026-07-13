"""运行状态与断开模式枚举。"""

from enum import StrEnum


class RunStatus(StrEnum):
    """单次运行的生命周期状态。"""

    pending = "pending"          # 等待中
    running = "running"          # 运行中
    success = "success"          # 成功完成
    error = "error"              # 出错终止
    timeout = "timeout"          # 超时
    interrupted = "interrupted"  # 被中断


class DisconnectMode(StrEnum):
    """SSE 消费者断开连接时的行为。"""

    cancel = "cancel"            # 取消运行
    continue_ = "continue"       # 继续执行
