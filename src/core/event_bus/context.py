"""
事件总线处理器执行上下文
Handler Execution Context for Event Bus

独立定义HandlerExecutionContext，避免循环导入
"""

import time
from dataclasses import dataclass
from typing import Optional, Any

# 注意：这里不导入Event，避免循环依赖
# Event在运行时通过类型提示或延迟导入获取


@dataclass
class HandlerExecutionContext:
    """处理器执行上下文"""
    event: Any  # Event类型，但使用Any避免循环导入
    handler_info: Any
    start_time: float
    timeout: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """检查是否已过期"""
        if self.timeout is None:
            return False
        return time.time() - self.start_time > self.timeout

__all__ = ['HandlerExecutionContext']

