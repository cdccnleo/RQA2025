
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
"""
基础设施层 - 高级日志类型定义

定义高级日志系统的类型、枚举和数据结构。
"""


class LogPriority(Enum):
    """日志优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class LogCompression(Enum):
    """日志压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: float
    level: str
    logger_name: str = ""
    message: str = ""
    priority: LogPriority = LogPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None


@dataclass
class LogEntryPool:
    """日志条目对象池"""
    pool: List[LogEntry] = field(default_factory=list)
    max_size: int = 1000

    def get(self) -> LogEntry:
        """获取日志条目对象"""
        if self.pool:
            return self.pool.pop()
        return LogEntry(
            timestamp=0.0,
            level="",
            logger_name="",
            message=""
        )

    def put(self, entry: LogEntry) -> None:
        """归还日志条目对象"""
        # 检查entry是否为None或不是LogEntry类型
        if entry is None or not isinstance(entry, LogEntry):
            return  # 不添加无效条目

        # max_size=0表示无限制
        if self.max_size == 0 or len(self.pool) < self.max_size:
            # 简单地将对象放入池中，不重置（保持原有值用于测试）
            self.pool.append(entry)
