"""
数据库适配器核心接口

本模块定义底层数据库适配器的接口和数据类型。

重要说明：
    本模块的QueryResult用于数据库适配器层，表示原始数据库查询结果。
    如需高级查询接口，请使用 src.infrastructure.utils.components.unified_query.QueryResult
    
    两者区别：
    - database_interfaces.QueryResult: 轻量级，List[Dict]格式，无外部依赖
    - unified_query.QueryResult: 高级接口，pd.DataFrame格式，支持跨存储查询

使用场景：
    ✅ 实现数据库适配器（PostgreSQL, Redis, SQLite等）
    ✅ 直接数据库CRUD操作
    ✅ 需要轻量级数据格式
    
    ❌ 不适用于复杂数据分析
    ❌ 不适用于跨存储查询聚合
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field

class ConnectionStatus(Enum):
    """连接状态枚举"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"

@dataclass
class QueryResult:
    """
    数据库查询结果（适配器层）
    
    用途：
        数据库适配器的直接查询结果，表示原始数据库返回的数据。
        用于底层数据库操作，轻量级设计，无外部依赖。
    
    注意：
        ⚠️ 本类与 unified_query.QueryResult 不同！
        - 本类用于数据库适配器层
        - 数据格式为 List[Dict]，轻量级
        - 如需高级查询和数据分析，使用 unified_query.QueryResult
    
    使用示例：
        >>> result = QueryResult(
        ...     success=True,
        ...     data=[{"id": 1, "name": "test"}],
        ...     row_count=1,
        ...     execution_time=0.5
        ... )
    
    Attributes:
        success: 查询是否成功
        data: 查询结果数据（字典列表格式）
        row_count: 返回的行数
        execution_time: 执行时间（秒）
        error_message: 错误信息（如果失败）
    """
    success: bool
    data: List[Any] = field(default_factory=list)
    row_count: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    query_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    storage_type: Optional[str] = None
    rows_affected: Optional[int] = None
    status: Optional[str] = None
    raw_response: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.row_count == 0 and self.data:
            self.row_count = len(self.data)
        if self.rows_affected is not None and self.row_count == 0:
            self.row_count = self.rows_affected

@dataclass
class WriteResult:
    """写入结果（兼容 legacy error 字段）"""

    success: bool
    affected_rows: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    error: Optional[str] = None
    insert_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    rows_affected: Optional[int] = None
    status: Optional[str] = None

    def __post_init__(self) -> None:
        # 兼容 legacy rows_affected 与 affected_rows
        if self.rows_affected is not None:
            self.affected_rows = self.rows_affected
        else:
            self.rows_affected = self.affected_rows

        # 同步 error / error_message 字段，确保双向可用
        if self.error is None and self.error_message is not None:
            self.error = self.error_message
        elif self.error is not None and self.error_message is None:
            self.error_message = self.error

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    is_healthy: bool
    response_time: float = 0.0
    message: str = ""
    details: Optional[Dict[str, Any]] = None

class IDatabaseAdapter(ABC):
    """数据库适配器接口"""

    @abstractmethod
    def connect(self) -> bool:
        """连接数据库"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开数据库连接"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> QueryResult:
        """执行查询"""
        pass

    @abstractmethod
    def execute_write(self, query: str, params: Optional[List[Any]] = None) -> WriteResult:
        """执行写入操作"""
        pass

    @abstractmethod
    def health_check(self) -> HealthCheckResult:
        """健康检查"""
        pass

class ITransaction(ABC):
    """事务接口"""

    @abstractmethod
    def begin(self):
        """开始事务"""
        pass

    @abstractmethod
    def commit(self):
        """提交事务"""
        pass

    @abstractmethod
    def rollback(self):
        """回滚事务"""
        pass
