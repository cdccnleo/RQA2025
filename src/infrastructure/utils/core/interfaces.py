"""
interfaces 模块

提供 interfaces 相关功能和接口。
"""

import logging

# 获取日志记录器

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
"""数据库接口定义

本模块定义了数据库适配器的标准接口，包括连接管理、查询执行、数据写入和健康检查等功能。
所有数据库适配器都应实现这些接口以保证一致性和可替换性。
"""

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """连接状态枚举"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class QueryResult:
    """查询结果"""

    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class WriteResult:
    """写入结果"""

    success: bool
    affected_rows: int
    execution_time: float
    error_message: Optional[str] = None


@dataclass
class HealthCheckResult:
    """健康检查结果"""

    is_healthy: bool
    response_time: float
    connection_count: int
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class IDatabaseAdapter(ABC):
    """数据库适配器接口"""

    def __init__(self):
        """初始化适配器"""
        try:
            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        except Exception as e:
            # 如果日志初始化失败，使用根日志记录器
            self._logger = logger
            self._logger.warning(f"Failed to initialize logger for {self.__class__.__name__}: {e}")

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置参数"""
        try:
            if not isinstance(config, dict):
                self._logger.error("Configuration must be a dictionary")
                return False

            if not config:
                self._logger.error("Configuration cannot be empty")
                return False

            return True
        except Exception as e:
            self._logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_query(self, query: str) -> bool:
        """验证查询语句"""
        try:
            if not isinstance(query, str):
                self._logger.error("Query must be a string")
                return False

            if not query.strip():
                self._logger.error("Query cannot be empty")
                return False

            return True
        except Exception as e:
            self._logger.error(f"Query validation failed: {e}")
            return False

    def _validate_data_list(self, data_list: List[Dict[str, Any]]) -> bool:
        """验证数据列表"""
        try:
            if not isinstance(data_list, list):
                self._logger.error("Data list must be a list")
                return False

            if not data_list:
                self._logger.warning("Data list is empty")
                return True  # 空列表是有效的

            for i, item in enumerate(data_list):
                if not isinstance(item, dict):
                    self._logger.error(f"Data item at index {i} must be a dictionary")
                    return False

            return True
        except Exception as e:
            self._logger.error(f"Data list validation failed: {e}")
            return False

    def _create_error_result(self, error_message: str, execution_time: float = 0.0) -> Any:
        """创建错误结果（需要在子类中实现）"""
        try:
            # 这是一个占位符方法，实际实现应在具体子类中
            raise NotImplementedError("Subclasses must implement _create_error_result")
        except Exception as e:
            self._logger.error(f"Error result creation failed: {e}")
            return None

    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """连接到数据库"""
        try:
            if not self._validate_config(config):
                return False
            # 具体实现由子类提供
        except Exception as e:
            self._logger.error(f"Connection failed: {e}")
            return False

    @abstractmethod
    def disconnect(self) -> bool:
        """断开数据库连接"""
        try:
            # 具体实现由子类提供
            pass
        except Exception as e:
            self._logger.error(f"Disconnection failed: {e}")
            return False

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> QueryResult:
        """执行查询"""
        try:
            if not self._validate_query(query):
                return self._create_error_result("Invalid query")
            # 具体实现由子类提供
        except Exception as e:
            self._logger.error(f"Query execution failed: {e}")
            return self._create_error_result(str(e))

    @abstractmethod
    def execute_write(self, query: str, params: Optional[Tuple] = None) -> WriteResult:
        """执行写入操作"""
        try:
            if not self._validate_query(query):
                return self._create_error_result("Invalid query")
            # 具体实现由子类提供
        except Exception as e:
            self._logger.error(f"Write execution failed: {e}")
            return self._create_error_result(str(e))

    @abstractmethod
    def batch_write(self, data_list: List[Dict[str, Any]]) -> WriteResult:
        """批量写入"""
        try:
            if not self._validate_data_list(data_list):
                return self._create_error_result("Invalid data list")
            # 具体实现由子类提供
        except Exception as e:
            self._logger.error(f"Batch write failed: {e}")
            return self._create_error_result(str(e))

    @abstractmethod
    def health_check(self) -> HealthCheckResult:
        """健康检查"""
        try:
            # 具体实现由子类提供
            pass
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                is_healthy=False,
                response_time=0.0,
                connection_count=0,
                error_message=str(e)
            )

    @property
    @abstractmethod
    def connection_status(self) -> ConnectionStatus:
        """获取连接状态"""
        try:
            # 具体实现由子类提供
            pass
        except Exception as e:
            self._logger.error(f"Failed to get connection status: {e}")
            return ConnectionStatus.ERROR

    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        try:
            # 具体实现由子类提供
            pass
        except Exception as e:
            self._logger.error(f"Failed to get connection info: {e}")
            return {"error": str(e)}


class ITransaction(ABC):
    """事务接口"""

    @abstractmethod
    def begin(self):
        """开始事务"""

    @abstractmethod
    def commit(self):
        """提交事务"""

    @abstractmethod
    def rollback(self):
        """回滚事务"""

    @abstractmethod
    def is_active(self) -> bool:
        """检查事务是否活跃"""


class IConcurrencyController(ABC):
    """并发控制器接口"""

    @abstractmethod
    def acquire(self, resource: str = "default") -> bool:
        """获取资源锁"""

    @abstractmethod
    def release(self, resource: str = "default") -> bool:
        """释放资源锁"""

    @abstractmethod
    def get_active_count(self, resource: str = "default") -> int:
        """获取活跃资源数量"""

    @property
    @abstractmethod
    def max_concurrent(self) -> int:
        """获取最大并发数"""
