"""
PostgreSQL连接管理器组件

负责PostgreSQL数据库的连接、断开和健康检查。
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from src.infrastructure.utils.interfaces.database_interfaces import (
        HealthCheckResult,
        ConnectionStatus,
    )
except ImportError:
    from dataclasses import dataclass
    from enum import Enum

    class ConnectionStatus(Enum):
        CONNECTED = "connected"
        DISCONNECTED = "disconnected"
        ERROR = "error"

    @dataclass
    class HealthCheckResult:
        is_healthy: bool
        response_time: float
        message: str = ""
        details: Optional[Dict[str, Any]] = None


class PostgreSQLConnectionManager:
    """PostgreSQL连接管理器"""
    
    def __init__(self):
        """初始化连接管理器"""
        self.client = None
        self.connected = False
        self.config = {}
        self.connection_info = {}
        self._fallback_logger: Optional[logging.Logger] = None
    
    def _safe_log(self, level: int, message: str) -> None:
        """在存在模拟handler时安全地记录日志，避免测试环境中的类型错误"""
        try:
            for handler in logger.handlers:
                handler_level = getattr(handler, "level", None)
                if handler_level is not None and not isinstance(handler_level, int):
                    raise TypeError("Invalid handler level type")
            logger.log(level, message)
        except TypeError:
            if self._fallback_logger is None:
                self._fallback_logger = logging.getLogger(f"{__name__}.safe")
                self._fallback_logger.propagate = False
                if not self._fallback_logger.handlers:
                    self._fallback_logger.addHandler(logging.NullHandler())
            self._fallback_logger.log(level, message)
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """
        连接PostgreSQL数据库
        
        Args:
            config: 连接配置
            
        Returns:
            是否连接成功
        """
        try:
            import psycopg2
            
            # 构建连接参数
            connection_params = {
                "host": config.get("host", "localhost"),
                "port": config.get("port", 5432),
                "database": config.get("database", ""),
                "user": config.get("user", ""),
                "password": config.get("password", ""),
                "connect_timeout": config.get("timeout", 30),
            }
            
            # 建立连接
            self.client = psycopg2.connect(**connection_params)
            self.connected = True
            self.config = config
            
            # 保存连接信息
            self.connection_info = {
                "host": connection_params["host"],
                "port": connection_params["port"],
                "database": connection_params["database"],
                "connected_at": time.time(),
            }
            
            self._safe_log(
                logging.INFO,
                f"PostgreSQL连接成功: {connection_params['host']}:{connection_params['port']}",
            )
            return True
            
        except Exception as e:
            self._safe_log(logging.ERROR, f"PostgreSQL连接失败: {e}")
            self.connected = False
            # 重新抛出异常以符合测试期望
            raise
    
    def disconnect(self) -> bool:
        """
        断开数据库连接
        
        Returns:
            是否断开成功
        """
        try:
            if self.client:
                self.client.close()
                self.client = None
            
            self.connected = False
            self._safe_log(logging.INFO, "PostgreSQL连接已断开")
            return True
            
        except Exception as e:
            self._safe_log(logging.ERROR, f"PostgreSQL断开连接失败: {e}")
            return False
    
    def health_check(self) -> HealthCheckResult:
        """
        执行健康检查
        
        Returns:
            健康检查结果
        """
        start_time = time.time()
        if not self.connected or not self.client:
            return HealthCheckResult(
                is_healthy=False,
                response_time=0.0,
                message="数据库未连接",
                details={
                    "error": "数据库未连接",
                    "timestamp": start_time,
                },
            )

        try:
            # 执行简单查询测试连接
            with self.client.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            
            # 获取数据库版本信息
            version = self._get_database_version()
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                is_healthy=True,
                response_time=response_time,
                message="connected",
                details={
                    "host": self.connection_info.get("host"),
                    "port": self.connection_info.get("port"),
                    "database": self.connection_info.get("database"),
                    "version": version,
                    "connected_at": self.connection_info.get("connected_at"),
                    "timestamp": time.time(),
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self._safe_log(logging.ERROR, f"健康检查失败: {e}")
            return HealthCheckResult(
                is_healthy=False,
                response_time=response_time,
                message=str(e),
                details={
                    "error": str(e),
                    "timestamp": time.time(),
                }
            )
    
    def get_connection_status(self) -> ConnectionStatus:
        """
        获取连接状态
        
        Returns:
            连接状态
        """
        if not self.connected or not self.client:
            return ConnectionStatus.DISCONNECTED
        
        try:
            # 测试连接是否有效
            with self.client.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return ConnectionStatus.CONNECTED
        except:
            self.connected = False
            return ConnectionStatus.ERROR
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return self.connection_info.copy()
    
    def _get_database_version(self) -> str:
        """获取数据库版本信息"""
        try:
            with self.client.cursor() as cursor:
                cursor.execute("SELECT version()")
                result = cursor.fetchone()
                return result[0] if result else "unknown"
        except:
            return "unknown"

