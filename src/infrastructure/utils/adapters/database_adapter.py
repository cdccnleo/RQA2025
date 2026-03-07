
# -*- coding: utf-8 -*-
import itertools
import threading
import time

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional
"""
基础设施层 - 工具组件组件

database_adapter 模块

通用工具组件
提供工具组件相关的功能实现。
"""

#!/usr/bin/env python3
"""数据库连接池模块"""


class DatabaseAdapter:
    """通用数据库适配器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化数据库适配器"""
        self.config = config or {}
        self.connection = None
    
    def connect(self) -> bool:
        """连接到数据库"""
        return True
    
    def execute(self, query: str, params: tuple = None) -> Any:
        """执行SQL查询"""
        return None
    
    def disconnect(self) -> bool:
        """断开数据库连接"""
        self.close()
        return True
    
    def close(self):
        """关闭连接"""
        if self.connection:
            self.connection = None


class DatabaseConnection(ABC):
    """数据库连接抽象基类"""

    _id_counter = itertools.count(1)

    def __init__(self, connection_id: Optional[str] = None):

        if connection_id is None:
            connection_id = f"conn-{next(self._id_counter)}"
        self.connection_id = connection_id
        self.created_at = time.time()
        self.last_used = time.time()
        self.usage_count = 0
        self._is_closed = False

    @abstractmethod
    def execute(self, query: str, params: tuple = None) -> Any:
        """执行SQL查询"""

    @abstractmethod
    def commit(self):
        """提交事务"""

    @abstractmethod
    def rollback(self):
        """回滚事务"""

    @abstractmethod
    def close(self):
        """关闭连接"""

    def is_closed(self) -> bool:
        """检查连接是否已关闭"""
        return self._is_closed

    def mark_used(self):
        """标记连接被使用"""
        self.last_used = time.time()
        self.usage_count += 1


class MockDatabaseConnection(DatabaseConnection):
    """模拟数据库连接"""

    def __init__(self, connection_id: Optional[str] = None, initial_data: Optional[Dict[str, Any]] = None):

        super().__init__(connection_id)
        self._data: Dict[str, Any] = initial_data.copy() if initial_data else {}

    def execute(self, query: str, params: tuple = None) -> Any:
        """模拟执行SQL查询"""
        if "SELECT" in query.upper():
            return MockCursor(self._data)
        if "INSERT" in query.upper() and params:
            key = params[0] if isinstance(params, (list, tuple)) and params else f"row-{len(self._data)+1}"
            self._data[key] = params
        elif "UPDATE" in query.upper() and params:
            key = params[0] if isinstance(params, (list, tuple)) and params else "updated"
            self._data[key] = params
        return MockCursor(self._data)

    def commit(self):
        """模拟提交事务"""

    def rollback(self):
        """模拟回滚事务"""

    def close(self):
        """关闭连接"""
        self._is_closed = True


class MockCursor:
    """模拟数据库游标"""

    def __init__(self, data: Optional[Dict[str, Any]] = None):

        self.data: Dict[str, Any] = data.copy() if data else {}
        self.rowcount = len(self.data)

    def fetchone(self):
        """获取一行数据"""
        try:
            return next(iter(self.data.values()))
        except StopIteration:
            return None

    def fetchall(self):
        """获取所有数据"""
        return list(self.data.values())


class DatabaseConnectionPool:
    """数据库连接池"""

    def __init__(
        self,
        max_size: int = 10,
        min_size: int = 2,
        idle_timeout: int = 300,
        max_usage: int = 1000,
        leak_detection: bool = True,
    ):
        self.max_size = max_size
        self.min_size = min_size
        self.idle_timeout = idle_timeout
        self.max_usage = max_usage
        self.leak_detection = leak_detection

        self._connections = []
        self._available = []
        self._in_use = {}
        self._lock = threading.Lock()
        self._leak_tracker = {}

        # 初始化最小连接数
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化连接池"""
        for i in range(self.min_size):
            conn = self._create_connection()
            self._connections.append(conn)
            self._available.append(conn)

    def _create_connection(self) -> DatabaseConnection:
        """创建新连接"""
        connection_id = f"conn_{len(self._connections)}_{int(time.time())}"
        return MockDatabaseConnection(connection_id)

    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        connection = self._acquire_connection()
        try:
            yield connection
        finally:
            self._release_connection(connection)

    def _acquire_connection(self) -> DatabaseConnection:
        """获取连接"""
        with self._lock:
            # 优先使用可用连接
            if self._available:
                conn = self._available.pop()
                self._in_use[conn.connection_id] = conn
                conn.mark_used()

                if self.leak_detection:
                    self._leak_tracker[conn.connection_id] = {
                        "acquired_at": time.time(),
                        "thread_id": threading.get_ident(),
                    }

                return conn

            # 如果没有可用连接，检查是否可以创建新连接
            if len(self._connections) < self.max_size:
                conn = self._create_connection()
                self._connections.append(conn)
                self._in_use[conn.connection_id] = conn

                if self.leak_detection:
                    self._leak_tracker[conn.connection_id] = {
                        "acquired_at": time.time(),
                        "thread_id": threading.get_ident(),
                    }

                return conn

            # 连接池已满
            raise RuntimeError("Connection pool exhausted")

    def _release_connection(self, connection: DatabaseConnection):
        """释放连接"""
        with self._lock:
            if connection.connection_id in self._in_use:
                del self._in_use[connection.connection_id]

            if self.leak_detection and connection.connection_id in self._leak_tracker:
                del self._leak_tracker[connection.connection_id]

            # 检查连接是否仍然可用
            if not connection.is_closed() and connection.usage_count < self.max_usage:
                self._available.append(connection)
            else:
                # 移除不可用连接
                if connection in self._connections:
                    self._connections.remove(connection)

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        with self._lock:
            total_connections = len(self._connections)
            available_connections = len(self._available)
            in_use_connections = len(self._in_use)

            # 检查泄漏
            leaks = []
        if self.leak_detection:
            current_time = time.time()
            for conn_id, info in self._leak_tracker.items():
                if current_time - info["acquired_at"] > 300:  # 5分钟未释放
                    leaks.append(
                        {
                            "connection_id": conn_id,
                            "acquired_at": info["acquired_at"],
                            "thread_id": info["thread_id"],
                        }
                    )

            return {
                "total": total_connections,
                "available": available_connections,
                "in_use": in_use_connections,
                "utilization": in_use_connections / max(total_connections, 1),
                "leaks": leaks,
                "max_size": self.max_size,
                "min_size": self.min_size,
                "idle_timeout": self.idle_timeout,
                "max_usage": self.max_usage,
                "leak_detection": self.leak_detection,
            }

        def update_config(self, **kwargs):
            """更新连接池配置"""
            with self._lock:
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

    def close_all(self):
        """关闭所有连接"""
        with self._lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()
            self._available.clear()
            self._in_use.clear()
            self._leak_tracker.clear()
