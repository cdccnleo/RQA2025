"""
sqlite_adapter 模块

提供 sqlite_adapter 相关功能和接口。
"""

import json

# SQLite适配器常量
import sqlite3
import threading
import time

from src.infrastructure.utils.core.error import UnifiedErrorHandler as ErrorHandler
from src.infrastructure.utils.interfaces.database_interfaces import (
    IDatabaseAdapter, QueryResult, WriteResult, HealthCheckResult, ConnectionStatus
)
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
"""SQLite数据库适配器"""

#     QueryResult,
#     WriteResult,
#     HealthCheckResult,
#     ConnectionStatus,
#     IDatabaseAdapter,


class SQLiteConstants:
    """SQLite适配器相关常量"""

    # 默认数据库路径
    DEFAULT_DB_PATH = "data/rqa2025.db"

    # 连接超时配置 (秒)
    DEFAULT_CONNECTION_TIMEOUT = 30.0

    # 默认执行时间
    DEFAULT_EXECUTION_TIME = 0.0

    # 默认错误计数
    DEFAULT_ERROR_COUNT = 1

    # 默认受影响行数
    DEFAULT_AFFECTED_ROWS = 0

    # 游标描述索引
    CURSOR_DESCRIPTION_INDEX = 0

    # 测试查询
    HEALTH_CHECK_QUERY = "SELECT 1"


class SQLiteAdapter(IDatabaseAdapter):
    """SQLite数据库适配器，实现统一接口"""

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """初始化SQLite适配器"""
        self.error_handler = error_handler or ErrorHandler()
        self._error_handler = self.error_handler  # 兼容性别名
        self.connection = None
        self._conn = self.connection  # 兼容性别名
        self.db_path = None
        self._connected = False
        self._config = {}
        self._connection_info = {}
        self._lock = threading.Lock()  # 添加线程锁

    def connect(self, config: Dict[str, Any]) -> bool:
        """连接到SQLite数据库"""
        with self._lock:
            try:
                self.db_path = Path(config.get("path", SQLiteConstants.DEFAULT_DB_PATH))
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self.connection = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,  # 允许多线程访问
                    timeout=SQLiteConstants.DEFAULT_CONNECTION_TIMEOUT,  # 设置超时时间
                )

                self._conn = self.connection  # 设置兼容性属性
                # 启用外键约束
                self.connection.execute("PRAGMA foreign_keys = ON")
                # 创建必要的表
                self._create_tables()
                self._connected = True
                return True
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle(e, "SQLite连接失败")
                self._connected = False
                return False

    def disconnect(self) -> bool:
        """断开数据库连接"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self._connected = False
            return True
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, "SQLite断开连接失败")
            return False

    def is_connected(self) -> bool:
        """检查是否已连接到数据库"""
        return self._connected and self.connection is not None

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """执行查询"""
        if not self._connected:
            return QueryResult(
                success=False,
                data=[],
                row_count=0,
                execution_time=0.0,
                error_message="数据库未连接"
            )

        start_time = time.time()
        with self._lock:  # 添加线程锁
            try:
                # 处理参数
                if params:
                    # 将字典参数转换为位置参数
                    param_list = list(params.values())
                else:
                    param_list = []
                cursor = self.connection.cursor()
                cursor.execute(query, param_list)
                # 获取结果
                if query.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    columns = [
                        description[SQLiteConstants.CURSOR_DESCRIPTION_INDEX] for description in cursor.description
                    ]
                    result_data = [dict(zip(columns, row)) for row in rows]
                else:
                    result_data = []
                execution_time = time.time() - start_time
                return QueryResult(
                    success=True,
                    data=result_data,
                    row_count=len(result_data),
                    execution_time=execution_time
                )

            except Exception as e:
                return QueryResult(
                    success=False,
                    data=[],
                    row_count=0,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

    def execute_write(self, data: Dict[str, Any]) -> WriteResult:
        """执行写入操作"""
        if not self._connected:
            return WriteResult(success=False, affected_rows=0, execution_time=0.0, error_message="数据库未连接")

        start_time = time.time()
        with self._lock:  # 添加线程锁
            try:
                measurement = data.get("measurement", "default")
                field_set = json.dumps(data.get("fields", {}))
                tag_set = json.dumps(data.get("tags", {}))
                cursor = self.connection.cursor()
                cursor.execute(
                    "INSERT INTO time_series (measurement, field_set, tag_set) VALUES (?, ?, ?)",
                    (measurement, field_set, tag_set)
                )

                affected_rows = cursor.rowcount
                execution_time = time.time() - start_time
                return WriteResult(
                    success=True,
                    affected_rows=affected_rows,
                    execution_time=execution_time
                )

            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle(e, "SQLite写入失败")
                return WriteResult(
                    success=False,
                    affected_rows=0,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

    def batch_write(self, data_list: List[Dict[str, Any]]) -> WriteResult:
        """批量写入操作"""
        if not self._connected:
            return WriteResult(success=False, affected_rows=0, execution_time=0.0, error_message="数据库未连接")

        start_time = time.time()
        with self._lock:  # 添加线程锁
            try:
                cursor = self.connection.cursor()
                affected_rows = SQLiteConstants.DEFAULT_AFFECTED_ROWS
                for data in data_list:
                    measurement = data.get("measurement", "default")
                    field_set = json.dumps(data.get("fields", {}))
                    tag_set = json.dumps(data.get("tags", {}))
                    cursor.execute(
                        "INSERT INTO time_series (measurement, field_set, tag_set) VALUES (?, ?, ?)",
                        (measurement, field_set, tag_set)
                    )
                    affected_rows += cursor.rowcount

                execution_time = time.time() - start_time
                return WriteResult(
                    success=True,
                    affected_rows=affected_rows,
                    execution_time=execution_time
                )

            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle(e, "SQLite批量写入失败")
                return WriteResult(
                    success=False,
                    affected_rows=0,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )

    def health_check(self) -> HealthCheckResult:
        """健康检查"""
        if not self._connected:
            return HealthCheckResult(
                is_healthy=False,
                response_time=SQLiteConstants.DEFAULT_EXECUTION_TIME,
                message="数据库未连接",
                details={"error": "数据库未连接"}
            )

        start_time = time.time()
        with self._lock:  # 添加线程锁
            try:
                cursor = self.connection.cursor()
                cursor.execute(SQLiteConstants.HEALTH_CHECK_QUERY)
                cursor.fetchone()
                response_time = time.time() - start_time
                return HealthCheckResult(
                    is_healthy=True,
                    response_time=response_time,
                    message="健康",
                    details={"database_path": str(self.db_path)}
                )

            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle(e, "SQLite健康检查失败")
                return HealthCheckResult(
                    is_healthy=False,
                    response_time=time.time() - start_time,
                    message=str(e),
                    details={"error": str(e)}
                )

    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            "database_path": str(self.db_path),
            "connected": self._connected,
            "database_type": "sqlite",
        }

    def connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        if not self._connected:
            return {
                "connected": False,
                "status": ConnectionStatus.DISCONNECTED.value,
                "database_type": "sqlite"
            }

        try:
            # 执行简单查询检查连接是否有效
            with self._lock:
                cursor = self.connection.cursor()
                cursor.execute(SQLiteConstants.HEALTH_CHECK_QUERY)
                cursor.fetchone()
            return {
                "connected": True,
                "status": ConnectionStatus.CONNECTED.value,
                "database_type": "sqlite"
            }
        except Exception:
            self._connected = False
            return {
                "connected": False,
                "status": ConnectionStatus.DISCONNECTED.value,
                "database_type": "sqlite"
            }

    def close(self) -> None:
        """关闭连接"""
        if self._conn:
            conn = self._conn
            self._conn = None
            if conn:
                conn.close()
            self.connection = None
            self._connected = False

    def write(
        self,
        measurement: str,
        data: Dict[str, Any],
        tags: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """兼容性写入方法"""
        if not self._connected:
            return False
        try:
            field_set = json.dumps(data)
            tag_set = json.dumps(tags or {})
            self._conn.execute(
                "INSERT INTO time_series (measurement, field_set, tag_set) VALUES (?, ?, ?)",
                (measurement, field_set, tag_set)
            )

            return True
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, "SQLite写入失败")
            raise e

    def query(self, query: str) -> List[Tuple]:
        """兼容性查询方法"""
        if not self._connected:
            return []
        try:
            cursor = self._conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, "SQLite查询失败")
            raise e

    def _create_tables(self):
        """创建基础表结构"""
        try:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS time_series (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    measurement TEXT NOT NULL,
                    field_set TEXT,
                    tag_set TEXT
                )
            """
            )
            # 创建索引
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_measurement
                ON time_series(measurement)
            """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON time_series(timestamp)
            """
            )
            self.connection.commit()
        except Exception:
            pass  # 如果表已存在或其他错误，忽略
    
    def begin_transaction(self):
        """开始事务"""
        if not self._connected:
            return None
        return self.connection
    
    def commit(self):
        """提交事务"""
        if not self._connected:
            return False
        try:
            self.connection.commit()
            return True
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, "SQLite事务提交失败")
            return False
    
    def rollback(self):
        """回滚事务"""
        if not self._connected:
            return False
        try:
            self.connection.rollback()
            return True
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle(e, "SQLite事务回滚失败")
            return False
