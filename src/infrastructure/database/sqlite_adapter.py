"""SQLite数据库适配器(临时替代方案)"""
import sqlite3
from typing import Dict, List, Optional
from pathlib import Path
from ..error import ErrorHandler

class SQLiteAdapter:
    """SQLite数据库适配器，提供与InfluxDB兼容的基础接口"""

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self._db_path = Path("rqa2025_temp.db")
        self._conn = None
        self._error_handler = error_handler or ErrorHandler()

    def connect(self, config: Dict):
        """连接数据库"""
        try:
            self._conn = sqlite3.connect(self._db_path)
            self._create_tables()
        except Exception as e:
            self._error_handler.handle(e, "SQLite连接失败")
            raise

    def _create_tables(self):
        """创建基础表结构"""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS time_series (
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    measurement TEXT,
                    field_set TEXT,
                    tag_set TEXT
                )
            """)

    def write(self, measurement: str, data: Dict, tags: Dict = None):
        """写入数据(简化实现)"""
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO time_series (measurement, field_set, tag_set) VALUES (?, ?, ?)",
                    (measurement, str(data), str(tags or {}))
                )
        except Exception as e:
            self._error_handler.handle(e, "SQLite写入失败")
            raise

    def query(self, query: str):
        """执行查询(简化实现)"""
        try:
            cursor = self._conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            self._error_handler.handle(e, "SQLite查询失败")
            raise

    def close(self):
        """关闭连接"""
        if self._conn:
            self._conn.close()
