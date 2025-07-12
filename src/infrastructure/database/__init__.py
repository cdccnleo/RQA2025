"""数据库抽象层模块"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

__all__ = ['DatabaseAdapter', 'DatabaseManager']

class DatabaseAdapter(ABC):
    """数据库适配器抽象基类"""
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> None:
        """连接数据库"""
        pass

    @abstractmethod
    def write(self, 
             measurement: str, 
             data: Dict[str, Any], 
             tags: Optional[Dict[str, str]] = None) -> None:
        """写入数据"""
        pass

    @abstractmethod
    def query(self, query: str) -> List[Dict[str, Any]]:
        """查询数据"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭连接"""
        pass

    def __enter__(self) -> 'DatabaseAdapter':
        """上下文管理入口"""
        return self

    def __exit__(self, 
                exc_type: Optional[type], 
                exc_val: Optional[Exception], 
                exc_tb: Optional[Any]) -> None:
        """上下文管理出口"""
        self.close()

from .database_manager import DatabaseManager
from .influxdb_adapter import InfluxDBAdapter
from .influxdb_manager import InfluxDBManager
from .sqlite_adapter import SQLiteAdapter

# 类型别名
DatabaseAdapter = DatabaseAdapter
