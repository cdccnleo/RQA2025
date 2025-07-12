"""数据库管理器，负责适配器选择和初始化"""
import json
import threading
from pathlib import Path
from typing import Optional, Any
from .influxdb_adapter import InfluxDBAdapter
from .influxdb_manager import InfluxDBManager
from .sqlite_adapter import SQLiteAdapter
from .connection_pool import ConnectionPool
from ..error import ErrorHandler

class DatabaseManager:
    """统一数据库管理器"""
    
    _config = None
    _instance = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        config_mock: Optional[Any] = None,
        pool_mock: Optional[Any] = None,
        adapter_mock: Optional[Any] = None,
        error_handler_mock: Optional[Any] = None
    ):
        """
        初始化数据库管理器
        
        Args:
            config_mock: 测试用mock的配置对象
            pool_mock: 测试用mock的连接池对象
            adapter_mock: 测试用mock的适配器对象
            error_handler_mock: 测试用mock的错误处理器对象
        """
        # 测试钩子：允许注入mock的依赖
        if config_mock is not None:
            self._config = config_mock
        if pool_mock is not None:
            self._pool = pool_mock
        else:
            self._pool = ConnectionPool(
                max_size=10,
                idle_timeout=300
            )
        if adapter_mock is not None:
            self._adapter = adapter_mock
        if error_handler_mock is not None:
            self._error_handler = error_handler_mock
        else:
            self._error_handler = ErrorHandler()
    
    @classmethod
    def _load_config(cls):
        """加载数据库配置"""
        if cls._config is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "database.json"
            with open(config_path, "r", encoding="utf-8") as f:
                cls._config = json.load(f)
        return cls._config
    
    @classmethod
    def _get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._pool = ConnectionPool(
                        max_size=10,
                        idle_timeout=300
                    )
        return cls._instance
    
    @classmethod
    def get_adapter(cls):
        """获取数据库适配器实例"""
        config = cls._load_config()
        if config.get("sqlite", {}).get("enabled", False):
            adapter = SQLiteAdapter(ErrorHandler())
        else:
            adapter = InfluxDBAdapter()
        return adapter
    
    @classmethod
    def get_manager(cls):
        """获取数据库管理器实例"""
        config = cls._load_config()
        if config.get("sqlite", {}).get("enabled", False):
            manager = SQLiteAdapter(ErrorHandler())
        else:
            manager = InfluxDBManager()
        return manager
    
    @classmethod
    def health_check(cls):
        """返回连接池健康状态"""
        return cls._get_instance()._pool.health_check()
