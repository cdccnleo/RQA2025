"""数据库管理器，负责适配器选择和初始化"""
import json
import threading
from pathlib import Path
from typing import Optional, Any, Dict
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

    def get_connection(self):
        """获取数据库连接"""
        return self._pool.get_connection()

    def release_connection(self, connection):
        """释放数据库连接"""
        self._pool.release_connection(connection)

    def execute_query(self, query: str, params: Dict = None):
        """执行查询"""
        if self._adapter:
            return self._adapter.execute_query(query, params)
        return None

    def close(self):
        """关闭数据库管理器"""
        if hasattr(self, '_pool'):
            self._pool.close()

    def get_status(self) -> Dict[str, Any]:
        """获取数据库状态"""
        return {
            'pool_size': getattr(self._pool, 'size', 0),
            'active_connections': getattr(self._pool, 'active_connections', 0),
            'adapter_type': type(self._adapter).__name__ if self._adapter else None
        }
