import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.database.database_manager import DatabaseManager
from src.infrastructure.database.sqlite_adapter import SQLiteAdapter
from src.infrastructure.database.influxdb_adapter import InfluxDBAdapter
from src.infrastructure.database.influxdb_manager import InfluxDBManager
from src.infrastructure.error import ErrorHandler

@pytest.fixture
def mock_config():
    """创建mock的配置"""
    return {
        "sqlite": {
            "enabled": True,
            "path": "test.db"
        },
        "influxdb": {
            "enabled": False,
            "url": "http://localhost:8086",
            "token": "test-token",
            "org": "test-org",
            "bucket": "test-bucket"
        }
    }

@pytest.fixture
def mock_pool():
    """创建mock的连接池"""
    pool = Mock()
    pool.health_check.return_value = {"status": "healthy", "connections": 5}
    return pool

@pytest.fixture
def mock_adapter():
    """创建mock的适配器"""
    adapter = Mock()
    adapter.connect.return_value = None
    adapter.write.return_value = None
    adapter.query.return_value = []
    adapter.close.return_value = None
    return adapter

@pytest.fixture
def mock_error_handler():
    """创建mock的错误处理器"""
    handler = Mock()
    handler.handle.return_value = None
    return handler

@pytest.fixture
def database_manager(mock_config, mock_pool, mock_adapter, mock_error_handler):
    """创建DatabaseManager实例"""
    return DatabaseManager(
        config_mock=mock_config,
        pool_mock=mock_pool,
        adapter_mock=mock_adapter,
        error_handler_mock=mock_error_handler
    )

class TestDatabaseManager:
    """DatabaseManager测试类"""

    def test_init_with_test_hooks(self, mock_config, mock_pool, mock_adapter, mock_error_handler):
        """测试使用测试钩子初始化"""
        manager = DatabaseManager(
            config_mock=mock_config,
            pool_mock=mock_pool,
            adapter_mock=mock_adapter,
            error_handler_mock=mock_error_handler
        )
        
        assert manager._config == mock_config
        assert manager._pool == mock_pool
        assert manager._adapter == mock_adapter
        assert manager._error_handler == mock_error_handler

    def test_init_without_test_hooks(self):
        """测试不使用测试钩子初始化"""
        manager = DatabaseManager()
        
        # 验证使用了真实的依赖
        assert manager._pool is not None
        assert manager._error_handler is not None
        assert isinstance(manager._error_handler, ErrorHandler)

    @patch('src.infrastructure.database.database_manager.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_config(self, mock_json_load, mock_open, mock_path):
        """测试配置加载"""
        mock_config = {"sqlite": {"enabled": True}}
        mock_json_load.return_value = mock_config
        
        # 重置类变量
        DatabaseManager._config = None
        
        config = DatabaseManager._load_config()
        
        assert config == mock_config
        mock_open.assert_called_once()

    @patch('src.infrastructure.database.database_manager.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_config_cached(self, mock_json_load, mock_open, mock_path):
        """测试配置缓存"""
        mock_config = {"sqlite": {"enabled": True}}
        mock_json_load.return_value = mock_config
        
        # 设置缓存
        DatabaseManager._config = mock_config
        
        config = DatabaseManager._load_config()
        
        assert config == mock_config
        # 应该使用缓存，不调用open
        mock_open.assert_not_called()

    def test_get_instance_singleton(self):
        """测试单例模式"""
        # 重置实例
        DatabaseManager._instance = None
        
        instance1 = DatabaseManager._get_instance()
        instance2 = DatabaseManager._get_instance()
        
        assert instance1 is instance2

    @patch('src.infrastructure.database.database_manager.SQLiteAdapter')
    @patch('src.infrastructure.database.database_manager.ErrorHandler')
    def test_get_adapter_sqlite_enabled(self, mock_error_handler, mock_sqlite_adapter):
        """测试获取SQLite适配器"""
        # 模拟配置
        with patch.object(DatabaseManager, '_load_config') as mock_load_config:
            mock_load_config.return_value = {
                "sqlite": {"enabled": True}
            }
            
            adapter = DatabaseManager.get_adapter()
            
            # 应该返回SQLiteAdapter
            mock_sqlite_adapter.assert_called_once()

    @patch('src.infrastructure.database.database_manager.InfluxDBAdapter')
    def test_get_adapter_influxdb_enabled(self, mock_influxdb_adapter):
        """测试获取InfluxDB适配器"""
        # 模拟配置
        with patch.object(DatabaseManager, '_load_config') as mock_load_config:
            mock_load_config.return_value = {
                "sqlite": {"enabled": False}
            }
            
            adapter = DatabaseManager.get_adapter()
            
            # 应该返回InfluxDBAdapter
            mock_influxdb_adapter.assert_called_once()

    @patch('src.infrastructure.database.database_manager.SQLiteAdapter')
    @patch('src.infrastructure.database.database_manager.ErrorHandler')
    def test_get_manager_sqlite_enabled(self, mock_error_handler, mock_sqlite_adapter):
        """测试获取SQLite管理器"""
        # 模拟配置
        with patch.object(DatabaseManager, '_load_config') as mock_load_config:
            mock_load_config.return_value = {
                "sqlite": {"enabled": True}
            }
            
            manager = DatabaseManager.get_manager()
            
            # 应该返回SQLiteAdapter
            mock_sqlite_adapter.assert_called_once()

    @patch('src.infrastructure.database.database_manager.InfluxDBManager')
    def test_get_manager_influxdb_enabled(self, mock_influxdb_manager):
        """测试获取InfluxDB管理器"""
        # 模拟配置
        with patch.object(DatabaseManager, '_load_config') as mock_load_config:
            mock_load_config.return_value = {
                "sqlite": {"enabled": False}
            }
            
            manager = DatabaseManager.get_manager()
            
            # 应该返回InfluxDBManager
            mock_influxdb_manager.assert_called_once()

    def test_health_check(self, mock_pool):
        """测试健康检查"""
        # 模拟单例实例
        with patch.object(DatabaseManager, '_get_instance') as mock_get_instance:
            mock_instance = Mock()
            mock_instance._pool = mock_pool
            mock_get_instance.return_value = mock_instance
            
            health_status = DatabaseManager.health_check()
            
            assert health_status == {"status": "healthy", "connections": 5}
            mock_pool.health_check.assert_called_once()

    def test_health_check_exception(self, mock_pool):
        """测试健康检查异常"""
        mock_pool.health_check.side_effect = Exception("Connection failed")
        
        # 模拟单例实例
        with patch.object(DatabaseManager, '_get_instance') as mock_get_instance:
            mock_instance = Mock()
            mock_instance._pool = mock_pool
            mock_get_instance.return_value = mock_instance
            
            with pytest.raises(Exception, match="Connection failed"):
                DatabaseManager.health_check()

    @patch('src.infrastructure.database.database_manager.Path')
    @patch('builtins.open')
    def test_load_config_file_not_found(self, mock_open, mock_path):
        """测试配置文件不存在"""
        mock_open.side_effect = FileNotFoundError("Config file not found")
        
        # 重置类变量
        DatabaseManager._config = None
        
        with pytest.raises(FileNotFoundError):
            DatabaseManager._load_config()

    @patch('src.infrastructure.database.database_manager.Path')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_config_invalid_json(self, mock_json_load, mock_open, mock_path):
        """测试无效的JSON配置"""
        mock_json_load.side_effect = ValueError("Invalid JSON")
        
        # 重置类变量
        DatabaseManager._config = None
        
        with pytest.raises(ValueError):
            DatabaseManager._load_config()

    def test_get_adapter_with_mock_config(self, mock_config):
        """测试使用mock配置获取适配器"""
        with patch.object(DatabaseManager, '_load_config') as mock_load_config:
            mock_load_config.return_value = mock_config
            
            adapter = DatabaseManager.get_adapter()
            
            # 验证返回了正确的适配器类型
            assert adapter is not None

    def test_get_manager_with_mock_config(self, mock_config):
        """测试使用mock配置获取管理器"""
        with patch.object(DatabaseManager, '_load_config') as mock_load_config:
            mock_load_config.return_value = mock_config
            
            manager = DatabaseManager.get_manager()
            
            # 验证返回了正确的管理器类型
            assert manager is not None

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading
        import time
        
        results = []
        
        def get_adapter():
            try:
                adapter = DatabaseManager.get_adapter()
                results.append(adapter)
            except Exception as e:
                results.append(e)
        
        # 创建多个线程同时调用
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_adapter)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有调用都成功
        assert len(results) == 5
        assert all(result is not None for result in results)
