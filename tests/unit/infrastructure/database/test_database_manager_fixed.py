import pytest
from unittest.mock import patch, Mock, MagicMock
from src.infrastructure.database.database_manager import DatabaseManager

class TestDatabaseManagerFixed:
    """数据库管理器修复版本测试"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        # 测试前后重置单例，避免污染
        DatabaseManager._instance = None
        DatabaseManager._config = None
        yield
        DatabaseManager._instance = None
        DatabaseManager._config = None

    def test_load_config_success(self, tmp_path):
        # 创建临时配置文件
        config_data = {"sqlite": {"enabled": True}}
        config_path = tmp_path / "database.json"
        # 正确写入JSON字符串
        config_path.write_text('{"sqlite": {"enabled": true}}', encoding="utf-8")
        with patch("src.infrastructure.database.database_manager.Path") as mock_path:
            mock_path.return_value = tmp_path
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = config_path.read_text()
                DatabaseManager._config = None
                config = DatabaseManager._load_config()
                assert "sqlite" in config

    def test_get_adapter_sqlite(self):
        # Mock配置为sqlite启用
        with patch.object(DatabaseManager, "_load_config", return_value={"sqlite": {"enabled": True}}):
            with patch("src.infrastructure.database.database_manager.SQLiteAdapter") as mock_sqlite:
                mock_sqlite.return_value = Mock()
                adapter = DatabaseManager.get_adapter()
                assert adapter == mock_sqlite.return_value

    def test_get_adapter_influx(self):
        # Mock配置为sqlite未启用
        with patch.object(DatabaseManager, "_load_config", return_value={"sqlite": {"enabled": False}}):
            with patch("src.infrastructure.database.database_manager.InfluxDBAdapter") as mock_influx:
                mock_influx.return_value = Mock()
                adapter = DatabaseManager.get_adapter()
                assert adapter == mock_influx.return_value

    def test_get_manager_sqlite(self):
        # Mock配置为sqlite启用
        with patch.object(DatabaseManager, "_load_config", return_value={"sqlite": {"enabled": True}}):
            with patch("src.infrastructure.database.database_manager.SQLiteAdapter") as mock_sqlite:
                with patch("src.infrastructure.database.database_manager.ErrorHandler") as mock_error_handler:
                    mock_sqlite.return_value = Mock()
                    mock_error_handler.return_value = Mock()
                    manager = DatabaseManager.get_manager()
                    assert manager == mock_sqlite.return_value

    def test_get_manager_influx(self):
        # Mock配置为sqlite未启用
        with patch.object(DatabaseManager, "_load_config", return_value={"sqlite": {"enabled": False}}):
            with patch("src.infrastructure.database.database_manager.InfluxDBManager") as mock_influx_mgr:
                mock_influx_mgr.return_value = Mock()
                manager = DatabaseManager.get_manager()
                assert manager == mock_influx_mgr.return_value

    def test_health_check(self):
        # Mock连接池健康检查
        pool_mock = Mock()
        pool_mock.health_check.return_value = {"status": "ok"}
        with patch.object(DatabaseManager, "_get_instance") as mock_get_instance:
            with patch("src.infrastructure.database.database_manager.ErrorHandler") as mock_error_handler:
                instance = DatabaseManager(pool_mock=pool_mock)
                instance._pool = pool_mock
                mock_get_instance.return_value = instance
                mock_error_handler.return_value = Mock()
                result = DatabaseManager.health_check()
                assert result == {"status": "ok"}

    def test_init_with_mocks(self):
        # 测试依赖注入
        config_mock = {"sqlite": {"enabled": True}}
        pool_mock = Mock()
        adapter_mock = Mock()
        error_handler_mock = Mock()
        mgr = DatabaseManager(config_mock, pool_mock, adapter_mock, error_handler_mock)
        assert mgr._config == config_mock
        assert mgr._pool == pool_mock
        assert mgr._adapter == adapter_mock
        assert mgr._error_handler == error_handler_mock 