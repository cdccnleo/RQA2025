"""
测试 ConfigManagerCore 核心功能

覆盖 ConfigManagerCore 的基本配置管理功能
"""

import pytest
from src.infrastructure.config.core.config_manager_core import ConfigManagerCore


class TestConfigManagerCore:
    """ConfigManagerCore 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = ConfigManagerCore()
        assert manager.configs == {}
        assert hasattr(manager, 'configs')

    def test_set_config(self):
        """测试设置配置"""
        manager = ConfigManagerCore()

        # Test setting simple value
        manager.set_config("key1", "value1")
        assert manager.configs["key1"] == "value1"

        # Test setting different types
        manager.set_config("key2", 123)
        assert manager.configs["key2"] == 123

        manager.set_config("key3", {"nested": "value"})
        assert manager.configs["key3"] == {"nested": "value"}

        manager.set_config("key4", [1, 2, 3])
        assert manager.configs["key4"] == [1, 2, 3]

    def test_get_config(self):
        """测试获取配置"""
        manager = ConfigManagerCore()

        # Test getting non-existent key
        result = manager.get_config("nonexistent")
        assert result is None

        # Test getting existing key
        manager.set_config("key1", "value1")
        result = manager.get_config("key1")
        assert result == "value1"

        # Test getting different types
        manager.set_config("key2", 123)
        result = manager.get_config("key2")
        assert result == 123

        manager.set_config("key3", {"nested": "value"})
        result = manager.get_config("key3")
        assert result == {"nested": "value"}

    def test_config_persistence(self):
        """测试配置持久性"""
        manager = ConfigManagerCore()

        # Set multiple configs
        manager.set_config("app.name", "MyApp")
        manager.set_config("app.version", "1.0.0")
        manager.set_config("database.host", "localhost")
        manager.set_config("database.port", 5432)

        # Verify all configs are stored
        assert manager.get_config("app.name") == "MyApp"
        assert manager.get_config("app.version") == "1.0.0"
        assert manager.get_config("database.host") == "localhost"
        assert manager.get_config("database.port") == 5432

    def test_config_overwrite(self):
        """测试配置覆盖"""
        manager = ConfigManagerCore()

        # Set initial value
        manager.set_config("key1", "initial_value")
        assert manager.get_config("key1") == "initial_value"

        # Overwrite with new value
        manager.set_config("key1", "new_value")
        assert manager.get_config("key1") == "new_value"

    def test_multiple_instances_isolation(self):
        """测试多实例隔离"""
        manager1 = ConfigManagerCore()
        manager2 = ConfigManagerCore()

        # Set different values in each manager
        manager1.set_config("key1", "value1")
        manager2.set_config("key1", "value2")

        # Verify isolation
        assert manager1.get_config("key1") == "value1"
        assert manager2.get_config("key1") == "value2"
        assert manager1.configs != manager2.configs

    def test_empty_key_handling(self):
        """测试空键处理"""
        manager = ConfigManagerCore()

        # Test empty string key
        manager.set_config("", "empty_key_value")
        result = manager.get_config("")
        assert result == "empty_key_value"

    def test_none_value_handling(self):
        """测试None值处理"""
        manager = ConfigManagerCore()

        # Test setting None value
        manager.set_config("key1", None)
        result = manager.get_config("key1")
        assert result is None

        # Test getting non-existent key (should also return None)
        result = manager.get_config("nonexistent")
        assert result is None