"""
配置管理器功能测试
测试配置管理器的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ConfigScope(Enum):
    """配置作用域枚举"""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    APPLICATION = "application"


@dataclass
class ConfigItem:
    """配置项"""
    key: str
    value: Any
    scope: ConfigScope = ConfigScope.GLOBAL
    description: Optional[str] = None


class ConfigError(Exception):
    """配置系统基础异常"""
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.config_key = config_key
        self.details = details or {}


class ConfigManager:
    """配置管理器"""

    def __init__(self):
        self._configs: Dict[str, ConfigItem] = {}
        self._listeners: List[callable] = []

    def set_config(self, key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL,
                   description: Optional[str] = None) -> None:
        """设置配置项"""
        if not key or not isinstance(key, str):
            raise ConfigError("Invalid config key", config_key=key)

        item = ConfigItem(key=key, value=value, scope=scope, description=description)
        self._configs[key] = item

        # Notify listeners
        self._notify_listeners(key, value, "set")

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        item = self._configs.get(key)
        return item.value if item else default

    def has_config(self, key: str) -> bool:
        """检查配置项是否存在"""
        return key in self._configs

    def delete_config(self, key: str) -> bool:
        """删除配置项"""
        if key in self._configs:
            del self._configs[key]
            self._notify_listeners(key, None, "delete")
            return True
        return False

    def list_configs(self, scope: Optional[ConfigScope] = None) -> Dict[str, Any]:
        """列出配置项"""
        if scope is None:
            return {k: v.value for k, v in self._configs.items()}
        else:
            return {k: v.value for k, v in self._configs.items() if v.scope == scope}

    def clear_configs(self, scope: Optional[ConfigScope] = None) -> int:
        """清空配置项"""
        if scope is None:
            count = len(self._configs)
            self._configs.clear()
            return count
        else:
            keys_to_delete = [k for k, v in self._configs.items() if v.scope == scope]
            for key in keys_to_delete:
                del self._configs[key]
            return len(keys_to_delete)

    def add_listener(self, listener: callable) -> None:
        """添加配置变更监听器"""
        self._listeners.append(listener)

    def remove_listener(self, listener: callable) -> None:
        """移除配置变更监听器"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self, key: str, value: Any, action: str) -> None:
        """通知监听器"""
        for listener in self._listeners:
            try:
                listener(key, value, action)
            except Exception:
                # Ignore listener errors
                pass

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        total = len(self._configs)
        scope_counts = {}
        for scope in ConfigScope:
            scope_counts[scope.value] = len([v for v in self._configs.values() if v.scope == scope])

        return {
            "total_configs": total,
            "global_configs": scope_counts["global"],
            "user_configs": scope_counts["user"],
            "session_configs": scope_counts["session"],
            "application_configs": scope_counts["application"],
            "listeners_count": len(self._listeners)
        }


class TestConfigManager:
    """配置管理器测试"""

    def setup_method(self):
        """测试前准备"""
        self.manager = ConfigManager()

    def test_set_and_get_config(self):
        """测试设置和获取配置"""
        # Set config
        self.manager.set_config("database.host", "localhost")
        self.manager.set_config("database.port", 5432, scope=ConfigScope.APPLICATION)

        # Get config
        assert self.manager.get_config("database.host") == "localhost"
        assert self.manager.get_config("database.port") == 5432
        assert self.manager.get_config("nonexistent", "default") == "default"

    def test_set_config_validation(self):
        """测试配置设置验证"""
        # Invalid key
        with pytest.raises(ConfigError):
            self.manager.set_config("", "value")

        with pytest.raises(ConfigError):
            self.manager.set_config(None, "value")

    def test_has_config(self):
        """测试配置存在性检查"""
        self.manager.set_config("test.key", "value")
        assert self.manager.has_config("test.key") is True
        assert self.manager.has_config("nonexistent") is False

    def test_delete_config(self):
        """测试删除配置"""
        self.manager.set_config("test.key", "value")
        assert self.manager.has_config("test.key") is True

        # Delete existing config
        assert self.manager.delete_config("test.key") is True
        assert self.manager.has_config("test.key") is False

        # Delete non-existing config
        assert self.manager.delete_config("nonexistent") is False

    def test_list_configs(self):
        """测试列出配置"""
        # Set configs with different scopes
        self.manager.set_config("global.key", "global_value", ConfigScope.GLOBAL)
        self.manager.set_config("user.key", "user_value", ConfigScope.USER)
        self.manager.set_config("app.key", "app_value", ConfigScope.APPLICATION)

        # List all configs
        all_configs = self.manager.list_configs()
        assert len(all_configs) == 3
        assert all_configs["global.key"] == "global_value"

        # List configs by scope
        user_configs = self.manager.list_configs(ConfigScope.USER)
        assert len(user_configs) == 1
        assert user_configs["user.key"] == "user_value"

        app_configs = self.manager.list_configs(ConfigScope.APPLICATION)
        assert len(app_configs) == 1
        assert app_configs["app.key"] == "app_value"

    def test_clear_configs(self):
        """测试清空配置"""
        # Set configs
        self.manager.set_config("global.key", "value1", ConfigScope.GLOBAL)
        self.manager.set_config("user.key", "value2", ConfigScope.USER)
        self.manager.set_config("app.key", "value3", ConfigScope.APPLICATION)

        # Clear all configs
        count = self.manager.clear_configs()
        assert count == 3
        assert len(self.manager.list_configs()) == 0

        # Set configs again and clear by scope
        self.manager.set_config("global1.key", "value1", ConfigScope.GLOBAL)
        self.manager.set_config("global2.key", "value2", ConfigScope.GLOBAL)
        self.manager.set_config("user.key", "value3", ConfigScope.USER)

        # Clear only global configs
        count = self.manager.clear_configs(ConfigScope.GLOBAL)
        assert count == 2
        assert len(self.manager.list_configs()) == 1  # Only user config remains

    def test_listeners(self):
        """测试配置变更监听器"""
        events = []

        def listener(key, value, action):
            events.append((key, value, action))

        # Add listener
        self.manager.add_listener(listener)
        assert len(self.manager._listeners) == 1

        # Set config (should trigger listener)
        self.manager.set_config("test.key", "test_value")
        assert len(events) == 1
        assert events[0] == ("test.key", "test_value", "set")

        # Delete config (should trigger listener)
        self.manager.delete_config("test.key")
        assert len(events) == 2
        assert events[1] == ("test.key", None, "delete")

        # Remove listener
        self.manager.remove_listener(listener)
        assert len(self.manager._listeners) == 0

        # Set config again (should not trigger listener)
        self.manager.set_config("test.key2", "value2")
        assert len(events) == 2  # No new events

    def test_listener_error_handling(self):
        """测试监听器错误处理"""
        def bad_listener(key, value, action):
            raise Exception("Listener error")

        def good_listener(key, value, action):
            pass

        # Add both listeners
        self.manager.add_listener(bad_listener)
        self.manager.add_listener(good_listener)

        # Set config (bad listener should not break good listener)
        self.manager.set_config("test.key", "value")
        # Should not raise exception

    def test_get_stats(self):
        """测试统计信息获取"""
        # Initially empty
        stats = self.manager.get_stats()
        assert stats["total_configs"] == 0
        assert stats["listeners_count"] == 0

        # Set some configs
        self.manager.set_config("global.key", "value1", ConfigScope.GLOBAL)
        self.manager.set_config("user.key", "value2", ConfigScope.USER)
        self.manager.set_config("app1.key", "value3", ConfigScope.APPLICATION)
        self.manager.set_config("app2.key", "value4", ConfigScope.APPLICATION)

        # Add listener
        self.manager.add_listener(lambda k, v, a: None)

        stats = self.manager.get_stats()
        assert stats["total_configs"] == 4
        assert stats["global_configs"] == 1
        assert stats["user_configs"] == 1
        assert stats["application_configs"] == 2
        assert stats["session_configs"] == 0
        assert stats["listeners_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
