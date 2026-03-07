"""
配置存储功能测试 (独立版本)
测试配置存储、监控和事件功能，避免复杂导入
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import tempfile
import os
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


class ConfigStorage:
    """配置存储"""

    def __init__(self):
        self._storage: Dict[str, ConfigItem] = {}
        self._listeners: List[callable] = []

    def store(self, key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL) -> None:
        """存储配置项"""
        item = ConfigItem(key=key, value=value, scope=scope)
        self._storage[key] = item
        self._notify_change(key, value, "stored")

    def retrieve(self, key: str) -> Optional[Any]:
        """检索配置项"""
        item = self._storage.get(key)
        return item.value if item else None

    def delete(self, key: str) -> bool:
        """删除配置项"""
        if key in self._storage:
            del self._storage[key]
            self._notify_change(key, None, "deleted")
            return True
        return False

    def exists(self, key: str) -> bool:
        """检查配置项是否存在"""
        return key in self._storage

    def list_keys(self, scope: Optional[ConfigScope] = None) -> List[str]:
        """列出配置键"""
        if scope is None:
            return list(self._storage.keys())
        return [k for k, v in self._storage.items() if v.scope == scope]

    def clear(self, scope: Optional[ConfigScope] = None) -> int:
        """清空配置"""
        if scope is None:
            count = len(self._storage)
            self._storage.clear()
            return count
        else:
            keys_to_delete = [k for k, v in self._storage.items() if v.scope == scope]
            for key in keys_to_delete:
                del self._storage[key]
            return len(keys_to_delete)

    def add_change_listener(self, listener: callable) -> None:
        """添加变更监听器"""
        self._listeners.append(listener)

    def remove_change_listener(self, listener: callable) -> None:
        """移除变更监听器"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_change(self, key: str, value: Any, action: str) -> None:
        """通知变更"""
        for listener in self._listeners:
            try:
                listener(key, value, action)
            except Exception:
                pass  # Ignore listener errors

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._storage)
        scope_counts = {}
        for scope in ConfigScope:
            scope_counts[scope.value] = len([v for v in self._storage.values() if v.scope == scope])

        return {
            "total_items": total,
            "global_scope": scope_counts["global"],
            "user_scope": scope_counts["user"],
            "session_scope": scope_counts["session"],
            "application_scope": scope_counts["application"],
            "listeners_count": len(self._listeners)
        }


class ConfigMonitor:
    """配置监控器"""

    def __init__(self):
        self._access_log: List[Dict[str, Any]] = []
        self._change_log: List[Dict[str, Any]] = []
        self._max_log_size = 1000

    def log_access(self, key: str, action: str, user: Optional[str] = None) -> None:
        """记录访问日志"""
        log_entry = {
            "timestamp": self._get_timestamp(),
            "key": key,
            "action": action,
            "user": user or "system"
        }
        self._access_log.append(log_entry)
        self._trim_log(self._access_log)

    def log_change(self, key: str, old_value: Any, new_value: Any, user: Optional[str] = None) -> None:
        """记录变更日志"""
        log_entry = {
            "timestamp": self._get_timestamp(),
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "user": user or "system"
        }
        self._change_log.append(log_entry)
        self._trim_log(self._change_log)

    def get_access_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取访问历史"""
        logs = self._access_log
        if key:
            logs = [log for log in logs if log["key"] == key]
        return logs[-limit:]

    def get_change_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取变更历史"""
        logs = self._change_log
        if key:
            logs = [log for log in logs if log["key"] == key]
        return logs[-limit:]

    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """获取最近活动统计"""
        import time
        cutoff_time = time.time() - (hours * 3600)

        recent_access = [log for log in self._access_log if log["timestamp"] > cutoff_time]
        recent_changes = [log for log in self._change_log if log["timestamp"] > cutoff_time]

        return {
            "hours": hours,
            "access_count": len(recent_access),
            "change_count": len(recent_changes),
            "unique_keys_accessed": len(set(log["key"] for log in recent_access)),
            "unique_keys_changed": len(set(log["key"] for log in recent_changes))
        }

    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()

    def _trim_log(self, log_list: List[Dict[str, Any]]) -> None:
        """修剪日志大小"""
        if len(log_list) > self._max_log_size:
            log_list[:] = log_list[-self._max_log_size:]


class TestConfigStorage:
    """配置存储测试"""

    def setup_method(self):
        """测试前准备"""
        self.storage = ConfigStorage()

    def test_store_and_retrieve(self):
        """测试存储和检索"""
        # Store different types of values
        self.storage.store("string_key", "string_value")
        self.storage.store("int_key", 42)
        self.storage.store("bool_key", True)
        self.storage.store("list_key", [1, 2, 3])

        # Retrieve and verify
        assert self.storage.retrieve("string_key") == "string_value"
        assert self.storage.retrieve("int_key") == 42
        assert self.storage.retrieve("bool_key") is True
        assert self.storage.retrieve("list_key") == [1, 2, 3]
        assert self.storage.retrieve("nonexistent") is None

    def test_store_with_scope(self):
        """测试带作用域的存储"""
        self.storage.store("global_key", "global_value", ConfigScope.GLOBAL)
        self.storage.store("user_key", "user_value", ConfigScope.USER)
        self.storage.store("app_key", "app_value", ConfigScope.APPLICATION)

        # Verify scopes are stored
        assert self.storage.retrieve("global_key") == "global_value"
        assert self.storage.retrieve("user_key") == "user_value"
        assert self.storage.retrieve("app_key") == "app_value"

    def test_exists_and_delete(self):
        """测试存在性检查和删除"""
        self.storage.store("test_key", "test_value")

        # Check exists
        assert self.storage.exists("test_key") is True
        assert self.storage.exists("nonexistent") is False

        # Delete existing
        assert self.storage.delete("test_key") is True
        assert self.storage.exists("test_key") is False
        assert self.storage.retrieve("test_key") is None

        # Delete non-existing
        assert self.storage.delete("nonexistent") is False

    def test_list_keys(self):
        """测试键列表"""
        # Store items with different scopes
        self.storage.store("global1", "val1", ConfigScope.GLOBAL)
        self.storage.store("global2", "val2", ConfigScope.GLOBAL)
        self.storage.store("user1", "val3", ConfigScope.USER)
        self.storage.store("app1", "val4", ConfigScope.APPLICATION)

        # List all keys
        all_keys = self.storage.list_keys()
        assert len(all_keys) == 4
        assert "global1" in all_keys
        assert "user1" in all_keys

        # List keys by scope
        global_keys = self.storage.list_keys(ConfigScope.GLOBAL)
        assert len(global_keys) == 2
        assert "global1" in global_keys
        assert "global2" in global_keys

        user_keys = self.storage.list_keys(ConfigScope.USER)
        assert len(user_keys) == 1
        assert "user1" in user_keys

    def test_clear_storage(self):
        """测试清空存储"""
        # Store items
        self.storage.store("global1", "val1", ConfigScope.GLOBAL)
        self.storage.store("user1", "val2", ConfigScope.USER)
        self.storage.store("app1", "val3", ConfigScope.APPLICATION)

        # Clear all
        count = self.storage.clear()
        assert count == 3
        assert len(self.storage.list_keys()) == 0

        # Store again and clear by scope
        self.storage.store("global1", "val1", ConfigScope.GLOBAL)
        self.storage.store("global2", "val2", ConfigScope.GLOBAL)
        self.storage.store("user1", "val3", ConfigScope.USER)

        # Clear only global scope
        count = self.storage.clear(ConfigScope.GLOBAL)
        assert count == 2
        assert len(self.storage.list_keys()) == 1  # Only user key remains

    def test_change_listeners(self):
        """测试变更监听器"""
        events = []

        def listener(key, value, action):
            events.append((key, value, action))

        # Add listener
        self.storage.add_change_listener(listener)
        assert len(self.storage._listeners) == 1

        # Store item (should trigger listener)
        self.storage.store("test_key", "test_value")
        assert len(events) == 1
        assert events[0] == ("test_key", "test_value", "stored")

        # Delete item (should trigger listener)
        self.storage.delete("test_key")
        assert len(events) == 2
        assert events[1] == ("test_key", None, "deleted")

        # Remove listener
        self.storage.remove_change_listener(listener)
        assert len(self.storage._listeners) == 0

        # Store again (should not trigger)
        self.storage.store("test_key2", "value2")
        assert len(events) == 2  # No new events

    def test_get_stats(self):
        """测试统计信息"""
        # Initially empty
        stats = self.storage.get_stats()
        assert stats["total_items"] == 0
        assert stats["listeners_count"] == 0

        # Add items
        self.storage.store("global1", "val1", ConfigScope.GLOBAL)
        self.storage.store("user1", "val2", ConfigScope.USER)
        self.storage.store("app1", "val3", ConfigScope.APPLICATION)
        self.storage.store("app2", "val4", ConfigScope.APPLICATION)

        # Add listener
        self.storage.add_change_listener(lambda k, v, a: None)

        stats = self.storage.get_stats()
        assert stats["total_items"] == 4
        assert stats["global_scope"] == 1
        assert stats["user_scope"] == 1
        assert stats["application_scope"] == 2
        assert stats["session_scope"] == 0
        assert stats["listeners_count"] == 1


class TestConfigMonitor:
    """配置监控测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ConfigMonitor()

    def test_log_access(self):
        """测试访问日志"""
        # Log access events
        self.monitor.log_access("config.key1", "read", "user1")
        self.monitor.log_access("config.key2", "write", "user2")
        self.monitor.log_access("config.key1", "read", "user1")

        # Get access history
        history = self.monitor.get_access_history()
        assert len(history) == 3

        # Check specific key history
        key1_history = self.monitor.get_access_history("config.key1")
        assert len(key1_history) == 2
        assert all(log["key"] == "config.key1" for log in key1_history)

    def test_log_change(self):
        """测试变更日志"""
        # Log change events
        self.monitor.log_change("db.host", "old_host", "new_host", "admin")
        self.monitor.log_change("db.port", 3306, 5432, "admin")

        # Get change history
        history = self.monitor.get_change_history()
        assert len(history) == 2

        # Verify change details
        host_change = history[0]
        assert host_change["key"] == "db.host"
        assert host_change["old_value"] == "old_host"
        assert host_change["new_value"] == "new_host"
        assert host_change["user"] == "admin"

    def test_recent_activity(self):
        """测试最近活动统计"""
        # Log some activities
        self.monitor.log_access("key1", "read")
        self.monitor.log_access("key2", "write")
        self.monitor.log_change("key3", "old", "new")

        # Get recent activity (should include all since they're recent)
        activity = self.monitor.get_recent_activity(hours=1)
        assert activity["access_count"] == 2
        assert activity["change_count"] == 1
        assert activity["unique_keys_accessed"] == 2
        assert activity["unique_keys_changed"] == 1

    def test_log_size_limit(self):
        """测试日志大小限制"""
        # Fill log beyond limit
        for i in range(1100):  # Max is 1000
            self.monitor.log_access(f"key{i}", "read")

        # Should be trimmed to max size
        history = self.monitor.get_access_history()
        assert len(history) <= 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
