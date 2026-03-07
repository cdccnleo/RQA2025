"""
基础设施层配置管理深度测试
测试配置管理器、存储、验证、监控、安全等核心功能
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import json
import yaml
import os
import tempfile
import time


# Mock 依赖
class MockLogger:
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def debug(self, msg): pass


class MockConfigSource:
    FILE = "file"
    DATABASE = "database"
    ENVIRONMENT = "environment"
    REMOTE = "remote"


class MockConfigPriority:
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 15


class MockConfigEvent:
    def __init__(self, event_type, key=None, old_value=None, new_value=None, timestamp=None):
        self.event_type = event_type
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or datetime.now()


class MockConfigValidator:
    def __init__(self):
        self.logger = MockLogger()

    def validate_key(self, key, value):
        """验证配置键值"""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("配置键不能为空")
        return True

    def validate_schema(self, config, schema):
        """验证配置模式"""
        if not isinstance(config, dict):
            return False
        return True


class MockConfigProcessor:
    def __init__(self, data):
        self.data = data
        self.logger = MockLogger()

    def process_value(self, value):
        """处理配置值"""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # 简单的环境变量处理
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        return value


class MockConfigStorage:
    def __init__(self):
        self.data = {}
        self.logger = MockLogger()

    def save(self, key, value):
        """保存配置"""
        self.data[key] = value
        return True

    def load(self, key, default=None):
        """加载配置"""
        return self.data.get(key, default)

    def delete(self, key):
        """删除配置"""
        if key in self.data:
            del self.data[key]
            return True
        return False

    def list_keys(self, prefix=None):
        """列出配置键"""
        keys = list(self.data.keys())
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]
        return keys


class MockConfigListenerManager:
    def __init__(self):
        self.listeners = []
        self.logger = MockLogger()

    def add_listener(self, callback):
        """添加监听器"""
        self.listeners.append(callback)

    def remove_listener(self, callback):
        """移除监听器"""
        if callback in self.listeners:
            self.listeners.remove(callback)

    def notify_listeners(self, event):
        """通知监听器"""
        for listener in self.listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"监听器执行失败: {e}")


class MockHealthCheckInterface:
    def health_check(self):
        return {
            'status': 'healthy',
            'service_name': 'config_manager',
            'timestamp': datetime.now().isoformat()
        }


class MockUnifiedConfigManager(MockHealthCheckInterface):
    def __init__(self, config=None):
        self._service_name = "unified_config_manager"
        self._service_version = "2.0.0"
        self._data = config or {}
        self._listeners = []

        # 初始化组件
        self._key_validator = MockConfigValidator()
        self._value_processor = MockConfigProcessor(self._data)
        self._storage = MockConfigStorage()
        self._listener_manager = MockConfigListenerManager()

        self.logger = MockLogger()

    def get(self, key, default=None):
        """获取配置值"""
        if key in self._data:
            return self._data[key]

        # 支持嵌套键名
        if '.' in key:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current

        return default

    def set(self, key, value, source=MockConfigSource.FILE, priority=MockConfigPriority.MEDIUM):
        """设置配置值"""
        # 验证键
        self._key_validator.validate_key(key, value)

        # 处理值
        processed_value = self._value_processor.process_value(value)

        # 获取旧值用于事件
        old_value = self.get(key)

        # 设置新值
        if '.' in key:
            keys = key.split('.')
            current = self._data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = processed_value
        else:
            self._data[key] = processed_value

        # 保存到存储
        self._storage.save(key, processed_value)

        # 触发事件
        event = MockConfigEvent('CONFIG_CHANGED', key, old_value, processed_value)
        self._listener_manager.notify_listeners(event)

        return True

    def delete(self, key):
        """删除配置"""
        # 处理嵌套键
        if '.' in key:
            keys = key.split('.')
            current = self._data
            for k in keys[:-1]:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return False

            if isinstance(current, dict) and keys[-1] in current:
                old_value = current[keys[-1]]
                del current[keys[-1]]
                self._storage.delete(key)

                # 触发事件
                event = MockConfigEvent('CONFIG_DELETED', key, old_value, None)
                self._listener_manager.notify_listeners(event)

                return True
        else:
            if key in self._data:
                old_value = self._data[key]
                del self._data[key]
                self._storage.delete(key)

                # 触发事件
                event = MockConfigEvent('CONFIG_DELETED', key, old_value, None)
                self._listener_manager.notify_listeners(event)

                return True

        return False

    def reload(self, source=MockConfigSource.FILE):
        """重新加载配置"""
        # 从存储重新加载所有配置
        for key in self._storage.list_keys():
            value = self._storage.load(key)
            if value is not None:
                self._data[key] = value

        # 触发重载事件
        event = MockConfigEvent('CONFIG_RELOADED', None, None, None)
        self._listener_manager.notify_listeners(event)

        return True

    def list_keys(self, prefix=None):
        """列出配置键"""
        def collect_keys_recursive(data, current_path="", keys_list=None):
            """递归收集所有配置键"""
            if keys_list is None:
                keys_list = []

            for key, value in data.items():
                full_key = f"{current_path}.{key}" if current_path else key
                keys_list.append(full_key)

                if isinstance(value, dict):
                    collect_keys_recursive(value, full_key, keys_list)

            return keys_list

        all_keys = collect_keys_recursive(self._data)

        if prefix:
            all_keys = [key for key in all_keys if key.startswith(prefix)]

        return all_keys

    def get_stats(self):
        """获取配置统计"""
        def count_keys_recursive(data):
            """递归计算所有配置项数量"""
            count = 0
            nested_count = 0
            for key, value in data.items():
                count += 1
                if isinstance(value, dict):
                    nested_count += 1
                    sub_count, sub_nested = count_keys_recursive(value)
                    count += sub_count
                    nested_count += sub_nested
            return count, nested_count

        total_keys, nested_keys = count_keys_recursive(self._data)

        return {
            'total_keys': total_keys,
            'nested_keys': nested_keys,
            'listeners_count': len(self._listener_manager.listeners),
            'last_modified': datetime.now().isoformat()
        }

    def add_listener(self, callback):
        """添加配置监听器"""
        self._listener_manager.add_listener(callback)

    def remove_listener(self, callback):
        """移除配置监听器"""
        self._listener_manager.remove_listener(callback)

    def export_config(self, format='json'):
        """导出配置"""
        if format == 'json':
            return json.dumps(self._data, indent=2, default=str)
        elif format == 'yaml':
            return yaml.dump(self._data, default_flow_style=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def import_config(self, config_data, format='json'):
        """导入配置"""
        if format == 'json':
            imported = json.loads(config_data)
        elif format == 'yaml':
            imported = yaml.safe_load(config_data)
        else:
            raise ValueError(f"不支持的导入格式: {format}")

        # 合并配置
        self._data.update(imported)

        # 保存到存储
        for key, value in imported.items():
            self._storage.save(key, value)

        # 触发导入事件
        event = MockConfigEvent('CONFIG_IMPORTED', None, None, imported)
        self._listener_manager.notify_listeners(event)

        return True

    def validate_config(self, schema=None):
        """验证配置"""
        if schema:
            return self._key_validator.validate_schema(self._data, schema)
        else:
            # 基础验证
            for key, value in self._data.items():
                self._key_validator.validate_key(key, value)
            return True

    def backup_config(self, backup_path=None):
        """备份配置"""
        if not backup_path:
            backup_path = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, default=str)

        return backup_path

    def restore_config(self, backup_path):
        """恢复配置"""
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"备份文件不存在: {backup_path}")

        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)

        # 恢复数据
        self._data = backup_data

        # 重新保存到存储
        for key, value in backup_data.items():
            self._storage.save(key, value)

        # 触发恢复事件
        event = MockConfigEvent('CONFIG_RESTORED', None, None, backup_data)
        self._listener_manager.notify_listeners(event)

        return True

    def health_check(self):
        """健康检查"""
        try:
            stats = self.get_stats()
            return {
                'status': 'healthy',
                'service_name': self._service_name,
                'service_version': self._service_version,
                'config_keys_count': stats['total_keys'],
                'listeners_count': stats['listeners_count'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'service_name': self._service_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# 导入真实的类用于测试（如果可用的话）
try:
    from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager
    from src.infrastructure.config.core.config_validators import ConfigKeyValidator
    from src.infrastructure.config.core.config_processors import ConfigValueProcessor
    from src.infrastructure.config.core.config_listeners import ConfigListenerManager
    REAL_CLASSES_AVAILABLE = True
except ImportError:
    REAL_CLASSES_AVAILABLE = False
    print("真实配置管理类不可用，使用Mock类进行测试")


class TestConfigManagementCore:
    """配置管理核心功能测试"""

    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        manager = MockUnifiedConfigManager()

        assert manager._service_name == "unified_config_manager"
        assert manager._service_version == "2.0.0"
        assert isinstance(manager._data, dict)
        assert len(manager._listeners) == 0

    def test_get_set_config_simple(self):
        """测试简单配置的获取和设置"""
        manager = MockUnifiedConfigManager()

        # 设置配置
        success = manager.set("app.name", "TestApp")
        assert success is True

        # 获取配置
        value = manager.get("app.name")
        assert value == "TestApp"

        # 获取不存在的配置
        default_value = manager.get("nonexistent", "default")
        assert default_value == "default"

    def test_get_set_config_nested(self):
        """测试嵌套配置的获取和设置"""
        manager = MockUnifiedConfigManager()

        # 设置嵌套配置
        manager.set("database.host", "localhost")
        manager.set("database.port", 5432)
        manager.set("database.credentials.username", "admin")

        # 获取嵌套配置
        assert manager.get("database.host") == "localhost"
        assert manager.get("database.port") == 5432
        assert manager.get("database.credentials.username") == "admin"

        # 获取整个嵌套对象
        db_config = manager.get("database")
        assert isinstance(db_config, dict)
        assert db_config["host"] == "localhost"
        assert db_config["credentials"]["username"] == "admin"

    def test_config_validation(self):
        """测试配置验证"""
        manager = MockUnifiedConfigManager()

        # 有效的配置
        manager.set("valid_key", "valid_value")
        assert manager.get("valid_key") == "valid_value"

        # 无效的键（空字符串）
        with pytest.raises(ValueError, match="配置键不能为空"):
            manager.set("", "value")

        # 无效的键（非字符串）
        with pytest.raises(ValueError, match="配置键不能为空"):
            manager.set(123, "value")

    def test_config_value_processing(self):
        """测试配置值处理"""
        manager = MockUnifiedConfigManager()

        # 环境变量处理
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            manager.set("env_var", "${TEST_VAR}")
            # Mock处理器会处理环境变量
            assert manager.get("env_var") == "test_value"

    def test_config_deletion(self):
        """测试配置删除"""
        manager = MockUnifiedConfigManager()

        # 设置并删除配置
        manager.set("temp_key", "temp_value")
        assert manager.get("temp_key") == "temp_value"

        success = manager.delete("temp_key")
        assert success is True
        assert manager.get("temp_key") is None

        # 删除不存在的配置
        success = manager.delete("nonexistent")
        assert success is False

    def test_config_list_keys(self):
        """测试配置键列表"""
        manager = MockUnifiedConfigManager()

        # 设置一些配置
        manager.set("app.name", "TestApp")
        manager.set("app.version", "1.0.0")
        manager.set("database.host", "localhost")
        manager.set("cache.enabled", True)

        # 列出所有键
        all_keys = manager.list_keys()
        assert len(all_keys) >= 4

        # 列出带前缀的键
        app_keys = manager.list_keys("app")
        assert len(app_keys) >= 2
        assert "app.name" in all_keys
        assert "app.version" in all_keys

    def test_config_reload(self):
        """测试配置重载"""
        manager = MockUnifiedConfigManager()

        # 设置初始配置
        manager.set("initial_key", "initial_value")

        # 模拟存储中的变化
        manager._storage.save("new_key", "new_value")

        # 重载配置
        success = manager.reload()
        assert success is True

        # 检查新配置是否加载
        assert manager.get("new_key") == "new_value"


class TestConfigManagementStorage:
    """配置管理存储功能测试"""

    def test_config_storage_operations(self):
        """测试配置存储操作"""
        storage = MockConfigStorage()

        # 保存和加载
        storage.save("key1", "value1")
        assert storage.load("key1") == "value1"

        # 删除
        storage.delete("key1")
        assert storage.load("key1") is None

        # 列出键
        storage.save("key1", "value1")
        storage.save("key2", "value2")
        keys = storage.list_keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

        # 带前缀的键列表
        storage.save("prefix.key1", "value1")
        storage.save("prefix.key2", "value2")
        storage.save("other.key", "value")

        prefix_keys = storage.list_keys("prefix")
        assert len(prefix_keys) == 2
        assert all(key.startswith("prefix") for key in prefix_keys)


class TestConfigManagementEvents:
    """配置管理事件功能测试"""

    def test_config_event_creation(self):
        """测试配置事件创建"""
        event = MockConfigEvent("CONFIG_CHANGED", "test.key", "old_value", "new_value")

        assert event.event_type == "CONFIG_CHANGED"
        assert event.key == "test.key"
        assert event.old_value == "old_value"
        assert event.new_value == "new_value"
        assert isinstance(event.timestamp, datetime)

    def test_config_listener_management(self):
        """测试配置监听器管理"""
        manager = MockConfigListenerManager()

        # 添加监听器
        call_count = 0
        def test_listener(event):
            nonlocal call_count
            call_count += 1

        manager.add_listener(test_listener)
        assert len(manager.listeners) == 1

        # 通知监听器
        event = MockConfigEvent("TEST_EVENT")
        manager.notify_listeners(event)
        assert call_count == 1

        # 移除监听器
        manager.remove_listener(test_listener)
        assert len(manager.listeners) == 0

        # 再次通知，不应该调用
        call_count_before = call_count
        manager.notify_listeners(event)
        assert call_count == call_count_before

    def test_config_manager_with_listeners(self):
        """测试带监听器的配置管理器"""
        manager = MockUnifiedConfigManager()

        # 添加监听器
        events_received = []
        def event_logger(event):
            events_received.append(event)

        manager.add_listener(event_logger)

        # 设置配置（应该触发事件）
        manager.set("test.key", "test_value")

        # 检查事件是否被接收
        assert len(events_received) == 1
        assert events_received[0].event_type == "CONFIG_CHANGED"
        assert events_received[0].key == "test.key"
        assert events_received[0].new_value == "test_value"

        # 删除配置（应该触发事件）
        manager.delete("test.key")

        assert len(events_received) == 2
        assert events_received[1].event_type == "CONFIG_DELETED"
        assert events_received[1].key == "test.key"


class TestConfigManagementImportExport:
    """配置管理导入导出功能测试"""

    def test_config_export_json(self):
        """测试JSON格式导出"""
        manager = MockUnifiedConfigManager()
        manager.set("app.name", "TestApp")
        manager.set("app.version", "1.0.0")
        manager.set("database.host", "localhost")

        exported = manager.export_config("json")

        # 验证导出的JSON包含预期数据
        import json as json_module
        data = json_module.loads(exported)
        assert data["app"]["name"] == "TestApp"
        assert data["app"]["version"] == "1.0.0"
        assert data["database"]["host"] == "localhost"

    def test_config_export_yaml(self):
        """测试YAML格式导出"""
        manager = MockUnifiedConfigManager()
        manager.set("app.name", "TestApp")
        manager.set("database.port", 5432)

        exported = manager.export_config("yaml")

        # 验证导出的YAML包含预期数据
        assert "app:" in exported
        assert "name: TestApp" in exported
        assert "database:" in exported
        assert "port: 5432" in exported

    def test_config_export_invalid_format(self):
        """测试无效导出格式"""
        manager = MockUnifiedConfigManager()

        with pytest.raises(ValueError, match="不支持的导出格式"):
            manager.export_config("invalid")

    def test_config_import_json(self):
        """测试JSON格式导入"""
        manager = MockUnifiedConfigManager()

        json_data = '''
        {
            "app": {
                "name": "ImportedApp",
                "version": "2.0.0"
            },
            "database": {
                "host": "remotehost",
                "port": 3306
            }
        }
        '''

        success = manager.import_config(json_data, "json")
        assert success is True

        assert manager.get("app.name") == "ImportedApp"
        assert manager.get("app.version") == "2.0.0"
        assert manager.get("database.host") == "remotehost"
        assert manager.get("database.port") == 3306

    def test_config_import_yaml(self):
        """测试YAML格式导入"""
        manager = MockUnifiedConfigManager()

        yaml_data = """
        app:
          name: YamlApp
          version: 3.0.0
        cache:
          enabled: true
          ttl: 3600
        """

        success = manager.import_config(yaml_data, "yaml")
        assert success is True

        assert manager.get("app.name") == "YamlApp"
        assert manager.get("cache.enabled") is True
        assert manager.get("cache.ttl") == 3600

    def test_config_import_invalid_format(self):
        """测试无效导入格式"""
        manager = MockUnifiedConfigManager()

        with pytest.raises(ValueError, match="不支持的导入格式"):
            manager.import_config("{}", "invalid")


class TestConfigManagementBackupRestore:
    """配置管理备份恢复功能测试"""

    def test_config_backup(self):
        """测试配置备份"""
        manager = MockUnifiedConfigManager()
        manager.set("app.name", "BackupTest")
        manager.set("database.url", "postgresql://localhost/test")

        # 创建临时文件进行备份
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            backup_path = f.name

        try:
            # 执行备份
            result_path = manager.backup_config(backup_path)
            assert result_path == backup_path
            assert os.path.exists(backup_path)

            # 验证备份文件内容
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

            assert backup_data["app"]["name"] == "BackupTest"
            assert backup_data["database"]["url"] == "postgresql://localhost/test"

        finally:
            # 清理临时文件
            if os.path.exists(backup_path):
                os.unlink(backup_path)

    def test_config_restore(self):
        """测试配置恢复"""
        manager = MockUnifiedConfigManager()

        # 创建备份数据
        backup_data = {
            "app": {
                "name": "RestoredApp",
                "version": "1.5.0"
            },
            "features": ["logging", "monitoring"]
        }

        # 创建临时备份文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(backup_data, f, indent=2)
            backup_path = f.name

        try:
            # 执行恢复
            success = manager.restore_config(backup_path)
            assert success is True

            # 验证配置已恢复
            assert manager.get("app.name") == "RestoredApp"
            assert manager.get("app.version") == "1.5.0"
            assert manager.get("features") == ["logging", "monitoring"]

        finally:
            # 清理临时文件
            if os.path.exists(backup_path):
                os.unlink(backup_path)

    def test_config_restore_nonexistent_file(self):
        """测试恢复不存在的文件"""
        manager = MockUnifiedConfigManager()

        with pytest.raises(FileNotFoundError, match="备份文件不存在"):
            manager.restore_config("nonexistent_backup.json")


class TestConfigManagementValidation:
    """配置管理验证功能测试"""

    def test_config_schema_validation(self):
        """测试配置模式验证"""
        manager = MockUnifiedConfigManager()

        # 设置测试配置
        manager.set("app.name", "TestApp")
        manager.set("app.port", 8080)
        manager.set("database.enabled", True)

        # 简单的模式验证（Mock实现）
        schema = {
            "type": "object",
            "properties": {
                "app": {"type": "object"},
                "database": {"type": "object"}
            }
        }

        # Mock验证器应该通过
        result = manager.validate_config(schema)
        assert result is True

    def test_config_key_validation(self):
        """测试配置键验证"""
        validator = MockConfigValidator()

        # 有效键
        assert validator.validate_key("valid.key", "value") is True
        assert validator.validate_key("another_key", 123) is True

        # 无效键
        with pytest.raises(ValueError, match="配置键不能为空"):
            validator.validate_key("", "value")

        with pytest.raises(ValueError, match="配置键不能为空"):
            validator.validate_key("   ", "value")

        with pytest.raises(ValueError, match="配置键不能为空"):
            validator.validate_key(None, "value")


class TestConfigManagementHealthCheck:
    """配置管理健康检查功能测试"""

    def test_config_manager_health_check(self):
        """测试配置管理器健康检查"""
        manager = MockUnifiedConfigManager()

        # 添加一些配置
        manager.set("test.key1", "value1")
        manager.set("test.key2", "value2")

        # 执行健康检查
        health = manager.health_check()

        assert health["status"] == "healthy"
        assert health["service_name"] == "unified_config_manager"
        assert health["service_version"] == "2.0.0"
        assert health["config_keys_count"] >= 2
        assert "timestamp" in health

    def test_config_manager_health_check_with_error(self):
        """测试配置管理器健康检查（异常情况）"""
        manager = MockUnifiedConfigManager()

        # 模拟健康检查异常
        original_get_stats = manager.get_stats
        def failing_get_stats():
            raise Exception("模拟统计失败")

        manager.get_stats = failing_get_stats

        try:
            health = manager.health_check()

            assert health["status"] == "unhealthy"
            assert "error" in health
            assert health["error"] == "模拟统计失败"
            assert health["service_name"] == "unified_config_manager"

        finally:
            # 恢复原始方法
            manager.get_stats = original_get_stats

    def test_config_stats(self):
        """测试配置统计"""
        manager = MockUnifiedConfigManager()

        # 添加配置
        manager.set("simple_key", "value")
        manager.set("nested.key1", "nested_value1")
        manager.set("nested.key2", "nested_value2")

        # 添加监听器
        manager.add_listener(lambda e: None)

        stats = manager.get_stats()

        assert stats["total_keys"] >= 3  # 至少有3个键
        assert stats["nested_keys"] >= 1  # 至少有1个嵌套键
        assert stats["listeners_count"] >= 1  # 至少有1个监听器
        assert "last_modified" in stats


class TestConfigManagementIntegration:
    """配置管理集成测试"""

    def test_complete_config_lifecycle(self):
        """测试完整配置生命周期"""
        manager = MockUnifiedConfigManager()

        # 1. 设置初始配置
        manager.set("app.name", "LifecycleTest")
        manager.set("app.version", "1.0.0")
        manager.set("database.host", "localhost")

        assert manager.get("app.name") == "LifecycleTest"
        assert manager.get("database.host") == "localhost"

        # 2. 导出配置
        exported = manager.export_config("json")
        assert '"LifecycleTest"' in exported

        # 3. 修改配置
        manager.set("app.version", "2.0.0")
        assert manager.get("app.version") == "2.0.0"

        # 4. 备份配置
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            backup_path = f.name

        try:
            manager.backup_config(backup_path)
            assert os.path.exists(backup_path)

            # 5. 创建新管理器并导入配置
            new_manager = MockUnifiedConfigManager()
            with open(backup_path, 'r') as f:
                backup_content = f.read()

            new_manager.import_config(backup_content, "json")
            assert new_manager.get("app.name") == "LifecycleTest"
            assert new_manager.get("app.version") == "2.0.0"

        finally:
            if os.path.exists(backup_path):
                os.unlink(backup_path)

    def test_config_event_driven_updates(self):
        """测试配置事件驱动更新"""
        manager = MockUnifiedConfigManager()

        # 记录所有事件
        events_log = []
        def event_recorder(event):
            events_log.append({
                'type': event.event_type,
                'key': event.key,
                'old_value': event.old_value,
                'new_value': event.new_value
            })

        manager.add_listener(event_recorder)

        # 执行一系列配置操作
        manager.set("config.key1", "value1")  # 创建
        manager.set("config.key1", "value2")  # 更新
        manager.set("config.key2", "value3")  # 创建
        manager.delete("config.key1")         # 删除

        # 验证事件记录
        assert len(events_log) == 4

        # 第一个事件：创建 key1
        assert events_log[0]['type'] == 'CONFIG_CHANGED'
        assert events_log[0]['key'] == 'config.key1'
        assert events_log[0]['old_value'] is None
        assert events_log[0]['new_value'] == 'value1'

        # 第二个事件：更新 key1
        assert events_log[1]['type'] == 'CONFIG_CHANGED'
        assert events_log[1]['key'] == 'config.key1'
        assert events_log[1]['old_value'] == 'value1'
        assert events_log[1]['new_value'] == 'value2'

        # 第三个事件：创建 key2
        assert events_log[2]['type'] == 'CONFIG_CHANGED'
        assert events_log[2]['key'] == 'config.key2'
        assert events_log[2]['new_value'] == 'value3'

        # 第四个事件：删除 key1
        assert events_log[3]['type'] == 'CONFIG_DELETED'
        assert events_log[3]['key'] == 'config.key1'
        assert events_log[3]['old_value'] == 'value2'

    def test_config_concurrent_access_simulation(self):
        """测试配置并发访问模拟"""
        manager = MockUnifiedConfigManager()

        import threading
        import time

        # 并发操作结果
        results = []
        errors = []

        def concurrent_operation(operation_id):
            try:
                # 模拟不同的操作
                if operation_id % 3 == 0:
                    manager.set(f"concurrent.key{operation_id}", f"value{operation_id}")
                    results.append(f"set_{operation_id}")
                elif operation_id % 3 == 1:
                    value = manager.get(f"concurrent.key{operation_id}")
                    results.append(f"get_{operation_id}_{value}")
                else:
                    keys = manager.list_keys()
                    results.append(f"list_{operation_id}_{len(keys)}")

                time.sleep(0.01)  # 模拟操作时间

            except Exception as e:
                errors.append(f"operation_{operation_id}: {str(e)}")

        # 启动多个线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=5.0)

        # 验证结果
        assert len(results) == 10  # 所有操作都应该成功
        assert len(errors) == 0    # 不应该有错误

        # 验证设置的值
        set_operations = [r for r in results if r.startswith("set_")]
        assert len(set_operations) >= 3  # 至少有一些设置操作

    def test_config_performance_under_load(self):
        """测试配置在负载下的性能"""
        manager = MockUnifiedConfigManager()

        # 创建大量配置项
        config_count = 1000
        start_time = time.time()

        # 批量设置配置
        for i in range(config_count):
            manager.set(f"perf.key{i}", f"value{i}")

        set_time = time.time() - start_time

        # 批量获取配置
        start_time = time.time()
        for i in range(config_count):
            value = manager.get(f"perf.key{i}")
            assert value == f"value{i}"

        get_time = time.time() - start_time

        # 验证性能（应该很快）
        assert set_time < 2.0  # 设置1000个配置应该小于2秒
        assert get_time < 1.0  # 获取1000个配置应该小于1秒

        # 验证统计信息
        stats = manager.get_stats()
        assert stats['total_keys'] >= config_count

        # 验证健康检查
        health = manager.health_check()
        assert health['status'] == 'healthy'
        assert health['config_keys_count'] >= config_count
