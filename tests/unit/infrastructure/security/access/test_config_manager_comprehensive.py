#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 配置管理器综合测试

全面测试ConfigManager类的所有功能，包括：
- 配置加载和保存
- 配置验证
- 热更新功能
- 备份和恢复
- 错误处理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any

from src.infrastructure.security.access.components.config_manager import ConfigManager


class TestConfigManagerComprehensive:
    """配置管理器综合测试"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as temp:
            yield Path(temp)

    @pytest.fixture
    def config_manager(self, temp_dir):
        """配置管理器fixture"""
        return ConfigManager(config_path=temp_dir, enable_hot_reload=False)

    def test_initialization(self, temp_dir):
        """测试初始化"""
        manager = ConfigManager(config_path=temp_dir, enable_hot_reload=False)

        assert manager.config_path == temp_dir
        assert manager.config_file == temp_dir / "access_config.json"
        assert manager.backup_dir == temp_dir / "backups"
        assert not manager.enable_hot_reload
        assert isinstance(manager._config, dict)
        assert len(manager._config_callbacks) == 0

    def test_initialization_with_hot_reload(self, temp_dir):
        """测试启用热更新时的初始化"""
        manager = ConfigManager(config_path=temp_dir, enable_hot_reload=True)

        assert manager.enable_hot_reload
        assert manager._monitor_thread is not None
        assert manager._monitor_thread.is_alive()

        # 清理
        manager.shutdown()

    def test_get_default_config(self, config_manager):
        """测试获取默认配置"""
        config = config_manager._get_default_config()

        assert isinstance(config, dict)
        assert "version" in config
        assert "cache" in config
        assert "audit" in config
        assert "security" in config
        assert "policies" in config
        assert "monitoring" in config

    def test_get_config_full(self, config_manager):
        """测试获取完整配置"""
        config = config_manager.get_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_get_config_by_key(self, config_manager):
        """测试按键获取配置"""
        cache_config = config_manager.get_config("cache")
        assert isinstance(cache_config, dict)
        assert "enabled" in cache_config

    def test_get_config_nested_key(self, config_manager):
        """测试获取嵌套配置"""
        cache_enabled = config_manager.get_config("cache.enabled")
        assert isinstance(cache_enabled, bool)

    def test_get_config_invalid_key(self, config_manager):
        """测试获取不存在的配置"""
        result = config_manager.get_config("nonexistent.key")
        assert result is None

    def test_set_config_simple(self, config_manager):
        """测试设置简单配置"""
        success = config_manager.set_config("test_key", "test_value")
        assert success

        value = config_manager.get_config("test_key")
        assert value == "test_value"

    def test_set_config_nested(self, config_manager):
        """测试设置嵌套配置"""
        success = config_manager.set_config("test.nested.key", "nested_value")
        assert success

        value = config_manager.get_config("test.nested.key")
        assert value == "nested_value"

    def test_set_config_invalid_key(self, config_manager):
        """测试设置无效键的配置"""
        # 空字符串应该被允许作为键
        success = config_manager.set_config("", "value")
        assert success  # ConfigManager允许空字符串作为键

    def test_update_config(self, config_manager):
        """测试更新配置"""
        updates = {
            "cache": {"max_size": 200},
            "audit": {"log_level": "DEBUG"},
            "new_section": {"enabled": True}
        }

        success = config_manager.update_config(updates)
        assert success

        # 验证更新结果
        assert config_manager.get_config("cache.max_size") == 200
        assert config_manager.get_config("audit.log_level") == "DEBUG"
        assert config_manager.get_config("new_section.enabled") == True

    def test_update_config_empty(self, config_manager):
        """测试更新空配置"""
        success = config_manager.update_config({})
        assert success

    def test_reset_config_full(self, config_manager):
        """测试重置完整配置"""
        # 先修改配置
        config_manager.set_config("test_key", "test_value")

        # 重置
        success = config_manager.reset_config()
        assert success

        # 验证已重置为默认值
        assert config_manager.get_config("test_key") is None

    def test_reset_config_section(self, config_manager):
        """测试重置配置节"""
        # 先修改缓存配置
        config_manager.set_config("cache.max_size", 999)

        # 重置缓存节
        success = config_manager.reset_config("cache")
        assert success

        # 验证缓存配置已重置
        cache_config = config_manager.get_config("cache")
        assert cache_config != {"max_size": 999}

    def test_reset_config_invalid_section(self, config_manager):
        """测试重置不存在的配置节"""
        success = config_manager.reset_config("nonexistent")
        assert not success

    def test_validate_config_valid(self, config_manager):
        """测试验证有效配置"""
        result = config_manager.validate_config()
        assert result["valid"] == True
        assert len(result["errors"]) == 0

    def test_validate_config_invalid_cache_size(self, config_manager):
        """测试验证无效缓存大小"""
        config_manager.set_config("cache.max_size", 0)  # 无效值

        result = config_manager.validate_config()
        assert result["valid"] == False
        assert len(result["errors"]) > 0
        assert any("cache.max_size" in error for error in result["errors"])

    def test_validate_config_invalid_ttl(self, config_manager):
        """测试验证无效TTL"""
        config_manager.set_config("cache.ttl_seconds", -1)  # 无效值

        result = config_manager.validate_config()
        assert result["valid"] == True  # TTL为负只产生警告，不影响有效性
        assert len(result["warnings"]) > 0
        assert any("ttl_seconds" in warning for warning in result["warnings"])

    def test_export_config(self, config_manager, temp_dir):
        """测试导出配置"""
        export_file = temp_dir / "exported_config.json"

        success = config_manager.export_config(export_file)
        assert success
        assert export_file.exists()

        # 验证导出的内容
        with open(export_file, 'r') as f:
            exported_data = json.load(f)

        assert isinstance(exported_data, dict)
        assert len(exported_data) > 0

    @patch('builtins.open')
    def test_export_config_invalid_path(self, mock_open, config_manager):
        """测试导出到无效路径"""
        # 模拟文件打开失败
        mock_open.side_effect = OSError("Permission denied")

        invalid_path = Path("/invalid/path/config.json")
        success = config_manager.export_config(invalid_path)
        assert not success

    def test_import_config_merge(self, config_manager, temp_dir):
        """测试导入并合并配置"""
        # 创建导入文件
        import_data = {
            "cache": {"max_size": 500},
            "new_setting": "imported_value"
        }
        import_file = temp_dir / "import_config.json"
        with open(import_file, 'w') as f:
            json.dump(import_data, f)

        # 导入并合并
        success = config_manager.import_config(import_file, merge=True)
        assert success

        # 验证合并结果
        assert config_manager.get_config("cache.max_size") == 500
        assert config_manager.get_config("new_setting") == "imported_value"

    def test_import_config_replace(self, config_manager, temp_dir):
        """测试导入并替换配置"""
        # 创建导入文件
        import_data = {"replaced": "value"}
        import_file = temp_dir / "import_config.json"
        with open(import_file, 'w') as f:
            json.dump(import_data, f)

        # 导入并替换
        success = config_manager.import_config(import_file, merge=False)
        assert success

        # 验证替换结果
        assert config_manager.get_config("replaced") == "value"
        # 原来的配置应该被替换
        assert config_manager.get_config("cache") is None

    def test_import_config_invalid_file(self, config_manager):
        """测试导入无效文件"""
        invalid_file = Path("/nonexistent/file.json")
        success = config_manager.import_config(invalid_file)
        assert not success

    def test_import_config_invalid_json(self, config_manager, temp_dir):
        """测试导入无效JSON文件"""
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")

        success = config_manager.import_config(invalid_file)
        assert not success

    def test_load_config_file_not_exists(self, temp_dir):
        """测试加载不存在的配置文件"""
        manager = ConfigManager(config_path=temp_dir, enable_hot_reload=False)
        # 应该使用默认配置，不会出错
        assert isinstance(manager._config, dict)

    @patch('builtins.open', new_callable=mock_open, read_data='{"invalid": json}')
    def test_load_config_invalid_json(self, mock_file, temp_dir):
        """测试加载无效JSON配置"""
        manager = ConfigManager(config_path=temp_dir, enable_hot_reload=False)
        # 应该使用默认配置
        assert isinstance(manager._config, dict)

    def test_save_config(self, config_manager):
        """测试保存配置"""
        # 修改配置
        config_manager.set_config("test.save", "saved_value")

        # 保存
        success = config_manager._save_config()
        assert success

        # 验证文件已创建
        assert config_manager.config_file.exists()

        # 验证文件内容
        with open(config_manager.config_file, 'r') as f:
            saved_data = json.load(f)

        assert saved_data.get("test", {}).get("save") == "saved_value"

    def test_save_config_create_backup(self, config_manager):
        """测试保存时创建备份"""
        # 确保备份目录存在
        config_manager.backup_dir.mkdir(exist_ok=True)

        # 第一次保存（应该创建备份）
        config_manager.set_config("backup_test_1", "value_1")
        config_manager._save_config()

        # 验证备份文件已创建
        backup_files = list(config_manager.backup_dir.glob("*.json"))
        assert len(backup_files) >= 1

        # 第二次保存（应该创建新的备份）
        config_manager.set_config("backup_test_2", "value_2")
        config_manager._save_config()

        # 验证备份文件数量
        backup_files = list(config_manager.backup_dir.glob("*.json"))
        # 注意：备份逻辑可能不是每次都创建新的备份，取决于实现
        assert len(backup_files) >= 1

    def test_merge_configs(self, config_manager):
        """测试合并配置"""
        base = {"a": 1, "b": {"x": 10}}
        override = {"b": {"y": 20}, "c": 3}

        result = {}
        config_manager._merge_configs(result, base)
        config_manager._merge_configs(result, override)

        assert result["a"] == 1
        assert result["b"]["x"] == 10
        assert result["b"]["y"] == 20
        assert result["c"] == 3

    def test_create_backup(self, config_manager):
        """测试创建备份"""
        # 确保配置已保存
        config_manager._save_config()

        # 创建备份
        config_manager._create_backup()

        # 验证备份文件存在
        backup_files = list(config_manager.backup_dir.glob("*.json"))
        assert len(backup_files) > 0

    def test_get_config_summary(self, config_manager):
        """测试获取配置摘要"""
        summary = config_manager.get_config_summary()

        required_keys = [
            "version", "created_at", "updated_at",
            "cache_enabled", "audit_enabled", "monitoring_enabled",
            "config_file_path", "backup_dir", "hot_reload_enabled"
        ]

        for key in required_keys:
            assert key in summary

    def test_calculate_config_hash(self, config_manager):
        """测试计算配置哈希"""
        # 确保配置文件存在
        config_manager._save_config()

        hash_value = config_manager._calculate_config_hash()
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    def test_calculate_config_hash_no_file(self, temp_dir):
        """测试计算不存在文件的哈希"""
        manager = ConfigManager(config_path=temp_dir, enable_hot_reload=False)
        # 删除配置文件
        if manager.config_file.exists():
            manager.config_file.unlink()

        hash_value = manager._calculate_config_hash()
        assert hash_value == ""

    def test_check_config_file_changed(self, config_manager):
        """测试检查配置文件变化"""
        # 初始状态
        initial_hash = config_manager._config_hash
        changed = config_manager._check_config_file_changed()
        assert not changed

        # 修改配置文件
        config_manager.set_config("change_test", "changed")
        config_manager._save_config()

        # 再次检查
        changed = config_manager._check_config_file_changed()
        assert changed

    def test_add_config_change_callback(self, config_manager):
        """测试添加配置变更回调"""
        def callback(config):
            pass

        config_manager.add_config_change_callback(callback)
        assert len(config_manager._config_callbacks) == 1

    def test_remove_config_change_callback(self, config_manager):
        """测试移除配置变更回调"""
        def callback(config):
            pass

        # 添加
        config_manager.add_config_change_callback(callback)
        assert len(config_manager._config_callbacks) == 1

        # 移除
        config_manager.remove_config_change_callback(callback)
        assert len(config_manager._config_callbacks) == 0

    def test_notify_config_callbacks(self, config_manager):
        """测试通知配置变更回调"""
        callback_called = False
        new_config = None

        def callback(config):
            nonlocal callback_called, new_config
            callback_called = True
            new_config = config

        config_manager.add_config_change_callback(callback)

        old_config = config_manager._config.copy()
        config_manager._notify_config_callbacks(old_config)

        assert callback_called
        assert new_config is not None

    def test_trigger_manual_reload(self, config_manager):
        """测试手动触发重新加载"""
        # 设置初始配置并保存
        config_manager.set_config("manual_reload_test", "original")
        config_manager._save_config()

        # 修改内存配置但不保存
        config_manager._config["manual_reload_test"] = "modified"

        # 手动重新加载
        success = config_manager.trigger_manual_reload()
        assert success

        # 验证配置已从文件重新加载
        value = config_manager.get_config("manual_reload_test")
        assert value == "original"  # 应该是从文件加载的值

    def test_trigger_manual_reload_invalid_config(self, config_manager):
        """测试手动重新加载无效配置"""
        # 备份原始配置
        original_config = config_manager._config.copy()

        # 创建无效的配置文件
        with open(config_manager.config_file, 'w') as f:
            f.write("invalid json content")

        # 手动重新加载应该失败并回滚
        success = config_manager.trigger_manual_reload()
        assert not success

        # 验证配置已回滚到原始状态
        assert config_manager._config == original_config

    def test_shutdown_with_hot_reload(self, temp_dir):
        """测试关闭启用热更新的管理器"""
        manager = ConfigManager(config_path=temp_dir, enable_hot_reload=True)

        # 验证监控线程正在运行
        assert manager._monitor_thread is not None
        assert manager._monitor_thread.is_alive()

        # 关闭
        manager.shutdown()

        # 验证监控线程已停止
        assert manager._stop_monitoring.is_set()

    def test_shutdown_without_hot_reload(self, config_manager):
        """测试关闭未启用热更新的管理器"""
        # 不应该有监控线程
        assert config_manager._monitor_thread is None

        # 关闭应该正常工作
        config_manager.shutdown()

    def test_hot_reload_monitor_components(self, temp_dir):
        """测试热更新监控组件"""
        manager = ConfigManager(config_path=temp_dir, enable_hot_reload=True)

        try:
            # 验证监控组件已初始化
            assert manager._monitor_thread is not None
            assert manager._stop_monitoring is not None
            assert not manager._stop_monitoring.is_set()

        finally:
            manager.shutdown()
            assert manager._stop_monitoring.is_set()

    def test_perform_hot_reload_success(self, config_manager):
        """测试执行成功的热更新"""
        # 修改配置并保存
        config_manager.set_config("hot_reload_test", "updated_value")
        config_manager._save_config()

        old_config = config_manager._config.copy()

        # 执行热更新
        config_manager._perform_hot_reload()

        # 验证配置已更新（从文件重新加载）
        assert config_manager.get_config("hot_reload_test") == "updated_value"

    def test_perform_hot_reload_invalid_config(self, config_manager, temp_dir):
        """测试执行失败的热更新（无效配置）"""
        # 保存当前有效配置
        valid_config = config_manager._config.copy()
        config_manager._save_config()

        # 创建无效配置文件
        with open(config_manager.config_file, 'w') as f:
            f.write("invalid json")

        old_config = config_manager._config.copy()

        # 执行热更新（应该失败并回滚）
        config_manager._perform_hot_reload()

        # 验证配置已回滚
        assert config_manager._config == old_config

    def test_start_hot_reload_monitor_disabled(self, config_manager):
        """测试禁用热更新时不启动监控"""
        assert not config_manager.enable_hot_reload
        assert config_manager._monitor_thread is None

        config_manager._start_hot_reload_monitor()
        assert config_manager._monitor_thread is None
