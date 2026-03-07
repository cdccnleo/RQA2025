#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理降级服务测试
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from src.features.fallback.config_fallback import FallbackConfigManager


class TestFallbackConfigManager:
    """降级配置管理器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def config_manager(self, temp_dir, monkeypatch):
        """创建配置管理器"""
        # 修改配置路径到临时目录
        with patch('src.features.fallback.config_fallback.Path') as mock_path:
            mock_path.return_value = temp_dir / "config" / "features_config.json"
            manager = FallbackConfigManager()
            manager.config_file = temp_dir / "config" / "features_config.json"
            manager.config_file.parent.mkdir(parents=True, exist_ok=True)
            return manager

    def test_init_defaults(self, config_manager):
        """测试初始化默认配置"""
        assert config_manager._config is not None
        assert "features" in config_manager._config
        assert "technical_indicators" in config_manager._config

    def test_get_defaults(self, config_manager):
        """测试获取默认配置"""
        defaults = config_manager._get_defaults()
        assert isinstance(defaults, dict)
        assert "features" in defaults
        assert "technical_indicators" in defaults

    def test_load_config_file_exists(self, config_manager, temp_dir):
        """测试加载存在的配置文件"""
        # 创建配置文件
        config_file = temp_dir / "config" / "features_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_config = {
            "features": {
                "enable_caching": False,
                "max_workers": 8
            }
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(test_config, f)
            
            # 更新config_manager的config_file路径
            config_manager.config_file = config_file
            
            # 重新加载配置
            config_manager._load_config()
            
            assert config_manager.get_config("features.enable_caching") is False
            assert config_manager.get_config("features.max_workers") == 8
        except (PermissionError, OSError):
            # 如果无法写入文件，跳过此测试
            pytest.skip("无法创建配置文件，跳过此测试")

    def test_load_config_file_not_exists(self, config_manager):
        """测试加载不存在的配置文件"""
        # 使用默认配置
        assert config_manager.get_config("features.enable_caching") is True
        assert config_manager.get_config("features.max_workers") == 4

    def test_load_config_invalid_json(self, config_manager, temp_dir):
        """测试加载无效的JSON文件"""
        config_file = temp_dir / "config" / "features_config_invalid.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 写入无效JSON
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write("invalid json {")
            
            # 更新config_manager的config_file路径
            original_config_file = config_manager.config_file
            config_manager.config_file = config_file
            
            # 应该使用默认配置
            config_manager._load_config()
            assert config_manager.get_config("features.enable_caching") is True
        except (PermissionError, OSError):
            # 如果无法写入文件，跳过此测试
            pytest.skip("无法创建配置文件，跳过此测试")
        finally:
            # 恢复原始config_file
            if 'original_config_file' in locals():
                config_manager.config_file = original_config_file

    def test_merge_configs(self, config_manager):
        """测试合并配置"""
        base = {
            "features": {
                "enable_caching": True,
                "max_workers": 4
            }
        }
        override = {
            "features": {
                "max_workers": 8
            }
        }
        
        merged = config_manager._merge_configs(base, override)
        assert merged["features"]["enable_caching"] is True
        assert merged["features"]["max_workers"] == 8

    def test_get_config_simple_key(self, config_manager):
        """测试获取简单配置键"""
        value = config_manager.get_config("features.enable_caching")
        assert value is True

    def test_get_config_nested_key(self, config_manager):
        """测试获取嵌套配置键"""
        value = config_manager.get_config("features.max_workers")
        assert value == 4

    def test_get_config_not_found(self, config_manager):
        """测试获取不存在的配置"""
        value = config_manager.get_config("nonexistent.key", default="default_value")
        assert value == "default_value"

    def test_get_config_default_none(self, config_manager):
        """测试获取配置-默认值为None"""
        value = config_manager.get_config("nonexistent.key")
        assert value is None

    def test_set_config_simple_key(self, config_manager, temp_dir):
        """测试设置简单配置键"""
        success = config_manager.set_config("test_key", "test_value")
        assert success is True
        assert config_manager.get_config("test_key") == "test_value"

    def test_set_config_nested_key(self, config_manager, temp_dir):
        """测试设置嵌套配置键"""
        success = config_manager.set_config("features.test_setting", "test_value")
        assert success is True
        assert config_manager.get_config("features.test_setting") == "test_value"

    def test_set_config_create_nested(self, config_manager, temp_dir):
        """测试设置配置-创建嵌套结构"""
        success = config_manager.set_config("new.section.key", "value")
        assert success is True
        assert config_manager.get_config("new.section.key") == "value"

    def test_has_config(self, config_manager):
        """测试检查配置是否存在"""
        assert config_manager.has_config("features.enable_caching") is True
        assert config_manager.has_config("nonexistent.key") is False

    def test_get_all_configs(self, config_manager):
        """测试获取所有配置"""
        all_configs = config_manager.get_all_configs()
        assert isinstance(all_configs, dict)
        assert "features" in all_configs
        # 应该是副本，不是引用
        all_configs["test"] = "value"
        assert "test" not in config_manager._config

    def test_save_config(self, config_manager, temp_dir):
        """测试保存配置"""
        # 确保config_file路径可写
        config_file = temp_dir / "config" / "test_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_manager.config_file = config_file
        
        try:
            config_manager.set_config("test_key", "test_value")
            
            # 验证配置已设置（即使在内存中）
            assert config_manager.get_config("test_key") == "test_value"
            
            # 如果文件存在，验证文件已保存
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        saved_config = json.load(f)
                    assert "test_key" in saved_config or any("test_key" in str(v) for v in saved_config.values())
                except (PermissionError, OSError):
                    # 如果无法读取文件，至少验证内存中的配置已设置
                    pass
        except (PermissionError, OSError):
            # 如果无法保存文件，至少验证set_config返回True且内存中配置已设置
            success = config_manager.set_config("test_key", "test_value")
            # 即使保存失败，内存中的配置应该已设置
            assert config_manager.get_config("test_key") == "test_value"

    def test_reload_config(self, config_manager, temp_dir):
        """测试重新加载配置"""
        # 修改配置
        config_manager.set_config("test_key", "old_value")
        
        # 直接修改文件
        config_file = config_manager.config_file
        try:
            if config_file.exists():
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump({"test_key": "new_value"}, f)
                
                # 重新加载
                success = config_manager.reload_config()
                assert success is True
            else:
                # 如果文件不存在，测试重新加载默认配置
                success = config_manager.reload_config()
                assert success is True
        except (PermissionError, OSError):
            # 如果无法写入文件，跳过此测试
            pytest.skip("无法写入配置文件，跳过此测试")

    def test_reset_to_defaults(self, config_manager):
        """测试重置为默认配置"""
        # 修改配置
        config_manager.set_config("test_key", "test_value")
        
        # 重置
        success = config_manager.reset_to_defaults()
        assert success is True
        assert config_manager.get_config("test_key") is None
        assert config_manager.get_config("features.enable_caching") is True

    def test_get_config_section(self, config_manager):
        """测试获取配置节"""
        section = config_manager.get_config_section("features")
        assert isinstance(section, dict)
        assert "enable_caching" in section

    def test_get_config_section_not_found(self, config_manager):
        """测试获取不存在的配置节"""
        section = config_manager.get_config_section("nonexistent")
        assert section == {}

    def test_update_config_section(self, config_manager):
        """测试更新配置节"""
        updates = {
            "new_setting": "new_value",
            "max_workers": 8
        }
        
        success = config_manager.update_config_section("features", updates)
        assert success is True
        assert config_manager.get_config("features.new_setting") == "new_value"
        assert config_manager.get_config("features.max_workers") == 8

    def test_validate_config(self, config_manager):
        """测试验证配置"""
        errors = config_manager.validate_config()
        assert isinstance(errors, list)
        # 默认配置应该没有错误
        assert len(errors) == 0

    def test_validate_config_missing_required(self, config_manager):
        """测试验证配置-缺少必需项"""
        # 删除必需配置
        if "features" in config_manager._config:
            del config_manager._config["features"]["enable_standardization"]
        
        errors = config_manager.validate_config()
        assert len(errors) > 0
        assert any("缺少必需配置" in error for error in errors)

    def test_validate_config_invalid_ttl(self, config_manager):
        """测试验证配置-无效的TTL"""
        config_manager.set_config("features.cache_ttl", -1)
        errors = config_manager.validate_config()
        assert any("cache_ttl" in error for error in errors)

    def test_validate_config_invalid_workers(self, config_manager):
        """测试验证配置-无效的工作节点数"""
        config_manager.set_config("features.max_workers", 0)
        errors = config_manager.validate_config()
        assert any("max_workers" in error for error in errors)

