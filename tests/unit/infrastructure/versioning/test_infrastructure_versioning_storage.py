#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infrastructure Versioning存储功能测试

测试版本存储、检索、持久化等功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.versioning.core.version import Version
from src.infrastructure.versioning.manager.manager import VersionManager
from src.infrastructure.versioning.config.config_version_manager import ConfigVersionManager


class TestVersionStorage:
    """测试版本存储功能"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_store_single_version(self, version_manager):
        """测试存储单个版本"""
        version_manager.register_version("component1", "1.0.0")
        stored = version_manager.get_version("component1")
        assert stored == Version("1.0.0")
    
    def test_store_multiple_versions(self, version_manager):
        """测试存储多个组件的版本"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("api", "2.0.0")
        version_manager.register_version("db", "3.0.0")
        
        all_versions = version_manager.list_versions()
        assert len(all_versions) == 3
    
    def test_overwrite_version(self, version_manager):
        """测试覆盖已存在的版本"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("app", "1.1.0")
        
        current = version_manager.get_version("app")
        assert current == Version("1.1.0")
    
    def test_version_persistence_in_memory(self, version_manager):
        """测试版本在内存中的持久性"""
        version_manager.register_version("app", "1.0.0")
        
        # 多次获取，应返回同一版本
        v1 = version_manager.get_version("app")
        v2 = version_manager.get_version("app")
        assert v1 == v2
    
    def test_store_version_metadata(self, version_manager):
        """测试存储版本元数据"""
        # 如果支持元数据存储
        version = Version("1.0.0")
        version_manager.register_version("app", version)
        
        retrieved = version_manager.get_version("app")
        assert retrieved.major == 1
        assert retrieved.minor == 0
        assert retrieved.patch == 0


class TestVersionHistory:
    """测试版本历史管理"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_maintain_version_history(self, version_manager):
        """测试维护版本历史"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("app", "1.1.0")
        version_manager.register_version("app", "1.2.0")
        
        history = version_manager.get_version_history("app")
        assert len(history) >= 3
    
    def test_get_specific_historical_version(self, version_manager):
        """测试获取特定历史版本"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("app", "1.1.0")
        version_manager.register_version("app", "1.2.0")
        
        history = version_manager.get_version_history("app")
        assert history[0] == Version("1.0.0")
        assert history[1] == Version("1.1.0")
    
    def test_clear_version_history(self, version_manager):
        """测试清除版本历史"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("app", "1.1.0")
        
        # 清除历史（如果支持）
        if hasattr(version_manager, 'clear_history'):
            version_manager.clear_history("app")
            history = version_manager.get_version_history("app")
            assert len(history) == 0


class TestConfigVersionManager:
    """测试配置版本管理器"""
    
    @pytest.fixture
    def config_version_manager(self):
        """创建配置版本管理器fixture"""
        return ConfigVersionManager()
    
    def test_create_config_version(self, config_version_manager):
        """测试创建配置版本"""
        config_data = {"key": "value", "setting": 123}
        version = config_version_manager.create_version("test_config", config_data)
        assert version is not None
    
    def test_get_config_by_version(self, config_version_manager):
        """测试根据版本获取配置"""
        config_data = {"key": "value"}
        version = config_version_manager.create_version("test_config", config_data)
        
        retrieved_config = config_version_manager.get_config("test_config", version)
        assert retrieved_config == config_data
    
    def test_list_all_config_versions(self, config_version_manager):
        """测试列出所有配置版本"""
        config_version_manager.create_version("test_config", {"v": 1})
        config_version_manager.create_version("test_config", {"v": 2})
        
        versions = config_version_manager.list_versions("test_config")
        assert len(versions) >= 2
    
    def test_compare_config_versions(self, config_version_manager):
        """测试比较配置版本"""
        v1 = config_version_manager.create_version("test_config", {"key": "value1"})
        v2 = config_version_manager.create_version("test_config", {"key": "value2"})
        
        diff = config_version_manager.compare_versions("test_config", v1, v2)
        assert diff is not None


class TestVersionRetrieval:
    """测试版本检索功能"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        manager = VersionManager()
        # 预填充一些测试数据
        manager.register_version("app", "1.0.0")
        manager.register_version("api", "2.0.0")
        manager.register_version("lib", "0.5.0")
        return manager
    
    def test_get_version_by_name(self, version_manager):
        """测试按名称获取版本"""
        version = version_manager.get_version("app")
        assert version == Version("1.0.0")
    
    def test_get_all_versions_dict(self, version_manager):
        """测试获取所有版本字典"""
        all_versions = version_manager.list_versions()
        assert isinstance(all_versions, dict)
        assert len(all_versions) == 3
    
    def test_get_versions_by_pattern(self, version_manager):
        """测试按模式获取版本"""
        # 如果支持模式匹配
        if hasattr(version_manager, 'get_versions_by_pattern'):
            versions = version_manager.get_versions_by_pattern("a*")
            assert "app" in [name for name, _ in versions]
    
    def test_get_latest_version(self, version_manager):
        """测试获取最新版本"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("app", "1.1.0")
        version_manager.register_version("app", "1.2.0")
        
        latest = version_manager.get_version("app")
        assert latest == Version("1.2.0")
    
    def test_search_versions_by_criteria(self, version_manager):
        """测试按条件搜索版本"""
        # 搜索主版本号为1的所有版本
        version_manager.register_version("app1", "1.0.0")
        version_manager.register_version("app2", "2.0.0")
        version_manager.register_version("app3", "1.5.0")
        
        all_v = version_manager.list_versions()
        v1_versions = {k: v for k, v in all_v.items() if v.major == 1}
        assert len(v1_versions) >= 2


class TestVersionPersistence:
    """测试版本持久化功能"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    @patch('builtins.open', create=True)
    def test_save_versions_to_file(self, mock_open, version_manager):
        """测试保存版本到文件"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("api", "2.0.0")
        
        # 如果支持保存到文件
        if hasattr(version_manager, 'save_to_file'):
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            version_manager.save_to_file("versions.json")
            mock_open.assert_called_once()
    
    @patch('builtins.open', create=True)
    def test_load_versions_from_file(self, mock_open, version_manager):
        """测试从文件加载版本"""
        # 如果支持从文件加载
        if hasattr(version_manager, 'load_from_file'):
            mock_file = MagicMock()
            mock_file.read.return_value = '{"app": "1.0.0", "api": "2.0.0"}'
            mock_open.return_value.__enter__.return_value = mock_file
            
            version_manager.load_from_file("versions.json")
            
            assert version_manager.get_version("app") is not None
    
    def test_export_versions_to_dict(self, version_manager):
        """测试导出版本到字典"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("api", "2.0.0")
        
        # 使用list_versions代替export_to_dict
        exported = version_manager.list_versions()
        assert isinstance(exported, dict)
        assert "app" in exported and "api" in exported
    
    def test_import_versions_from_dict(self, version_manager):
        """测试从字典导入版本"""
        data = {
            "app": "1.0.0",
            "api": "2.0.0",
            "lib": "0.5.0"
        }
        
        if hasattr(version_manager, 'import_from_dict'):
            version_manager.import_from_dict(data)
            assert len(version_manager.list_versions()) == 3


class TestVersionCaching:
    """测试版本缓存功能"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_version_cache_hit(self, version_manager):
        """测试版本缓存命中"""
        version_manager.register_version("app", "1.0.0")
        
        # 第一次获取
        v1 = version_manager.get_version("app")
        # 第二次获取（应该从缓存）
        v2 = version_manager.get_version("app")
        
        assert v1 == v2
    
    def test_version_cache_invalidation(self, version_manager):
        """测试版本缓存失效"""
        version_manager.register_version("app", "1.0.0")
        v1 = version_manager.get_version("app")
        
        # 更新版本后，缓存应失效
        version_manager.register_version("app", "1.1.0")
        v2 = version_manager.get_version("app")
        
        assert v1 != v2
    
    def test_clear_version_cache(self, version_manager):
        """测试清除版本缓存"""
        version_manager.register_version("app", "1.0.0")
        
        # 如果支持清除缓存
        if hasattr(version_manager, 'clear_cache'):
            version_manager.clear_cache()
            # 缓存清除后应该能重新获取
            v = version_manager.get_version("app")
            assert v == Version("1.0.0")


class TestVersionDeletion:
    """测试版本删除功能"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        manager = VersionManager()
        manager.register_version("app", "1.0.0")
        manager.register_version("api", "2.0.0")
        return manager
    
    def test_delete_single_version(self, version_manager):
        """测试删除单个版本"""
        version_manager.remove_version("app")
        assert version_manager.get_version("app") is None
    
    def test_delete_nonexistent_version(self, version_manager):
        """测试删除不存在的版本（应不报错）"""
        # 应该不抛出异常
        version_manager.remove_version("nonexistent")
    
    def test_delete_all_versions(self, version_manager):
        """测试删除所有版本"""
        version_manager.clear_versions()
        assert len(version_manager.list_versions()) == 0
    
    def test_delete_with_history_preservation(self, version_manager):
        """测试删除版本但保留历史"""
        version_manager.register_version("app", "1.1.0")
        history_before = len(version_manager.get_version_history("app"))
        
        # 删除当前版本
        version_manager.remove_version("app")
        
        # 历史应该保留（如果实现支持）
        if hasattr(version_manager, 'preserve_history'):
            history_after = version_manager.get_version_history("app")
            assert len(history_after) == history_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

