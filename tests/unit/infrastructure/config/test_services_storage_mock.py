#!/usr/bin/env python3
"""
测试配置存储服务 - 使用模拟对象避免复杂依赖

测试覆盖：
- ConfigStorageService类的核心功能
- 存储和缓存操作
- 统计信息收集
- 存储后端管理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any

from src.infrastructure.config.services.config_storage_service import ConfigStorageService


class TestConfigStorageService:
    """测试配置存储服务 - 使用模拟对象"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟存储后端
        self.mock_backend = MagicMock()
        self.mock_backend.load.return_value = {"test": "value"}
        self.mock_backend.save.return_value = True
        self.mock_backend.reload.return_value = True

        # 创建服务实例
        self.service = ConfigStorageService(
            storage_backend=self.mock_backend,
            cache_enabled=True,
            cache_size=100
        )

    def test_initialization(self):
        """测试初始化"""
        assert self.service is not None
        assert hasattr(self.service, '_storage_backend')
        assert hasattr(self.service, '_cache')
        assert hasattr(self.service, '_cache_enabled')
        assert hasattr(self.service, '_cache_size')
        assert hasattr(self.service, '_cache_timestamps')
        assert hasattr(self.service, '_cache_access_times')
        assert hasattr(self.service, '_stats')

    def test_initialization_without_backend(self):
        """测试不带后端初始化"""
        service = ConfigStorageService()

        assert service._storage_backend is None
        assert service._cache_enabled is True
        assert service._cache_size == 1000  # 默认值

    def test_initialization_cache_disabled(self):
        """测试禁用缓存的初始化"""
        service = ConfigStorageService(cache_enabled=False)

        assert service._cache_enabled is False
        assert service._cache == {}
        assert service._cache_timestamps == {}
        assert service._cache_access_times == {}

    def test_load_from_backend(self):
        """测试从后端加载配置"""
        # 配置mock
        self.mock_backend.load.return_value = {"key": "backend_value"}

        result = self.service.load("test_source")

        assert result == {"key": "backend_value"}
        self.mock_backend.load.assert_called_once_with("test_source")

    def test_save_to_backend(self):
        """测试保存配置到后端"""
        # 配置mock
        self.mock_backend.save.return_value = True

        config_data = {"key": "value"}
        result = self.service.save(config_data, "test_target")

        assert result is True
        self.mock_backend.save.assert_called_once_with(config_data, "test_target")

    def test_reload_from_backend(self):
        """测试从后端重新加载配置"""
        # 配置mock
        self.mock_backend.reload.return_value = True

        result = self.service.reload()

        assert result is True
        self.mock_backend.reload.assert_called_once()

    def test_get_cache_stats(self):
        """测试获取缓存统计"""
        stats = self.service.get_cache_stats()

        # 根据实际的 get_cache_stats 方法返回格式进行断言
        assert isinstance(stats, dict)
        assert 'enabled' in stats
        if stats['enabled']:
            assert 'size' in stats
            assert 'max_size' in stats
            assert 'hit_rate' in stats
            assert 'entries' in stats

    def test_get_storage_stats(self):
        """测试获取存储统计"""
        stats = self.service.get_storage_stats()

        # 存储统计应该包含基本的统计信息
        assert isinstance(stats, dict)

    def test_set_storage_backend(self):
        """测试设置存储后端"""
        new_backend = MagicMock()
        self.service.set_storage_backend(new_backend)

        assert self.service._storage_backend == new_backend

    def test_cleanup(self):
        """测试清理操作"""
        # 设置一些数据
        self.service._cache["test_key"] = {"data": "value"}

        self.service.cleanup()

        # 缓存应该被清理
        assert self.service._cache == {}
        assert self.service._cache_timestamps == {}
        assert self.service._cache_access_times == {}
