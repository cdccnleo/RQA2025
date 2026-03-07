#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Config Factory Utils 测试

测试 src/infrastructure/config/core/config_factory_utils.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch

# 尝试导入模块
try:
    from src.infrastructure.config.core.config_factory_utils import (
        get_config_factory,
        reset_global_factory,
        create_config_manager,
        get_available_config_types,
        get_factory_stats,
        ConfigManagerFactory,
        get_config_manager,
        register_config_manager
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigFactoryUtils:
    """测试config_factory_utils.py的功能"""

    def setup_method(self):
        """测试前准备"""
        # 重置全局工厂状态
        reset_global_factory()

    def teardown_method(self):
        """测试后清理"""
        # 重置全局工厂状态
        reset_global_factory()

    def test_get_config_factory_initial(self):
        """测试首次获取配置工厂"""
        factory = get_config_factory()
        assert factory is not None

    def test_get_config_factory_singleton(self):
        """测试配置工厂单例模式"""
        factory1 = get_config_factory()
        factory2 = get_config_factory()
        
        # 应该是同一个实例
        assert factory1 is factory2

    def test_reset_global_factory(self):
        """测试重置全局工厂"""
        # 获取初始工厂
        factory1 = get_config_factory()
        assert factory1 is not None
        
        # 重置
        reset_global_factory()
        
        # 获取新工厂
        factory2 = get_config_factory()
        assert factory2 is not None
        
        # 应该是不同的实例
        assert factory1 is not factory2

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_create_config_manager_basic(self, mock_get_factory):
        """测试创建配置管理器"""
        # Mock工厂和管理器
        mock_manager = StandardMockBuilder.create_config_mock()
        mock_factory = Mock()
        mock_factory.create_manager.return_value = mock_manager
        mock_get_factory.return_value = mock_factory
        
        # 测试创建管理器
        result = create_config_manager("unified")
        
        mock_factory.create_manager.assert_called_once_with("unified")
        assert result == mock_manager

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_create_config_manager_with_kwargs(self, mock_get_factory):
        """测试使用参数创建配置管理器"""
        mock_manager = StandardMockBuilder.create_config_mock()
        mock_factory = Mock()
        mock_factory.create_manager.return_value = mock_manager
        mock_get_factory.return_value = mock_factory
        
        kwargs = {"config": {"test": "value"}}
        result = create_config_manager("test_type", **kwargs)
        
        mock_factory.create_manager.assert_called_once_with("test_type", config={"test": "value"})
        assert result == mock_manager

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_get_available_config_types(self, mock_get_factory):
        """测试获取可用配置类型"""
        mock_factory = Mock()
        mock_stats = {"available_types": ["unified", "test", "custom"]}
        mock_factory.get_stats.return_value = mock_stats
        mock_get_factory.return_value = mock_factory
        
        result = get_available_config_types()
        
        mock_factory.get_stats.assert_called_once()
        assert result == ["unified", "test", "custom"]

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_get_factory_stats(self, mock_get_factory):
        """测试获取工厂统计信息"""
        mock_factory = Mock()
        mock_stats = {
            "created_count": 10,
            "cached_hits": 5,
            "errors": 0
        }
        mock_factory.get_stats.return_value = mock_stats
        mock_get_factory.return_value = mock_factory
        
        result = get_factory_stats()
        
        mock_factory.get_stats.assert_called_once()
        assert result == mock_stats

    def test_config_manager_factory_alias(self):
        """测试ConfigManagerFactory别名"""
        from src.infrastructure.config.core.config_factory_core import UnifiedConfigFactory
        
        # 验证别名指向正确的类
        assert ConfigManagerFactory is UnifiedConfigFactory

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_get_config_manager_alias(self, mock_get_factory):
        """测试get_config_manager别名函数"""
        mock_manager = StandardMockBuilder.create_config_mock()
        mock_factory = Mock()
        mock_factory.create_manager.return_value = mock_manager
        mock_get_factory.return_value = mock_factory
        
        result = get_config_manager("test_type", config="test_config")
        
        mock_factory.create_manager.assert_called_once_with("test_type", config="test_config")
        assert result == mock_manager

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_register_config_manager(self, mock_get_factory):
        """测试注册配置管理器"""
        mock_factory = Mock()
        mock_manager_class = Mock()
        mock_get_factory.return_value = mock_factory
        
        register_config_manager("test_manager", mock_manager_class)
        
        mock_factory.register_manager.assert_called_once_with("test_manager", mock_manager_class)

    def test_module_imports(self):
        """测试模块导入完整性"""
        # 验证所有函数都可以导入
        assert get_config_factory is not None
        assert reset_global_factory is not None
        assert create_config_manager is not None
        assert get_available_config_types is not None
        assert get_factory_stats is not None
        assert ConfigManagerFactory is not None
        assert get_config_manager is not None
        assert register_config_manager is not None

    def test_functions_are_callable(self):
        """测试所有函数都是可调用的"""
        callable_functions = [
            get_config_factory,
            reset_global_factory,
            create_config_manager,
            get_available_config_types,
            get_factory_stats,
            get_config_manager,
            register_config_manager
        ]
        
        for func in callable_functions:
            assert callable(func), f"{func.__name__} is not callable"


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigFactoryUtilsIntegration:
    """测试配置工厂工具集成功能"""

    def setup_method(self):
        """测试前准备"""
        reset_global_factory()

    def teardown_method(self):
        """测试后清理"""
        reset_global_factory()

    def test_workflow_integration(self):
        """测试工作流集成"""
        # 获取工厂
        factory = get_config_factory()
        assert factory is not None
        
        # 获取统计信息
        stats = get_factory_stats()
        assert isinstance(stats, dict)
        
        # 重置工厂
        reset_global_factory()
        
        # 验证重置后获取新工厂
        new_factory = get_config_factory()
        assert new_factory is not None
        assert new_factory is not factory

    def test_convenience_functions_consistency(self):
        """测试便捷函数一致性"""
        # create_config_manager和get_config_manager应该是相同的
        # 但实际实现可能不同，我们只验证它们都是函数
        assert callable(create_config_manager)
        assert callable(get_config_manager)
        
        # 验证默认参数
        # 这些函数可能会抛出异常或返回None，但我们主要测试它们可调用
        try:
            # 尝试调用（可能会失败，但函数应该存在）
            create_config_manager()
        except Exception:
            pass  # 预期的，因为我们mock了依赖
        
        try:
            get_config_manager()
        except Exception:
            pass  # 预期的

    def test_global_state_management(self):
        """测试全局状态管理"""
        # 测试初始状态
        factory1 = get_config_factory()
        
        # 重置
        reset_global_factory()
        
        # 再次获取应该创建新实例
        factory2 = get_config_factory()
        assert factory1 is not factory2
        
        # 再次获取应该返回相同实例
        factory3 = get_config_factory()
        assert factory2 is factory3
