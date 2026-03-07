"""
测试 ConfigFactoryUtils 工具函数

覆盖 config_factory_utils.py 中的全局工厂实例和便捷函数
"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.config.core.config_factory_utils import (
    get_config_factory,
    reset_global_factory,
    create_config_manager,
    get_available_config_types,
    get_factory_stats,
    get_config_manager,
    register_config_manager
)


class TestGlobalFactoryInstance:
    """全局工厂实例测试"""

    def test_get_config_factory_returns_unified_factory(self):
        """测试get_config_factory返回UnifiedConfigFactory实例"""
        # Reset first to ensure clean state
        reset_global_factory()

        factory = get_config_factory()
        assert factory is not None
        # Should be an instance of UnifiedConfigFactory
        from src.infrastructure.config.core.config_factory_core import UnifiedConfigFactory
        assert isinstance(factory, UnifiedConfigFactory)

    def test_get_config_factory_singleton(self):
        """测试get_config_factory单例行为"""
        # Reset first
        reset_global_factory()

        factory1 = get_config_factory()
        factory2 = get_config_factory()

        # Should return the same instance
        assert factory1 is factory2

    def test_reset_global_factory(self):
        """测试重置全局工厂"""
        # Get a factory first
        factory1 = get_config_factory()

        # Reset
        reset_global_factory()

        # Get factory again - should be a new instance
        factory2 = get_config_factory()

        # Should be different instances
        assert factory1 is not factory2


class TestConvenienceFunctions:
    """便捷函数测试"""

    def setup_method(self):
        """每个测试前重置全局工厂"""
        reset_global_factory()

    def teardown_method(self):
        """每个测试后重置全局工厂"""
        reset_global_factory()

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_create_config_manager(self, mock_get_factory):
        """测试创建配置管理器便捷函数"""
        mock_factory = Mock()
        mock_manager = Mock()
        mock_factory.create_manager.return_value = mock_manager
        mock_get_factory.return_value = mock_factory

        result = create_config_manager("test_type", param1="value1")

        assert result == mock_manager
        mock_get_factory.assert_called_once()
        mock_factory.create_manager.assert_called_once_with("test_type", param1="value1")

    def test_get_available_config_types(self):
        """测试获取可用配置类型"""
        # This function should return a list of available config types
        result = get_available_config_types()
        assert isinstance(result, list)
        # Should contain at least "unified"
        assert len(result) > 0

    def test_get_factory_stats(self):
        """测试获取工厂统计信息"""
        result = get_factory_stats()
        assert isinstance(result, dict)

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_get_config_manager(self, mock_get_factory):
        """测试获取配置管理器便捷函数"""
        mock_factory = Mock()
        mock_manager = Mock()
        mock_factory.create_manager.return_value = mock_manager
        mock_get_factory.return_value = mock_factory

        result = get_config_manager("test_type", param1="value1")

        assert result == mock_manager
        mock_get_factory.assert_called_once()
        mock_factory.create_manager.assert_called_once_with("test_type", param1="value1")

    @patch('src.infrastructure.config.core.config_factory_utils.get_config_factory')
    def test_register_config_manager(self, mock_get_factory):
        """测试注册配置管理器便捷函数"""
        mock_factory = Mock()
        mock_get_factory.return_value = mock_factory

        mock_manager_class = Mock()
        register_config_manager("test_manager", mock_manager_class)

        mock_get_factory.assert_called_once()
        mock_factory.register_manager.assert_called_once_with("test_manager", mock_manager_class)


class TestFactoryUtilsIntegration:
    """工厂工具集成测试"""

    def setup_method(self):
        """每个测试前重置全局工厂"""
        reset_global_factory()

    def teardown_method(self):
        """每个测试后重置全局工厂"""
        reset_global_factory()

    def test_complete_workflow(self):
        """测试完整工作流"""
        # Get available types
        types = get_available_config_types()
        assert isinstance(types, list)

        # Get factory stats
        stats = get_factory_stats()
        assert isinstance(stats, dict)

        # Create a config manager
        manager = create_config_manager("unified", config={"test": "value"})
        assert manager is not None

    def test_factory_isolation_between_tests(self):
        """测试测试间的工厂隔离"""
        # This test ensures that the setup/teardown works correctly
        factory1 = get_config_factory()
        reset_global_factory()
        factory2 = get_config_factory()

        assert factory1 is not factory2

    def test_functions_are_callable(self):
        """测试所有函数都是可调用的"""
        assert callable(get_config_factory)
        assert callable(reset_global_factory)
        assert callable(create_config_manager)
        assert callable(get_available_config_types)
        assert callable(get_factory_stats)
        assert callable(get_config_manager)
        assert callable(register_config_manager)

    def test_default_parameters(self):
        """测试默认参数"""
        # Test create_config_manager default
        manager = create_config_manager()
        assert manager is not None

        # Test get_config_manager default
        manager2 = get_config_manager()
        assert manager2 is not None

    def test_factory_stats_structure(self):
        """测试工厂统计信息结构"""
        stats = get_factory_stats()
        # Stats should be a dictionary, even if empty
        assert isinstance(stats, dict)
        # Should not contain unexpected types
        for key, value in stats.items():
            assert isinstance(key, str)
            # Values can be various types
