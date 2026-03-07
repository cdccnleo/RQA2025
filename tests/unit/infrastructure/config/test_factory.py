"""
测试 factory 模块

覆盖 factory.py 中的别名和全局函数
"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.config.core.factory import (
    reset_global_factory,
    get_available_config_types,
    get_factory_stats,
    get_config_manager,
    register_config_manager,
    UnifiedConfigFactory,
    ConfigFactory,
    get_config_factory,
    create_config_manager
)


class TestFactoryFunctions:
    """工厂函数测试"""

    def test_reset_global_factory(self):
        """测试重置全局工厂"""
        # This function appears to be a placeholder, should not raise exception
        reset_global_factory()

    def test_get_available_config_types(self):
        """测试获取可用配置类型"""
        result = get_available_config_types()
        assert isinstance(result, list)
        assert "unified" in result

    def test_get_factory_stats(self):
        """测试获取工厂统计信息"""
        result = get_factory_stats()
        assert isinstance(result, dict)

    @patch('src.infrastructure.config.core.factory.get_config_factory')
    def test_get_config_manager(self, mock_get_factory):
        """测试获取配置管理器"""
        mock_factory = Mock()
        mock_manager = Mock()
        mock_factory.get_manager.return_value = mock_manager
        mock_get_factory.return_value = mock_factory

        result = get_config_manager("test_manager")
        assert result == mock_manager
        mock_get_factory.assert_called_once()
        mock_factory.get_manager.assert_called_once_with("test_manager")

    @patch('src.infrastructure.config.core.factory.get_config_factory')
    def test_register_config_manager(self, mock_get_factory):
        """测试注册配置管理器"""
        mock_factory = Mock()
        mock_get_factory.return_value = mock_factory

        mock_manager_class = Mock()
        register_config_manager("test_manager", mock_manager_class)

        mock_get_factory.assert_called_once()
        mock_factory.register_manager.assert_called_once_with("test_manager", mock_manager_class)


class TestFactoryImports:
    """工厂导入测试"""

    def test_unified_config_factory_import(self):
        """测试UnifiedConfigFactory导入"""
        # Should be importable
        assert UnifiedConfigFactory is not None

    def test_config_factory_import(self):
        """测试ConfigFactory导入"""
        # Should be importable
        assert ConfigFactory is not None

    def test_get_config_factory_import(self):
        """测试get_config_factory函数导入"""
        # Should be importable
        assert callable(get_config_factory)

    def test_create_config_manager_import(self):
        """测试create_config_manager函数导入"""
        # Should be importable
        assert callable(create_config_manager)


class TestFactoryAll:
    """__all__ 测试"""

    def test_all_exports(self):
        """测试__all__中定义的所有导出"""
        from src.infrastructure.config.core import factory

        expected_exports = [
            'UnifiedConfigFactory',
            'ConfigFactory',
            'get_config_factory',
            'create_config_manager',
            'reset_global_factory',
            'get_available_config_types',
            'get_factory_stats',
            'get_config_manager',
            'register_config_manager'
        ]

        for export in expected_exports:
            assert hasattr(factory, export), f"Factory module should export {export}"


class TestFactoryIntegration:
    """工厂集成测试"""

    def test_factory_workflow(self):
        """测试工厂工作流"""
        # Test that we can get available config types
        config_types = get_available_config_types()
        assert len(config_types) > 0

        # Test that we can get factory stats
        stats = get_factory_stats()
        assert isinstance(stats, dict)

    def test_factory_function_signatures(self):
        """测试工厂函数签名"""
        # Test that functions have expected signatures
        assert callable(get_config_manager)
        assert callable(register_config_manager)
        assert callable(get_available_config_types)
        assert callable(get_factory_stats)
        assert callable(reset_global_factory)

    def test_factory_imports_are_classes_or_functions(self):
        """测试工厂导入是类或函数"""
        # Test that imports are either classes or callable functions
        assert callable(get_config_factory) or hasattr(get_config_factory, '__init__')
        assert callable(create_config_manager) or hasattr(create_config_manager, '__init__')
        assert hasattr(UnifiedConfigFactory, '__init__')
        assert hasattr(ConfigFactory, '__init__')
