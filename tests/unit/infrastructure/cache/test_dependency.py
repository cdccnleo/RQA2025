#!/usr/bin/env python3
"""
基础设施层 - 依赖检查工具测试

测试dependency.py中的配置依赖检查功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import patch
from src.infrastructure.cache.utils.dependency import DependencyChecker


class TestDependencyChecker:
    """测试依赖检查器"""

    def test_check_dependencies_empty_config(self):
        """测试检查空配置"""
        new_config = {}
        full_config = {}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert errors == {}

    def test_check_dependencies_no_changes(self):
        """测试无配置变更"""
        new_config = {}
        full_config = {'cache.enabled': True, 'cache.size': 1000}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert errors == {}

    def test_check_cache_dependencies_enabled_valid(self):
        """测试启用缓存的有效依赖"""
        new_config = {'cache.enabled': True}
        full_config = {'cache.size': 1000}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert errors == {}

    def test_check_cache_dependencies_enabled_missing_size(self):
        """测试启用缓存但缺少大小配置"""
        new_config = {'cache.enabled': True}
        full_config = {}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert 'cache.size' in errors
        assert 'Cache size must be set when cache is enabled' in errors['cache.size']

    def test_check_cache_dependencies_enabled_invalid_size_type(self):
        """测试启用缓存但大小配置类型无效"""
        new_config = {'cache.enabled': True}
        full_config = {'cache.size': 'invalid'}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert 'cache.size' in errors
        assert 'Cache size must be a number' in errors['cache.size']

    def test_check_cache_dependencies_enabled_zero_size(self):
        """测试启用缓存但大小为零"""
        new_config = {'cache.enabled': True}
        full_config = {'cache.size': 0}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert 'cache.size' in errors
        assert 'Cache size must be positive' in errors['cache.size']

    def test_check_cache_dependencies_enabled_negative_size(self):
        """测试启用缓存但大小为负数"""
        new_config = {'cache.enabled': True}
        full_config = {'cache.size': -100}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert 'cache.size' in errors
        assert 'Cache size must be positive' in errors['cache.size']

    def test_check_cache_dependencies_string_values(self):
        """测试缓存启用配置的字符串值处理"""
        test_cases = [
            ('true', True, {}),
            ('false', False, {}),
            ('1', True, {}),
            ('0', False, {}),
            ('yes', True, {}),
            ('no', False, {}),
            ('on', True, {}),
            ('off', False, {}),
            ('invalid', False, {'cache.enabled': 'Invalid string value for cache.enabled'}),
        ]

        for string_val, should_check_size, expected_errors in test_cases:
            new_config = {'cache.enabled': string_val}
            full_config = {'cache.size': 1000} if should_check_size else {}

            errors = DependencyChecker.check_dependencies(new_config, full_config)

            if expected_errors:
                assert errors == expected_errors
            else:
                # 如果没有预期错误，检查是否通过了验证
                if should_check_size:
                    assert errors == {}
                else:
                    # 如果不应该检查大小，但配置中没有大小，也应该没有错误
                    pass

    def test_check_cache_dependencies_numeric_values(self):
        """测试缓存启用配置的数值处理"""
        test_cases = [
            (1, True, {}),
            (0, False, {}),
            (1.0, True, {}),
            (0.0, False, {}),
            (2, False, {'cache.enabled': 'Numeric value must be 0 or 1'}),
            (-1, False, {'cache.enabled': 'Numeric value must be 0 or 1'}),
        ]

        for num_val, should_check_size, expected_errors in test_cases:
            new_config = {'cache.enabled': num_val}
            full_config = {'cache.size': 1000} if should_check_size else {}

            errors = DependencyChecker.check_dependencies(new_config, full_config)
            assert errors == expected_errors

    def test_check_cache_dependencies_boolean_values(self):
        """测试缓存启用配置的布尔值处理"""
        test_cases = [
            (True, True, {}),
            (False, False, {}),
        ]

        for bool_val, should_check_size, expected_errors in test_cases:
            new_config = {'cache.enabled': bool_val}
            full_config = {'cache.size': 1000} if should_check_size else {}

            errors = DependencyChecker.check_dependencies(new_config, full_config)
            assert errors == expected_errors

    def test_check_cache_dependencies_invalid_type(self):
        """测试缓存启用配置的无效类型"""
        new_config = {'cache.enabled': []}  # 列表是无效类型
        full_config = {'cache.size': 1000}

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert 'cache.enabled' in errors
        assert 'Invalid type for cache.enabled' in errors['cache.enabled']

    def test_check_database_dependencies_enabled_valid(self):
        """测试启用数据库的有效依赖"""
        new_config = {'database.enabled': True}
        full_config = {
            'database.host': 'localhost',
            'database.port': 5432,
            'database.name': 'testdb'
        }

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert errors == {}

    def test_check_database_dependencies_enabled_missing_configs(self):
        """测试启用数据库但缺少必需配置"""
        new_config = {'database.enabled': True}
        full_config = {}  # 缺少所有必需配置

        errors = DependencyChecker.check_dependencies(new_config, full_config)

        expected_missing = ['database.host', 'database.port', 'database.name']
        for config_key in expected_missing:
            assert config_key in errors

    def test_check_database_dependencies_enabled_partial_missing(self):
        """测试启用数据库但部分缺少配置"""
        new_config = {'database.enabled': True}
        full_config = {
            'database.host': 'localhost',
            # 缺少port和name
        }

        errors = DependencyChecker.check_dependencies(new_config, full_config)

        assert 'database.port' in errors
        assert 'database.name' in errors
        assert 'database.host' not in errors

    def test_check_trading_dependencies_enabled_valid(self):
        """测试启用交易的有效依赖"""
        new_config = {'trading.enabled': True}
        full_config = {
            'trading.max_order_size': 1000,
            'trading.risk_limit': 0.1
        }

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert errors == {}

    def test_check_trading_dependencies_enabled_missing_configs(self):
        """测试启用交易但缺少必需配置"""
        new_config = {'trading.enabled': True}
        full_config = {}  # 缺少所有必需配置

        errors = DependencyChecker.check_dependencies(new_config, full_config)

        expected_missing = ['trading.max_order_size', 'trading.risk_limit']
        for config_key in expected_missing:
            assert config_key in errors

    def test_check_multiple_dependencies(self):
        """测试同时检查多种依赖"""
        new_config = {
            'cache.enabled': True,
            'database.enabled': True,
            'trading.enabled': True
        }
        full_config = {
            'cache.size': 1000,
            'database.host': 'localhost',
            'database.port': 5432,
            'database.name': 'testdb',
            'trading.max_order_size': 1000,
            'trading.risk_limit': 0.1
        }

        errors = DependencyChecker.check_dependencies(new_config, full_config)
        assert errors == {}  # 所有依赖都满足

    def test_check_multiple_dependencies_with_errors(self):
        """测试同时检查多种依赖（包含错误）"""
        new_config = {
            'cache.enabled': True,
            'database.enabled': True,
            'trading.enabled': True
        }
        full_config = {
            # 缺少cache.size
            # 缺少database.port
            'database.host': 'localhost',
            'database.name': 'testdb',
            'trading.max_order_size': 1000
            # 缺少trading.risk_limit
        }

        errors = DependencyChecker.check_dependencies(new_config, full_config)

        # 应该有3个错误
        assert len(errors) == 3
        assert 'cache.size' in errors
        assert 'database.port' in errors
        assert 'trading.risk_limit' in errors

    @patch('src.infrastructure.cache.utils.dependency.logger')
    def test_check_dependencies_logging(self, mock_logger):
        """测试依赖检查的日志记录"""
        new_config = {'cache.enabled': True}
        full_config = {}  # 缺少cache.size

        errors = DependencyChecker.check_dependencies(new_config, full_config)

        # 验证错误日志被记录
        mock_logger.error.assert_called_with("Cache size not set when enabling cache")

        # 验证调试日志被记录
        mock_logger.debug.assert_called()


class TestDependencyCheckerPrivateMethods:
    """测试依赖检查器的私有方法"""

    def test_check_cache_dependencies_direct_call(self):
        """直接测试缓存依赖检查方法"""
        new_config = {'cache.enabled': True}
        temp_config = {'cache.size': 1000}

        errors = DependencyChecker._check_cache_dependencies(new_config, temp_config)
        assert errors == {}

    def test_check_database_dependencies_direct_call(self):
        """直接测试数据库依赖检查方法"""
        new_config = {'database.enabled': True}
        temp_config = {
            'database.host': 'localhost',
            'database.port': 5432,
            'database.name': 'testdb'
        }

        errors = DependencyChecker._check_database_dependencies(new_config, temp_config)
        assert errors == {}

    def test_check_trading_dependencies_direct_call(self):
        """直接测试交易依赖检查方法"""
        new_config = {'trading.enabled': True}
        temp_config = {
            'trading.max_order_size': 1000,
            'trading.risk_limit': 0.1
        }

        errors = DependencyChecker._check_trading_dependencies(new_config, temp_config)
        assert errors == {}