"""
测试environment.py模块的基础功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import os
import pytest
from unittest.mock import patch

try:
    from src.infrastructure.config.environment import (
        is_production,
        is_development,
        is_testing
    )
except ImportError:
    is_production = None
    is_development = None
    is_testing = None


class TestEnvironmentSimple:
    """测试environment.py模块的基础功能"""

    def setup_method(self):
        """测试前准备"""
        if is_production is None:
            pytest.skip("environment模块导入失败，跳过测试")

    @patch.dict(os.environ, {'ENV': 'production'})
    def test_is_production_true_with_production_env(self):
        """测试生产环境检测为真"""
        assert is_production() is True

    @patch.dict(os.environ, {'ENV': 'PRODUCTION'})
    def test_is_production_true_with_uppercase(self):
        """测试生产环境检测为真（大写）"""
        assert is_production() is True

    @patch.dict(os.environ, {'ENV': 'development'})
    def test_is_production_false_with_development_env(self):
        """测试生产环境检测为假（开发环境）"""
        assert is_production() is False

    @patch.dict(os.environ, {'ENV': 'testing'})
    def test_is_production_false_with_testing_env(self):
        """测试生产环境检测为假（测试环境）"""
        assert is_production() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_production_false_with_no_env(self):
        """测试生产环境检测为假（无环境变量，默认development）"""
        assert is_production() is False

    @patch.dict(os.environ, {'ENV': ''})
    def test_is_production_false_with_empty_env(self):
        """测试生产环境检测为假（空环境变量）"""
        assert is_production() is False

    def test_is_production_false_with_none_env(self):
        """测试生产环境检测为假（None环境变量）"""
        with patch('os.getenv', return_value=None):
            assert is_production() is False

    @patch.dict(os.environ, {'ENV': 'development'})
    def test_is_development_true_with_development_env(self):
        """测试开发环境检测为真"""
        assert is_development() is True

    @patch.dict(os.environ, {'ENV': 'testing'})
    def test_is_development_true_with_testing_env(self):
        """测试开发环境检测为真（测试环境被视为非生产环境）"""
        assert is_development() is True

    @patch.dict(os.environ, {'ENV': 'production'})
    def test_is_development_false_with_production_env(self):
        """测试开发环境检测为假（生产环境）"""
        assert is_development() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_development_true_with_no_env(self):
        """测试开发环境检测为真（无环境变量默认为开发环境）"""
        assert is_development() is True

    def test_is_development_none_env_variable(self):
        """测试开发环境检测（None环境变量）"""
        with patch('os.getenv', return_value=None):
            # None环境变量不是production，所以is_development应该返回True
            assert is_development() is True

    @patch.dict(os.environ, {'PYTEST_CURRENT_TEST': 'test_module.py::test_function'})
    def test_is_testing_true_with_pytest_env(self):
        """测试测试环境检测为真（有PYTEST_CURRENT_TEST环境变量）"""
        assert is_testing() is True

    def test_is_testing_false_without_pytest_env(self):
        """测试测试环境检测为假（无PYTEST_CURRENT_TEST环境变量）"""
        # 在pytest环境中，is_testing会检查sys.modules中的pytest，所以会返回True
        # 我们需要mock sys.modules来模拟非pytest环境
        with patch('sys.modules', {}), patch('sys.argv', ['python', 'script.py']):
            with patch('os.environ.get', return_value=None):
                assert is_testing() is False

    @patch.dict(os.environ, {'PYTEST_CURRENT_TEST': ''})
    def test_is_testing_true_with_empty_pytest_env(self):
        """测试测试环境检测为真（空字符串不等于None）"""
        assert is_testing() is True

    def test_environment_functions_consistency(self):
        """测试环境函数的一致性"""
        # 测试环境变量为None的情况
        with patch('os.getenv', return_value=None):
            assert is_production() is False
            assert is_development() is True
            # is_testing依赖于PYTEST_CURRENT_TEST环境变量
        
        # 测试生产环境
        with patch('os.getenv', return_value='production'):
            assert is_production() is True
            assert is_development() is False

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空字符串
        with patch('os.getenv', return_value=''):
            assert is_production() is False
            assert is_development() is True

        # 测试带空格的字符串
        with patch('os.getenv', return_value=' production '):
            assert is_production() is False  # 因为' production '.lower() != 'production'
            assert is_development() is True

    def test_is_testing_edge_cases(self):
        """测试is_testing的边界情况"""
        # 测试environ.get返回None的情况，需要mock非pytest环境
        with patch('sys.modules', {}), patch('sys.argv', ['python', 'script.py']):
            with patch('os.environ.get', return_value=None):
                assert is_testing() is False
        
        # 测试environ.get返回空字符串的情况
        with patch('os.environ.get', return_value=''):
            # 空字符串不是None，所以应该返回True
            assert is_testing() is True
