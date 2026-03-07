from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

from src.infrastructure.config.environment.__init__ import (
    EnvironmentType,
    get_current_environment,
    is_production,
    is_development,
    is_testing,
    is_staging,
    get_environment_config_path,
    ConfigEnvironment
)


class TestEnvironmentType:
    """测试环境类型枚举"""
    
    def test_environment_type_values(self):
        """测试环境类型枚举值"""
        assert EnvironmentType.DEVELOPMENT.value == "development"
        assert EnvironmentType.TESTING.value == "testing"
        assert EnvironmentType.STAGING.value == "staging"
        assert EnvironmentType.PRODUCTION.value == "production"


class TestEnvironmentFunctions:
    """测试环境检测函数"""

    def test_get_current_environment_default(self):
        """测试获取默认环境 (覆盖28-48)"""
        with patch.dict(os.environ, {}, clear=True):
            result = get_current_environment()
            assert result == EnvironmentType.DEVELOPMENT

    def test_get_current_environment_production(self):
        """测试获取生产环境"""
        with patch.dict(os.environ, {'ENV': 'production'}):
            result = get_current_environment()
            assert result == EnvironmentType.PRODUCTION

    def test_get_current_environment_config_env(self):
        """测试CONFIG_ENV变量优先级"""
        with patch.dict(os.environ, {'ENV': 'development', 'CONFIG_ENV': 'production'}):
            result = get_current_environment()
            assert result == EnvironmentType.PRODUCTION

    def test_get_current_environment_variants(self):
        """测试环境名称变体"""
        variants = [
            ('dev', EnvironmentType.DEVELOPMENT),
            ('stage', EnvironmentType.STAGING),
            ('prod', EnvironmentType.PRODUCTION),
            ('test', EnvironmentType.TESTING),
        ]
        
        for env_var, expected in variants:
            with patch.dict(os.environ, {'ENV': env_var}):
                result = get_current_environment()
                assert result == expected

    def test_is_production_function(self):
        """测试生产环境检测 (覆盖51-58)"""
        with patch.dict(os.environ, {'ENV': 'production'}):
            assert is_production() is True
        
        with patch.dict(os.environ, {'ENV': 'development'}):
            assert is_production() is False

    def test_is_development_function(self):
        """测试开发环境检测 (覆盖61-68)"""
        with patch.dict(os.environ, {'ENV': 'development'}):
            assert is_development() is True
        
        with patch.dict(os.environ, {'ENV': 'production'}):
            assert is_development() is False

    def test_is_testing_function_with_pytest(self):
        """测试测试环境检测 (覆盖71-82)"""
        # 测试pytest模块检测
        with patch('sys.modules', {'pytest': MagicMock()}):
            assert is_testing() is True
        
        # 测试sys.argv检测
        with patch('sys.argv', ['pytest', 'test_file.py']):
            assert is_testing() is True
        
        # 测试正常环境
        with patch.dict(os.environ, {'ENV': 'testing'}):
            with patch('sys.modules', {}), patch('sys.argv', []):
                assert is_testing() is True

    def test_is_staging_function(self):
        """测试预发布环境检测 (覆盖85-92)"""
        with patch.dict(os.environ, {'ENV': 'staging'}):
            assert is_staging() is True
        
        with patch.dict(os.environ, {'ENV': 'production'}):
            assert is_staging() is False

    def test_get_environment_config_path_found(self):
        """测试获取环境配置文件路径 (覆盖95-122)"""
        test_config_content = '{"key": "value"}'
        
        with patch('os.path.exists') as mock_exists, \
             patch('os.path.dirname') as mock_dirname:
            mock_exists.return_value = True
            mock_dirname.return_value = '/test/path'
            
            with patch.dict(os.environ, {'ENV': 'development'}):
                result = get_environment_config_path()
                assert result is not None

    def test_get_environment_config_path_not_found(self):
        """测试配置文件不存在情况"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = get_environment_config_path()
            assert result is None


class TestConfigEnvironment:
    """测试配置环境管理器"""

    @pytest.fixture
    def env_manager(self):
        """创建ConfigEnvironment实例"""
        return ConfigEnvironment()

    def test_init(self, env_manager):
        """测试初始化 (覆盖124-130)"""
        assert hasattr(env_manager, '_env_cache')
        assert hasattr(env_manager, '_env_vars')
        assert isinstance(env_manager._env_cache, dict)
        assert isinstance(env_manager._env_vars, dict)

    def test_get_environment_default(self, env_manager):
        """测试获取默认环境 (覆盖131-133)"""
        with patch.dict(os.environ, {}, clear=True):
            result = env_manager.get_environment()
            assert result == 'development'

    def test_get_environment_custom(self, env_manager):
        """测试获取自定义环境"""
        with patch.dict(os.environ, {'ENV': 'staging'}):
            result = env_manager.get_environment()
            assert result == 'staging'

    def test_is_production_method(self, env_manager):
        """测试生产环境检测方法 (覆盖135-137)"""
        with patch.object(env_manager, 'get_environment') as mock_get_env:
            mock_get_env.return_value = 'production'
            assert env_manager.is_production() is True
            
            mock_get_env.return_value = 'development'
            assert env_manager.is_production() is False

    def test_is_development_method(self, env_manager):
        """测试开发环境检测方法 (覆盖139-141)"""
        with patch.object(env_manager, 'is_production') as mock_is_prod:
            mock_is_prod.return_value = False
            assert env_manager.is_development() is True
            
            mock_is_prod.return_value = True
            assert env_manager.is_development() is False

    def test_is_testing_method(self, env_manager):
        """测试测试环境检测方法 (覆盖143-145)"""
        with patch.dict(os.environ, {'PYTEST_CURRENT_TEST': 'test_function'}):
            assert env_manager.is_testing() is True
        
        with patch.dict(os.environ, {}, clear=True):
            assert env_manager.is_testing() is False

    def test_get_env_var_cached(self, env_manager):
        """测试获取环境变量缓存 (覆盖147-151)"""
        with patch.object(os, 'getenv') as mock_getenv:
            mock_getenv.return_value = 'test_value'
            
            # 第一次调用
            result1 = env_manager.get_env_var('TEST_KEY')
            # 第二次调用应该使用缓存，不会再次调用os.getenv
            result2 = env_manager.get_env_var('TEST_KEY')
            
            assert result1 == 'test_value'
            assert result2 == 'test_value'
            # 由于缓存机制，只调用一次os.getenv
            assert mock_getenv.call_count == 1

    def test_get_env_var_default(self, env_manager):
        """测试获取环境变量默认值"""
        # 由于os.getenv会在找不到变量时返回默认值，我们需要模拟这个行为
        def mock_getenv(key, default=""):
            if key == 'NONEXISTENT_KEY':
                return default
            return None
        
        with patch.object(os, 'getenv', side_effect=mock_getenv):
            result = env_manager.get_env_var('NONEXISTENT_KEY', 'default_value')
            assert result == 'default_value'

    def test_set_env_var_success(self, env_manager):
        """测试设置环境变量成功 (覆盖153-160)"""
        with patch.dict(os.environ, {}, clear=True):
            result = env_manager.set_env_var('TEST_KEY', 'test_value')
            assert result is True
            assert os.environ['TEST_KEY'] == 'test_value'
            assert env_manager._env_cache['TEST_KEY'] == 'test_value'

    def test_set_env_var_exception(self, env_manager):
        """测试设置环境变量异常"""
        # 直接测试set_env_var方法的异常处理
        # 由于原方法有try-except包装，我们需要模拟os.environ设置失败
        class FailingOSEnviron:
            def __setitem__(self, key, value):
                raise PermissionError("Permission denied")
        
        with patch('os.environ', FailingOSEnviron()):
            result = env_manager.set_env_var('TEST_KEY', 'test_value')
            assert result is False

    def test_get_config_for_environment(self, env_manager):
        """测试根据环境获取配置 (覆盖162-171)"""
        base_config = {
            'common': 'value',
            'development': {'key': 'dev_value'},
            'production': {'key': 'prod_value'}
        }
        
        with patch.object(env_manager, 'get_environment') as mock_get_env:
            mock_get_env.return_value = 'development'
            result = env_manager.get_config_for_environment(base_config)
            
            assert result['common'] == 'value'
            assert result['key'] == 'dev_value'

    def test_get_config_for_environment_no_overrides(self, env_manager):
        """测试无环境覆盖的情况"""
        base_config = {'common': 'value'}
        
        with patch.object(env_manager, 'get_environment') as mock_get_env:
            mock_get_env.return_value = 'staging'
            result = env_manager.get_config_for_environment(base_config)
            
            assert result == base_config

    def test_get_environment_info(self, env_manager):
        """测试获取环境信息 (覆盖173-182)"""
        with patch.object(env_manager, 'get_environment') as mock_get_env, \
             patch.object(env_manager, 'is_production') as mock_is_prod, \
             patch.object(env_manager, 'is_development') as mock_is_dev, \
             patch.object(env_manager, 'is_testing') as mock_is_test, \
             patch.object(sys, 'version', '3.9.0'), \
             patch.object(sys, 'platform', 'win32'):
            
            mock_get_env.return_value = 'testing'
            mock_is_prod.return_value = False
            mock_is_dev.return_value = False
            mock_is_test.return_value = True
            
            result = env_manager.get_environment_info()
            
            assert result['environment'] == 'testing'
            assert result['is_production'] is False
            assert result['is_development'] is False
            assert result['is_testing'] is True
            assert 'python_version' in result
            assert 'platform' in result
