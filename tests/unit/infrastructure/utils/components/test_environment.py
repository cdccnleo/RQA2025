#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层环境检测组件测试

测试目标：提升utils/components/environment.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.environment模块
"""

import pytest
import os
from unittest.mock import patch


class TestEnvironmentDetection:
    """测试环境检测函数"""
    
    def test_is_production(self):
        """测试生产环境检测"""
        from src.infrastructure.utils.components.environment import is_production
        
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert is_production() is True
        
        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
            assert is_production() is True
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_production() is False
        
        with patch.dict(os.environ, {}, clear=True):
            assert is_production() is False
    
    def test_is_development(self):
        """测试开发环境检测"""
        from src.infrastructure.utils.components.environment import is_development
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_development() is True
        
        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}):
            assert is_development() is True
        
        with patch.dict(os.environ, {}, clear=True):
            assert is_development() is True
        
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert is_development() is False
    
    def test_is_testing(self):
        """测试测试环境检测"""
        from src.infrastructure.utils.components.environment import is_testing
        
        with patch.dict(os.environ, {"ENVIRONMENT": "testing"}):
            assert is_testing() is True
        
        with patch.dict(os.environ, {"ENVIRONMENT": "test"}):
            assert is_testing() is True
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_testing() is False
        
        with patch.dict(os.environ, {}, clear=True):
            assert is_testing() is False
    
    def test_get_environment(self):
        """测试获取环境名称"""
        from src.infrastructure.utils.components.environment import get_environment
        
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert get_environment() == "production"
        
        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
            assert get_environment() == "production"
        
        with patch.dict(os.environ, {"ENVIRONMENT": "testing"}):
            assert get_environment() == "testing"
        
        with patch.dict(os.environ, {"ENVIRONMENT": "test"}):
            assert get_environment() == "testing"
        
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert get_environment() == "development"
        
        with patch.dict(os.environ, {}, clear=True):
            assert get_environment() == "development"


class TestConfigFunctions:
    """测试配置函数"""
    
    def test_get_config_value(self):
        """测试获取配置值"""
        from src.infrastructure.utils.components.environment import get_config_value
        
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            assert get_config_value("TEST_KEY") == "test_value"
        
        assert get_config_value("NON_EXISTENT_KEY", "default") == "default"
        assert get_config_value("NON_EXISTENT_KEY") is None
    
    def test_get_database_config(self):
        """测试获取数据库配置"""
        from src.infrastructure.utils.components.environment import get_database_config
        
        with patch.dict(os.environ, {
            "DB_HOST": "test_host",
            "DB_PORT": "5433",
            "DB_NAME": "test_db",
            "DB_USER": "test_user",
            "DB_PASSWORD": "test_pass",
            "DB_SSL_MODE": "require"
        }):
            config = get_database_config()
            assert config["host"] == "test_host"
            assert config["port"] == 5433
            assert config["name"] == "test_db"
            assert config["user"] == "test_user"
            assert config["password"] == "test_pass"
            assert config["ssl_mode"] == "require"
    
    def test_get_database_config_defaults(self):
        """测试获取数据库配置默认值"""
        from src.infrastructure.utils.components.environment import get_database_config
        
        with patch.dict(os.environ, {}, clear=True):
            config = get_database_config()
            assert config["host"] == "localhost"
            assert config["port"] == 5432
            assert config["name"] == "rqa2025"
            assert config["user"] == "rqa2025"
            assert config["password"] == ""
            assert config["ssl_mode"] == "prefer"
    
    def test_get_database_config_invalid_port(self):
        """测试获取数据库配置无效端口"""
        from src.infrastructure.utils.components.environment import get_database_config
        
        with patch.dict(os.environ, {"DB_PORT": "invalid"}):
            config = get_database_config()
            # 应该返回默认配置
            assert config["port"] == 5432
    
    def test_get_redis_config(self):
        """测试获取Redis配置"""
        from src.infrastructure.utils.components.environment import get_redis_config
        
        with patch.dict(os.environ, {
            "REDIS_HOST": "redis_host",
            "REDIS_PORT": "6380",
            "REDIS_DB": "1",
            "REDIS_PASSWORD": "redis_pass"
        }):
            config = get_redis_config()
            assert config["host"] == "redis_host"
            assert config["port"] == 6380
            assert config["db"] == 1
            assert config["password"] == "redis_pass"
            assert config["decode_responses"] is True
    
    def test_get_redis_config_defaults(self):
        """测试获取Redis配置默认值"""
        from src.infrastructure.utils.components.environment import get_redis_config
        
        with patch.dict(os.environ, {}, clear=True):
            config = get_redis_config()
            assert config["host"] == "localhost"
            assert config["port"] == 6379
            assert config["db"] == 0
            assert config["password"] == ""
            assert config["decode_responses"] is True
    
    def test_get_redis_config_invalid_port(self):
        """测试获取Redis配置无效端口"""
        from src.infrastructure.utils.components.environment import get_redis_config
        
        with patch.dict(os.environ, {"REDIS_PORT": "invalid"}):
            config = get_redis_config()
            # 应该返回默认配置
            assert config["port"] == 6379
