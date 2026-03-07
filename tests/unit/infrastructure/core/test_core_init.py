#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层core/__init__.py模块测试

测试目标：提升core/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.core模块
"""

import pytest


class TestCoreInit:
    """测试core模块初始化"""
    
    def test_infrastructure_service_provider_import(self):
        """测试InfrastructureServiceProvider导入"""
        from src.infrastructure.core import InfrastructureServiceProvider
        
        assert InfrastructureServiceProvider is not None
    
    def test_infrastructure_service_status_import(self):
        """测试InfrastructureServiceStatus导入"""
        from src.infrastructure.core import InfrastructureServiceStatus
        
        assert InfrastructureServiceStatus is not None
    
    def test_get_infrastructure_service_provider_import(self):
        """测试get_infrastructure_service_provider函数导入"""
        from src.infrastructure.core import get_infrastructure_service_provider
        
        assert callable(get_infrastructure_service_provider)
    
    def test_health_check_result_import(self):
        """测试HealthCheckResult导入"""
        from src.infrastructure.core import HealthCheckResult
        
        assert HealthCheckResult is not None
    
    def test_health_check_interface_import(self):
        """测试HealthCheckInterface导入"""
        from src.infrastructure.core import HealthCheckInterface
        
        assert HealthCheckInterface is not None
    
    def test_infrastructure_health_checker_import(self):
        """测试InfrastructureHealthChecker导入"""
        from src.infrastructure.core import InfrastructureHealthChecker
        
        assert InfrastructureHealthChecker is not None
    
    def test_infrastructure_exception_import(self):
        """测试InfrastructureException导入"""
        from src.infrastructure.core import InfrastructureException
        
        assert InfrastructureException is not None
    
    def test_configuration_error_import(self):
        """测试ConfigurationError导入"""
        from src.infrastructure.core import ConfigurationError
        
        assert ConfigurationError is not None
    
    def test_cache_error_import(self):
        """测试CacheError导入"""
        from src.infrastructure.core import CacheError
        
        assert CacheError is not None
    
    def test_cache_constants_import(self):
        """测试CacheConstants导入"""
        from src.infrastructure.core import CacheConstants
        
        assert CacheConstants is not None
    
    def test_config_constants_import(self):
        """测试ConfigConstants导入"""
        from src.infrastructure.core import ConfigConstants
        
        assert ConfigConstants is not None
    
    def test_default_timeout_import(self):
        """测试DEFAULT_TIMEOUT导入"""
        from src.infrastructure.core import DEFAULT_TIMEOUT
        
        assert DEFAULT_TIMEOUT is not None
    
    def test_health_check_params_import(self):
        """测试HealthCheckParams导入"""
        from src.infrastructure.core import HealthCheckParams
        
        assert HealthCheckParams is not None
    
    def test_base_mock_service_import(self):
        """测试BaseMockService导入"""
        from src.infrastructure.core import BaseMockService
        
        assert BaseMockService is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.core import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "InfrastructureServiceProvider" in __all__
        assert "HealthCheckResult" in __all__
        assert "InfrastructureException" in __all__

