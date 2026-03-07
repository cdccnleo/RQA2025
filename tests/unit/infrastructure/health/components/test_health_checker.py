#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""健康检查器组件测试"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.infrastructure.health.components.health_checker import (
    _CompatFloat,
    IHealthCheckProvider,
    IHealthCheckExecutor,
    IHealthCheckFramework,
    IHealthCheckerComponent,
    AsyncHealthCheckerComponent,
    HealthChecker,
    BatchHealthChecker,
    MonitoringHealthChecker,
    DEFAULT_SERVICE_TIMEOUT,
    DEFAULT_BATCH_TIMEOUT,
    DEFAULT_CONCURRENT_LIMIT,
    HEALTH_STATUS_UP,
    HEALTH_STATUS_DOWN
)


class TestCompatFloat:
    """测试兼容浮点数"""

    def test_compat_float_creation(self):
        """测试_CompatFloat创建"""
        cf = _CompatFloat(5.0, 10.0, 15.0)
        assert float(cf) == 5.0
        assert cf._aliases == (10.0, 15.0)

    def test_compat_float_equality(self):
        """测试_CompatFloat相等性"""
        cf = _CompatFloat(5.0, 10.0)
        assert cf == 5.0
        assert cf == 10.0
        assert cf != 15.0

    def test_compat_float_hash(self):
        """测试_CompatFloat哈希"""
        cf = _CompatFloat(5.0)
        assert hash(cf) == hash(5.0)


class TestConstants:
    """测试常量定义"""

    def test_default_constants(self):
        """测试默认常量值"""
        assert DEFAULT_SERVICE_TIMEOUT == 5.0
        assert DEFAULT_BATCH_TIMEOUT == 30.0
        assert DEFAULT_CONCURRENT_LIMIT == 10

    def test_health_status_constants(self):
        """测试健康状态常量"""
        assert HEALTH_STATUS_UP == "UP"
        assert HEALTH_STATUS_DOWN == "DOWN"


class TestIHealthCheckProvider:
    """测试健康检查提供者接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        with pytest.raises(TypeError):
            IHealthCheckProvider()

    def test_interface_has_required_methods(self):
        """测试接口定义了所需的方法"""
        # 检查方法是否存在
        assert hasattr(IHealthCheckProvider, 'check_health_async')
        assert hasattr(IHealthCheckProvider, 'check_health_sync')
        assert hasattr(IHealthCheckProvider, 'get_health_metrics')


class TestIHealthCheckExecutor:
    """测试健康检查执行器接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        with pytest.raises(TypeError):
            IHealthCheckExecutor()

    def test_interface_has_required_methods(self):
        """测试接口定义了所需的方法"""
        assert hasattr(IHealthCheckExecutor, 'execute_check')
        assert hasattr(IHealthCheckExecutor, 'get_check_status')
        assert hasattr(IHealthCheckExecutor, 'register_service')


class TestIHealthCheckFramework:
    """测试健康检查框架接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        with pytest.raises(TypeError):
            IHealthCheckFramework()

    def test_interface_has_required_methods(self):
        """测试接口定义了所需的方法"""
        assert hasattr(IHealthCheckFramework, 'initialize')
        assert hasattr(IHealthCheckFramework, 'shutdown')
        assert hasattr(IHealthCheckFramework, 'perform_health_check')


class TestIHealthCheckerComponent:
    """测试健康检查器组件接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        with pytest.raises(TypeError):
            IHealthCheckerComponent()

    def test_interface_inheritance(self):
        """测试接口继承关系"""
        assert issubclass(IHealthCheckerComponent, IHealthCheckProvider)
        assert issubclass(IHealthCheckerComponent, IHealthCheckExecutor)

    def test_interface_has_combined_methods(self):
        """测试接口组合了所有方法"""
        # Provider方法
        assert hasattr(IHealthCheckerComponent, 'check_health_async')
        assert hasattr(IHealthCheckerComponent, 'check_health_sync')
        assert hasattr(IHealthCheckerComponent, 'get_health_metrics')

        # Executor方法
        assert hasattr(IHealthCheckerComponent, 'execute_check')
        assert hasattr(IHealthCheckerComponent, 'get_check_status')
        assert hasattr(IHealthCheckerComponent, 'register_service')


class TestAsyncHealthCheckerComponent:
    """测试异步健康检查器组件"""

    def setup_method(self):
        """测试前准备"""
        self.service_name = "test_async_checker"
        self.component = AsyncHealthCheckerComponent(self.service_name)

    def test_init_with_service_name(self):
        """测试使用服务名初始化"""
        assert self.component.service_name == self.service_name
        assert isinstance(self.component.check_functions, dict)
        assert self.component.running is False

    def test_init_without_service_name(self):
        """测试无服务名初始化"""
        component = AsyncHealthCheckerComponent()
        assert component.service_name == "async_checker"
        assert component.running is False

    def test_register_check_function(self):
        """测试注册检查函数"""
        check_func = Mock(return_value={'status': 'UP'})
        self.component.register_check_function('test_check', check_func)
        assert 'test_check' in self.component.check_functions

    def test_unregister_check_function(self):
        """测试注销检查函数"""
        check_func = Mock(return_value={'status': 'UP'})
        self.component.register_check_function('test_check', check_func)
        result = self.component.unregister_check_function('test_check')
        assert result is True
        assert 'test_check' not in self.component.check_functions

    def test_get_registered_checks(self):
        """测试获取注册的检查"""
        check_func = Mock(return_value={'status': 'UP'})
        self.component.register_check_function('check1', check_func)
        self.component.register_check_function('check2', check_func)

        checks = list(self.component.get_registered_checks())
        assert len(checks) == 2
        assert 'check1' in checks
        assert 'check2' in checks

    def test_get_monitoring_status(self):
        """测试获取监控状态"""
        status = self.component.get_monitoring_status()
        assert isinstance(status, dict)
        assert 'running' in status

    def test_get_health_status_summary(self):
        """测试获取健康状态摘要"""
        summary = self.component.get_health_status_summary()
        assert isinstance(summary, dict)
        assert 'service' in summary


class TestHealthChecker:
    """测试基础健康检查器"""

    def test_init(self):
        """测试初始化"""
        checker = HealthChecker()
        assert isinstance(checker, HealthChecker)


class TestBatchHealthChecker:
    """测试批量健康检查器"""

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(BatchHealthChecker, HealthChecker)

    def test_init(self):
        """测试初始化"""
        checker = BatchHealthChecker()
        assert isinstance(checker, HealthChecker)


class TestMonitoringHealthChecker:
    """测试监控健康检查器"""

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(MonitoringHealthChecker, HealthChecker)

    def test_init(self):
        """测试初始化"""
        checker = MonitoringHealthChecker()
        assert isinstance(checker, HealthChecker)
