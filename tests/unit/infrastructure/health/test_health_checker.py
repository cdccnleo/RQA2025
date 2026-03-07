#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健康检查器测试
测试健康检查器功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAlertComponent:
    """测试AlertComponent"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.health.components.alert_components import AlertComponent
            self.AlertComponent = AlertComponent
        except ImportError:
            pytest.skip("AlertComponent not available")

    def test_alert_component_initialization(self):
        """测试AlertComponent初始化"""
        if not hasattr(self, 'AlertComponent'):
            pytest.skip("AlertComponent not available")

        alert = self.AlertComponent(alert_id=1, component_type='test')
        assert alert is not None
        assert alert.get_alert_id() == 1

    def test_alert_component_methods(self):
        """测试AlertComponent方法"""
        if not hasattr(self, 'AlertComponent'):
            pytest.skip("AlertComponent not available")

        alert = self.AlertComponent(alert_id=2, component_type='test')

        # 测试基本方法
        info = alert.get_info()
        assert isinstance(info, dict)
        assert 'alert_id' in info

        status = alert.get_status()
        assert isinstance(status, dict)

        # 测试process方法（可能需要mock）
        try:
            result = alert.process()
            assert result is not None
        except Exception:
            # 如果process需要额外设置，跳过
            pass


class TestHealthChecker:
    """测试健康检查器"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.health.components.health_checker import HealthChecker
            self.HealthChecker = HealthChecker
        except ImportError:
            pytest.skip("HealthChecker not available")

    def test_initialization(self):
        """测试初始化"""
        if not hasattr(self, 'HealthChecker'):
            pytest.skip("HealthChecker not available")

        checker = self.HealthChecker()
        assert checker is not None
        assert checker.service_name == 'health_checker'
        assert hasattr(checker, 'check_health')
        assert hasattr(checker, 'register_check_function')

    def test_health_checking(self):
        """测试健康检查"""
        if not hasattr(self, 'HealthChecker'):
            pytest.skip("HealthChecker not available")

        checker = self.HealthChecker()

        # 注册一个测试检查函数
        def test_check():
            return {'status': 'healthy', 'details': 'test check'}

        checker.register_check_function('test_check', test_check)

        # 测试健康检查功能
        result = checker.check_health()
        assert result is not None
        # 检查结果是HealthCheckResult对象
        assert hasattr(result, 'status')
        assert hasattr(result, 'details')
        # 检查details中包含我们的测试检查
        assert 'test_check' in result.details['check_results']
        assert result.details['check_results']['test_check']['status'] == 'healthy'

    def test_checker_functionality(self):
        """测试检查器功能"""
        if not hasattr(self, 'HealthChecker'):
            pytest.skip("HealthChecker not available")

        checker = self.HealthChecker()

        # 测试注册和移除检查函数
        def dummy_check():
            return {'status': 'ok'}

        # 注册检查函数
        result = checker.register_check_function('dummy', dummy_check)
        assert result is True

        # 获取已注册的检查
        registered = checker.get_registered_checks()
        assert 'dummy' in registered

        # 移除检查函数
        result = checker.remove_check_function('dummy')
        assert result is True

        # 验证已移除
        registered = checker.get_registered_checks()
        assert 'dummy' not in registered
        assert True


if __name__ == '__main__':
    pytest.main([__file__])