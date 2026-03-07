#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健康检查核心测试
测试健康检查的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestHealthCheckCore:
    """测试健康检查核心"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.health.health_check_core import HealthCheckCore
            self.HealthCheckCore = HealthCheckCore
        except ImportError:
            pytest.skip("HealthCheckCore not available")

    def test_initialization(self):
        """测试初始化"""
        if not hasattr(self, 'HealthCheckCore'):
            pytest.skip("HealthCheckCore not available")

        core = self.HealthCheckCore()
        assert core is not None

    def test_core_health_checks(self):
        """测试核心健康检查"""
        if not hasattr(self, 'HealthCheckCore'):
            pytest.skip("HealthCheckCore not available")

        core = self.HealthCheckCore()

        # 测试核心健康检查功能
        result = core.check_all_health()
        assert isinstance(result, dict)

    def test_core_functionality(self):
        """测试核心功能"""
        if not hasattr(self, 'HealthCheckCore'):
            pytest.skip("HealthCheckCore not available")

        core = self.HealthCheckCore()
        # 验证核心功能
        assert hasattr(core, 'register_provider')
        assert hasattr(core, 'get_overall_health')
        assert hasattr(core, 'get_health_summary')


if __name__ == '__main__':
    pytest.main([__file__])