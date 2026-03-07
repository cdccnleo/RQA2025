#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层health/__init__.py模块测试

测试目标：提升health/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.health模块
"""

import pytest


class TestHealthInit:
    """测试health模块初始化"""
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.health import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        expected_exports = [
            "HealthCheck",
            "HealthCheckCore",
            "MonitoringDashboard",
            "HealthCheckResult",
            "CheckType",
            "HealthStatus",
            "HealthStatusEnum",
            "MetricsCollector",
            "MetricType",
            "FastAPIHealthChecker",
            "EnhancedHealthChecker",
            "DatabaseHealthMonitor",
            "HealthCheckPrometheusExporter",
            "get_status",
            "is_available"
        ]
        for export in expected_exports:
            assert export in __all__, f"{export} should be in __all__"
    
    def test_get_status_function(self):
        """测试get_status函数"""
        from src.infrastructure.health import get_status
        
        assert callable(get_status)
        # 测试函数调用（可能返回默认状态）
        result = get_status()
        assert isinstance(result, dict)
        assert "status" in result
    
    def test_is_available_function(self):
        """测试is_available函数"""
        from src.infrastructure.health import is_available
        
        assert callable(is_available)
        # 测试函数调用（可能返回默认值）
        result = is_available()
        assert isinstance(result, bool)
    
    def test_module_version(self):
        """测试模块版本"""
        from src.infrastructure.health import __version__
        
        assert isinstance(__version__, str)
        assert len(__version__) > 0

