#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层ops/__init__.py模块测试

测试目标：提升ops/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.ops模块
"""

import pytest


class TestOpsInit:
    """测试ops模块初始化"""
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.ops import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        # 检查预期的导出项
        expected_exports = [
            'MonitoringDashboard',
            'DashboardConfig',
            'Metric',
            'Alert',
            'MetricType',
            'AlertSeverity'
        ]
        for export in expected_exports:
            assert export in __all__, f"{export} should be in __all__"

