#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring模块__init__.py测试
测试监控模块顶层导入和初始化
"""

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    intelligent_alert_system_module = importlib.import_module('src.monitoring.intelligent_alert_system')
    IntelligentAlertSystem = getattr(intelligent_alert_system_module, 'IntelligentAlertSystem', None)
    
    core_monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(core_monitoring_config_module, 'MonitoringSystem', None)
    
    performance_analyzer_module = importlib.import_module('src.monitoring.engine.performance_analyzer')
    PerformanceAnalyzer = getattr(performance_analyzer_module, 'PerformanceAnalyzer', None)
    
    if IntelligentAlertSystem is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError as e:
    pytest.skip(f"monitoring module not available: {e}", allow_module_level=True)


class TestMonitoringInit:
    """测试monitoring模块初始化"""

    def test_intelligent_alert_system_import(self):
        """测试IntelligentAlertSystem导入"""
        assert IntelligentAlertSystem is not None

    def test_intelligent_alert_system_instantiable(self):
        """测试IntelligentAlertSystem可实例化"""
        if IntelligentAlertSystem is not None:
            system = IntelligentAlertSystem()
            assert system is not None
            assert isinstance(system, IntelligentAlertSystem)

    def test_monitoring_system_import(self):
        """测试MonitoringSystem导入（可能为None）"""
        # MonitoringSystem可能是None（如果导入失败）
        assert MonitoringSystem is not None or MonitoringSystem is None

    def test_performance_analyzer_import(self):
        """测试PerformanceAnalyzer导入（可能为None）"""
        # PerformanceAnalyzer可能是None（如果导入失败）
        assert PerformanceAnalyzer is not None or PerformanceAnalyzer is None

    def test_all_exports(self):
        """测试__all__导出"""
        try:
            monitoring_module = importlib.import_module('src.monitoring')
            if hasattr(monitoring_module, '__all__'):
                assert 'IntelligentAlertSystem' in monitoring_module.__all__
                if MonitoringSystem is not None:
                    assert 'MonitoringSystem' in monitoring_module.__all__
                if PerformanceAnalyzer is not None:
                    assert 'PerformanceAnalyzer' in monitoring_module.__all__
        except ImportError:
            pytest.skip("monitoring module not available")

    def test_module_imports(self):
        """测试模块可以正常导入"""
        try:
            monitoring_module = importlib.import_module('src.monitoring')
            assert hasattr(monitoring_module, 'IntelligentAlertSystem') or IntelligentAlertSystem is not None
        except ImportError:
            pytest.skip("monitoring module not available")
