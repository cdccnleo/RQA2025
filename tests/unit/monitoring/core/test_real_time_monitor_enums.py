#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor枚举类型测试
补充AlertLevel和MonitorType枚举的测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from enum import Enum

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
    trading_trading_monitor_module = importlib.import_module('src.monitoring.trading.trading_monitor')
    AlertLevel = getattr(trading_trading_monitor_module, 'AlertLevel', None)
    MonitorType = getattr(trading_trading_monitor_module, 'MonitorType', None)
    if AlertLevel is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestAlertLevelEnum:
    """测试AlertLevel枚举"""

    def test_alert_level_is_enum(self):
        """测试AlertLevel是枚举类型"""
        assert issubclass(AlertLevel, Enum)

    def test_alert_level_info(self):
        """测试INFO级别"""
        assert AlertLevel.INFO.value == "info"

    def test_alert_level_warning(self):
        """测试WARNING级别"""
        assert AlertLevel.WARNING.value == "warning"

    def test_alert_level_critical(self):
        """测试CRITICAL级别"""
        assert AlertLevel.CRITICAL.value == "critical"

    def test_alert_level_error(self):
        """测试ERROR级别"""
        assert AlertLevel.ERROR.value == "error"

    def test_alert_level_all_values(self):
        """测试所有AlertLevel值"""
        expected_values = {"info", "warning", "critical", "error"}
        actual_values = {level.value for level in AlertLevel}
        assert actual_values == expected_values or actual_values.issubset(expected_values)

    def test_alert_level_string_comparison(self):
        """测试AlertLevel与字符串比较"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"

    def test_alert_level_enumeration(self):
        """测试AlertLevel可以枚举"""
        levels = list(AlertLevel)
        assert len(levels) >= 4  # 至少包含4个级别

    def test_alert_level_name_access(self):
        """测试AlertLevel名称访问"""
        assert AlertLevel.INFO.name == "INFO"
        assert AlertLevel.WARNING.name == "WARNING"


class TestMonitorTypeEnum:
    """测试MonitorType枚举"""

    def test_monitor_type_is_enum(self):
        """测试MonitorType是枚举类型"""
        assert issubclass(MonitorType, Enum)

    def test_monitor_type_performance(self):
        """测试PERFORMANCE类型"""
        assert MonitorType.PERFORMANCE.value == "performance"

    def test_monitor_type_strategy(self):
        """测试STRATEGY类型"""
        assert MonitorType.STRATEGY.value == "strategy"

    def test_monitor_type_risk(self):
        """测试RISK类型"""
        assert MonitorType.RISK.value == "risk"

    def test_monitor_type_system(self):
        """测试SYSTEM类型"""
        assert MonitorType.SYSTEM.value == "system"

    def test_monitor_type_market(self):
        """测试MARKET类型"""
        assert MonitorType.MARKET.value == "market"

    def test_monitor_type_all_values(self):
        """测试所有MonitorType值"""
        expected_values = {"performance", "strategy", "risk", "system", "market"}
        actual_values = {monitor_type.value for monitor_type in MonitorType}
        assert actual_values == expected_values or actual_values.issubset(expected_values)

    def test_monitor_type_string_comparison(self):
        """测试MonitorType与字符串比较"""
        assert MonitorType.PERFORMANCE.value == "performance"
        assert MonitorType.SYSTEM.value == "system"

    def test_monitor_type_enumeration(self):
        """测试MonitorType可以枚举"""
        types = list(MonitorType)
        assert len(types) >= 4  # 至少包含4个类型

    def test_monitor_type_name_access(self):
        """测试MonitorType名称访问"""
        assert MonitorType.PERFORMANCE.name == "PERFORMANCE"
        assert MonitorType.SYSTEM.name == "SYSTEM"


class TestEnumIntegration:
    """测试枚举集成使用"""

    def test_alert_level_in_dict(self):
        """测试AlertLevel可以作为字典键或值"""
        alert_dict = {
            AlertLevel.INFO: "Informational message",
            AlertLevel.WARNING: "Warning message"
        }
        assert AlertLevel.INFO in alert_dict
        assert alert_dict[AlertLevel.INFO] == "Informational message"

    def test_monitor_type_in_dict(self):
        """测试MonitorType可以作为字典键或值"""
        monitor_dict = {
            MonitorType.PERFORMANCE: "Performance monitoring",
            MonitorType.SYSTEM: "System monitoring"
        }
        assert MonitorType.PERFORMANCE in monitor_dict

    def test_enum_value_consistency(self):
        """测试枚举值一致性"""
        # 验证每个枚举值都是字符串
        for level in AlertLevel:
            assert isinstance(level.value, str)
        
        for monitor_type in MonitorType:
            assert isinstance(monitor_type.value, str)

    def test_enum_unique_values(self):
        """测试枚举值唯一性"""
        alert_level_values = [level.value for level in AlertLevel]
        assert len(alert_level_values) == len(set(alert_level_values))
        
        monitor_type_values = [mt.value for mt in MonitorType]
        assert len(monitor_type_values) == len(set(monitor_type_values))

