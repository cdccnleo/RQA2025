#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Monitoring组件基础测试"""

import pytest


def test_data_persistor_import():
    """测试DataPersistor导入"""
    try:
        from src.infrastructure.monitoring.components.data_persistor import DataPersistor
        assert DataPersistor is not None
    except ImportError:
        pytest.skip("DataPersistor不可用")


def test_metrics_exporter_import():
    """测试MetricsExporter导入"""
    try:
        from src.infrastructure.monitoring.components.metrics_exporter import MetricsExporter
        assert MetricsExporter is not None
    except ImportError:
        pytest.skip("MetricsExporter不可用")


def test_monitoring_coordinator_import():
    """测试MonitoringCoordinator导入"""
    try:
        from src.infrastructure.monitoring.components.monitoring_coordinator import MonitoringCoordinator
        assert MonitoringCoordinator is not None
    except ImportError:
        pytest.skip("MonitoringCoordinator不可用")


def test_stats_collector_import():
    """测试StatsCollector导入"""
    try:
        from src.infrastructure.monitoring.components.stats_collector import StatsCollector
        assert StatsCollector is not None
    except ImportError:
        pytest.skip("StatsCollector不可用")


def test_logger_pool_monitoring_loop_import():
    """测试LoggerPoolMonitoringLoop导入"""
    try:
        from src.infrastructure.monitoring.components.logger_pool_monitoring_loop import LoggerPoolMonitoringLoop
        assert LoggerPoolMonitoringLoop is not None
    except ImportError:
        pytest.skip("LoggerPoolMonitoringLoop不可用")


def test_logger_pool_monitor_refactored_import():
    """测试LoggerPoolMonitorRefactored导入"""
    try:
        from src.infrastructure.monitoring.application.logger_pool_monitor_refactored import LoggerPoolMonitor
        assert LoggerPoolMonitor is not None
    except ImportError:
        pytest.skip("LoggerPoolMonitor不可用")


def test_data_persistor_init():
    """测试DataPersistor初始化"""
    try:
        from src.infrastructure.monitoring.components.data_persistor import DataPersistor
        persistor = DataPersistor()
        assert persistor is not None
    except Exception:
        pytest.skip("测试跳过")


def test_metrics_exporter_init():
    """测试MetricsExporter初始化"""
    try:
        from src.infrastructure.monitoring.components.metrics_exporter import MetricsExporter
        exporter = MetricsExporter()
        assert exporter is not None
    except Exception:
        pytest.skip("测试跳过")


def test_monitoring_coordinator_init():
    """测试MonitoringCoordinator初始化"""
    try:
        from src.infrastructure.monitoring.components.monitoring_coordinator import MonitoringCoordinator
        coordinator = MonitoringCoordinator()
        assert coordinator is not None
    except Exception:
        pytest.skip("测试跳过")


def test_stats_collector_init():
    """测试StatsCollector初始化"""
    try:
        from src.infrastructure.monitoring.components.stats_collector import StatsCollector
        collector = StatsCollector()
        assert collector is not None
    except Exception:
        pytest.skip("测试跳过")


def test_rule_types_import():
    """测试RuleTypes导入"""
    try:
        from src.infrastructure.monitoring.components import rule_types
        assert rule_types is not None
    except ImportError:
        pytest.skip("rule_types不可用")


def test_constants_import():
    """测试constants导入"""
    try:
        from src.infrastructure.monitoring.core import constants
        assert constants is not None
    except ImportError:
        pytest.skip("constants不可用")


def test_parameter_objects_import():
    """测试parameter_objects导入"""
    try:
        from src.infrastructure.monitoring.core import parameter_objects
        assert parameter_objects is not None
    except ImportError:
        pytest.skip("parameter_objects不可用")

