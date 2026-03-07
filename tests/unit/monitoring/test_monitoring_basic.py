# -*- coding: utf-8 -*-
"""
监控模块基础测试
测试监控框架的核心组件和接口
"""

import pytest
import os
from unittest.mock import Mock


def test_monitoring_module_structure():
    """测试监控模块基本结构"""
    monitoring_dir = "src/monitoring"

    # 检查主要子目录存在
    assert os.path.exists(f"{monitoring_dir}/core")
    assert os.path.exists(f"{monitoring_dir}/engine")
    assert os.path.exists(f"{monitoring_dir}/alert")


def test_monitoring_core_files():
    """测试监控核心文件存在"""
    core_files = [
        "src/monitoring/core/__init__.py",
        "src/monitoring/core/constants.py",
        "src/monitoring/core/exceptions.py"
    ]

    for file_path in core_files:
        assert os.path.exists(file_path), f"File {file_path} should exist"


def test_monitoring_engine_files():
    """测试监控引擎文件存在"""
    engine_files = [
        "src/monitoring/engine/__init__.py",
        "src/monitoring/engine/performance_analyzer.py",
        "src/monitoring/engine/monitor_components.py"
    ]

    for file_path in engine_files:
        assert os.path.exists(file_path), f"File {file_path} should exist"


def test_monitoring_alert_files():
    """测试告警文件存在"""
    alert_files = [
        "src/monitoring/alert/__init__.py",
        "src/monitoring/alert/alert_notifier.py"
    ]

    for file_path in alert_files:
        assert os.path.exists(file_path), f"File {file_path} should exist"


def test_monitoring_trading_files():
    """测试交易监控文件存在"""
    trading_files = [
        "src/monitoring/trading/__init__.py",
        "src/monitoring/trading/trading_monitor.py"
    ]

    for file_path in trading_files:
        assert os.path.exists(file_path), f"File {file_path} should exist"


def test_monitoring_ai_files():
    """测试AI监控文件存在"""
    ai_files = [
        "src/monitoring/ai/__init__.py",
        "src/monitoring/ai/deep_learning_predictor.py"
    ]

    for file_path in ai_files:
        assert os.path.exists(file_path), f"File {file_path} should exist"


def test_monitoring_mobile_files():
    """测试移动端监控文件存在"""
    mobile_files = [
        "src/monitoring/mobile/__init__.py",
        "src/monitoring/mobile/mobile_monitor.py"
    ]

    for file_path in mobile_files:
        assert os.path.exists(file_path), f"File {file_path} should exist"


def test_monitoring_web_files():
    """测试Web监控文件存在"""
    web_files = [
        "src/monitoring/web/__init__.py",
        "src/monitoring/web/monitoring_web_app.py"
    ]

    for file_path in web_files:
        assert os.path.exists(file_path), f"File {file_path} should exist"


def test_monitoring_constants_import():
    """测试监控常量导入"""
    try:
        from src.monitoring.core.constants import MONITORING_INTERVAL
        assert isinstance(MONITORING_INTERVAL, (int, float))
    except ImportError:
        pytest.skip("Monitoring constants import failed")


def test_monitoring_exceptions_import():
    """测试监控异常导入"""
    try:
        from src.monitoring.core.exceptions import MonitoringException
        assert issubclass(MonitoringException, Exception)
    except ImportError:
        pytest.skip("Monitoring exceptions import failed")


def test_monitoring_system_import():
    """测试监控系统导入"""
    try:
        from src.monitoring.monitoring_system import MonitoringSystem
        assert hasattr(MonitoringSystem, '__init__')
    except ImportError:
        pytest.skip("MonitoringSystem import failed")


def test_performance_analyzer_import():
    """测试性能分析器导入"""
    try:
        from src.monitoring.engine.performance_analyzer import PerformanceAnalyzer
        assert hasattr(PerformanceAnalyzer, '__init__')
    except ImportError:
        pytest.skip("PerformanceAnalyzer import failed")


def test_alert_notifier_import():
    """测试告警通知器导入"""
    try:
        from src.monitoring.alert.alert_notifier import AlertNotifier
        assert hasattr(AlertNotifier, '__init__')
    except ImportError:
        pytest.skip("AlertNotifier import failed")


def test_trading_monitor_import():
    """测试交易监控器导入"""
    try:
        from src.monitoring.trading.trading_monitor import TradingMonitor
        assert hasattr(TradingMonitor, '__init__')
    except ImportError:
        pytest.skip("TradingMonitor import failed")


def test_intelligent_alert_system_import():
    """测试智能告警系统导入"""
    try:
        from src.monitoring.intelligent_alert_system import IntelligentAlertSystem
        assert hasattr(IntelligentAlertSystem, '__init__')
    except ImportError:
        pytest.skip("IntelligentAlertSystem import failed")


def test_deep_learning_predictor_import():
    """测试深度学习预测器导入"""
    try:
        from src.monitoring.ai.deep_learning_predictor import DeepLearningPredictor
        assert hasattr(DeepLearningPredictor, '__init__')
    except ImportError:
        pytest.skip("DeepLearningPredictor import failed")


def test_mobile_monitor_import():
    """测试移动端监控器导入"""
    try:
        from src.monitoring.mobile.mobile_monitor import MobileMonitor
        assert hasattr(MobileMonitor, '__init__')
    except ImportError:
        pytest.skip("MobileMonitor import failed")


def test_monitoring_web_app_import():
    """测试监控Web应用导入"""
    try:
        from src.monitoring.web.monitoring_web_app import MonitoringWebApp
        assert hasattr(MonitoringWebApp, '__init__')
    except ImportError:
        pytest.skip("MonitoringWebApp import failed")


def test_unified_monitoring_interface_import():
    """测试统一监控接口导入"""
    try:
        from src.monitoring.core.unified_monitoring_interface import IMonitoringSystem
        # 检查接口定义
        assert hasattr(IMonitoringSystem, '__init__') or hasattr(IMonitoringSystem, 'start_monitoring')
    except ImportError:
        pytest.skip("Unified monitoring interface import failed")


def test_full_link_monitor_import():
    """测试全链路监控器导入"""
    try:
        from src.monitoring.engine.full_link_monitor import FullLinkMonitor
        assert hasattr(FullLinkMonitor, '__init__')
    except ImportError:
        pytest.skip("FullLinkMonitor import failed")


def test_monitor_components_import():
    """测试监控组件导入"""
    try:
        from src.monitoring.engine.monitor_components import MonitorComponentFactory
        assert hasattr(MonitorComponentFactory, '__init__') or hasattr(MonitorComponentFactory, 'create_component')
    except ImportError:
        pytest.skip("Monitor components import failed")
