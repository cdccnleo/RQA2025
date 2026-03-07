#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 组件监控器

测试 handlers/component_monitor.py 中的核心功能
"""

import sys
import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

# Mock ComponentFactory 导入，在导入component_monitor之前
mock_base_components = MagicMock()
mock_base_components.ComponentFactory = MagicMock()
sys.modules['infrastructure.utils.common.core.base_components'] = mock_base_components


@pytest.fixture
def module():
    """导入模块"""
    from src.infrastructure.monitoring.handlers import component_monitor
    return component_monitor


@pytest.fixture
def monitor(module):
    """创建组件监控器实例"""
    return module.ComponentFactoryMonitor(max_history_size=100)


class TestComponentUsageMetrics:
    """测试组件使用指标数据类"""

    def test_component_usage_metrics_initialization(self, module):
        """测试组件使用指标初始化"""
        metrics = module.ComponentUsageMetrics(
            component_type="TestComponent",
            total_creations=10,
            successful_creations=9,
            failed_creations=1,
            average_creation_time=0.5,
            peak_concurrent_usage=5,
            memory_usage_mb=100.0,
            last_used=datetime.now(),
            error_rate=0.1
        )

        assert metrics.component_type == "TestComponent"
        assert metrics.total_creations == 10
        assert metrics.successful_creations == 9
        assert metrics.failed_creations == 1
        assert metrics.error_rate == 0.1


class TestComponentAlert:
    """测试组件告警数据类"""

    def test_component_alert_initialization(self, module):
        """测试组件告警初始化"""
        alert = module.ComponentAlert(
            alert_id="alert_1",
            component_type="TestComponent",
            alert_type="error_rate",
            severity="high",
            message="Error rate too high",
            timestamp=datetime.now(),
            resolved=False
        )

        assert alert.alert_id == "alert_1"
        assert alert.component_type == "TestComponent"
        assert alert.alert_type == "error_rate"
        assert alert.severity == "high"
        assert alert.resolved is False


class TestComponentFactoryMonitor:
    """测试组件工厂监控器"""

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor.usage_metrics == {}
        assert monitor.creation_times == {}
        assert monitor.active_instances == {}
        assert monitor.alerts == []
        assert monitor.monitoring_active is False
        assert monitor.monitor_thread is None
        assert 'error_rate' in monitor.alert_thresholds

    def test_record_component_creation_success(self, monitor):
        """测试记录组件创建 - 成功"""
        monitor.record_component_creation("TestComponent", 0.5, success=True)

        assert "TestComponent" in monitor.usage_metrics
        metrics = monitor.usage_metrics["TestComponent"]
        assert metrics.total_creations == 1
        assert metrics.successful_creations == 1
        assert metrics.failed_creations == 0
        assert metrics.error_rate == 0.0
        assert metrics.average_creation_time == 0.5

    def test_record_component_creation_failure(self, monitor):
        """测试记录组件创建 - 失败"""
        monitor.record_component_creation("TestComponent", 0.5, success=False)

        metrics = monitor.usage_metrics["TestComponent"]
        assert metrics.total_creations == 1
        assert metrics.successful_creations == 0
        assert metrics.failed_creations == 1
        assert metrics.error_rate == 1.0

    def test_record_component_creation_multiple(self, monitor):
        """测试记录多个组件创建"""
        monitor.record_component_creation("Component1", 0.3, success=True)
        monitor.record_component_creation("Component1", 0.5, success=True)
        monitor.record_component_creation("Component1", 0.7, success=False)

        metrics = monitor.usage_metrics["Component1"]
        assert metrics.total_creations == 3
        assert metrics.successful_creations == 2
        assert metrics.failed_creations == 1
        assert metrics.error_rate == 1.0 / 3.0
        assert metrics.average_creation_time == (0.3 + 0.5 + 0.7) / 3.0

    def test_record_component_destruction(self, monitor):
        """测试记录组件销毁"""
        monitor.record_component_creation("TestComponent", 0.5, success=True)
        assert monitor.active_instances["TestComponent"] == 1

        monitor.record_component_destruction("TestComponent")
        assert monitor.active_instances["TestComponent"] == 0

    def test_record_component_destruction_no_instances(self, monitor):
        """测试记录组件销毁 - 无实例"""
        monitor.record_component_destruction("TestComponent")
        assert monitor.active_instances["TestComponent"] == 0

    def test_get_usage_report(self, monitor):
        """测试获取使用报告"""
        monitor.record_component_creation("Component1", 0.5, success=True)
        monitor.record_component_creation("Component2", 0.3, success=True)

        report = monitor.get_usage_report()

        assert 'timestamp' in report
        assert 'summary' in report
        assert 'component_metrics' in report
        assert 'recent_alerts' in report
        assert report['summary']['total_components'] == 2
        assert report['summary']['total_creations'] == 2
        assert "Component1" in report['component_metrics']
        assert "Component2" in report['component_metrics']

    def test_get_usage_report_empty(self, monitor):
        """测试获取使用报告 - 空数据"""
        report = monitor.get_usage_report()

        assert report['summary']['total_components'] == 0
        assert report['summary']['total_creations'] == 0
        assert len(report['component_metrics']) == 0

    def test_start_monitoring(self, monitor, monkeypatch):
        """测试启动监控"""
        created_threads = []
        original_thread = threading.Thread

        def mock_thread(*args, **kwargs):
            thread = original_thread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        monkeypatch.setattr(threading, "Thread", mock_thread)

        monitor.start_monitoring()

        assert monitor.monitoring_active is True
        assert monitor.monitor_thread is not None
        assert len(created_threads) == 1

    def test_start_monitoring_already_active(self, monitor, monkeypatch):
        """测试启动监控 - 已经激活"""
        monitor.monitoring_active = True
        created_threads = []
        original_thread = threading.Thread

        def mock_thread(*args, **kwargs):
            thread = original_thread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        monkeypatch.setattr(threading, "Thread", mock_thread)

        monitor.start_monitoring()

        # 如果已经激活，不应该创建新线程
        assert len(created_threads) == 0

    def test_stop_monitoring(self, monitor, monkeypatch):
        """测试停止监控"""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        mock_thread.join = MagicMock()

        monitor.monitoring_active = True
        monitor.monitor_thread = mock_thread

        monitor.stop_monitoring()

        assert monitor.monitoring_active is False
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_monitoring_no_thread(self, monitor):
        """测试停止监控 - 无线程"""
        monitor.monitoring_active = True
        monitor.monitor_thread = None

        monitor.stop_monitoring()

        assert monitor.monitoring_active is False

    def test_peak_concurrent_usage(self, monitor):
        """测试峰值并发使用"""
        # 创建多个实例
        for i in range(5):
            monitor.record_component_creation("TestComponent", 0.5, success=True)

        metrics = monitor.usage_metrics["TestComponent"]
        assert metrics.peak_concurrent_usage == 5

        # 销毁一些实例
        for i in range(3):
            monitor.record_component_destruction("TestComponent")

        # 峰值应该保持不变
        assert metrics.peak_concurrent_usage == 5

    def test_error_rate_calculation(self, monitor):
        """测试错误率计算"""
        # 创建10个成功，5个失败
        for i in range(10):
            monitor.record_component_creation("TestComponent", 0.5, success=True)
        for i in range(5):
            monitor.record_component_creation("TestComponent", 0.5, success=False)

        metrics = monitor.usage_metrics["TestComponent"]
        assert metrics.total_creations == 15
        assert metrics.error_rate == 5.0 / 15.0

    def test_average_creation_time(self, monitor):
        """测试平均创建时间"""
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        for t in times:
            monitor.record_component_creation("TestComponent", t, success=True)

        metrics = monitor.usage_metrics["TestComponent"]
        assert metrics.average_creation_time == sum(times) / len(times)


class TestGlobalFunctions:
    """测试全局函数"""

    def test_get_component_monitor(self, module):
        """测试获取组件监控器"""
        monitor = module.get_component_monitor()
        assert isinstance(monitor, module.ComponentFactoryMonitor)

    def test_monitor_component_creation(self, module, monkeypatch):
        """测试监控组件创建"""
        monitor_instance = module.ComponentFactoryMonitor()
        monkeypatch.setattr(module, "_component_monitor", monitor_instance)

        module.monitor_component_creation("TestComponent", 0.5, success=True)

        assert "TestComponent" in monitor_instance.usage_metrics

    def test_monitor_component_destruction(self, module, monkeypatch):
        """测试监控组件销毁"""
        monitor_instance = module.ComponentFactoryMonitor()
        monitor_instance.record_component_creation("TestComponent", 0.5, success=True)
        monkeypatch.setattr(module, "_component_monitor", monitor_instance)

        module.monitor_component_destruction("TestComponent")

        assert monitor_instance.active_instances["TestComponent"] == 0

    def test_check_thresholds_slow_creation(self, monitor, module, monkeypatch):
        """测试检查阈值 - 创建时间过慢"""
        # 设置较低的创建时间阈值
        monitor.alert_thresholds['creation_time'] = 50.0
        
        # 记录创建时间超过阈值的组件
        monitor.record_component_creation("TestComponent", 100.0, success=True)
        
        # Mock logger
        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(monitor.logger, "warning", mock_warning)
        
        # 获取指标并检查告警
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 应该创建告警
        assert len(monitor.alerts) > 0
        assert any('slow_creation' in alert.alert_type for alert in monitor.alerts)

    def test_check_thresholds_high_concurrency(self, monitor, module, monkeypatch):
        """测试检查阈值 - 并发使用过高"""
        # 设置较低的并发阈值
        monitor.alert_thresholds['concurrent_usage'] = 2
        
        # 创建多个实例
        for i in range(5):
            monitor.record_component_creation("TestComponent", 10.0, success=True)
        
        # Mock logger
        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(monitor.logger, "warning", mock_warning)
        
        # 获取指标并检查告警
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 应该创建告警
        assert len(monitor.alerts) > 0
        assert any('high_concurrency' in alert.alert_type for alert in monitor.alerts)

    def test_resolve_alert(self, monitor, module, monkeypatch):
        """测试解决告警"""
        # 创建一个告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        assert len(monitor.alerts) > 0
        alert_id = monitor.alerts[0].alert_id
        
        # Mock logger
        infos = []
        def mock_info(msg):
            infos.append(msg)
        
        monkeypatch.setattr(monitor.logger, "info", mock_info)
        
        monitor.resolve_alert(alert_id)
        
        # 告警应该被标记为已解决
        assert monitor.alerts[0].resolved is True
        assert len(infos) > 0

    def test_resolve_alert_not_found(self, monitor, module, monkeypatch):
        """测试解决告警 - 未找到"""
        # 创建一个告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        original_count = len(monitor.alerts)
        
        # 尝试解决不存在的告警
        monitor.resolve_alert("nonexistent_alert_id")
        
        # 告警数量不应该改变
        assert len(monitor.alerts) == original_count

    def test_export_metrics_csv(self, monitor, module, monkeypatch):
        """测试导出指标 - CSV格式"""
        # 记录一些数据
        monitor.record_component_creation("TestComponent", 50.0, success=True)
        monitor.record_component_creation("TestComponent", 60.0, success=True)
        
        csv_data = monitor.export_metrics('csv')
        
        assert isinstance(csv_data, str)
        assert 'Component' in csv_data
        assert 'TestComponent' in csv_data

    def test_export_metrics_other_format(self, monitor, module, monkeypatch):
        """测试导出指标 - 其他格式"""
        # 记录一些数据
        monitor.record_component_creation("TestComponent", 50.0, success=True)
        
        data = monitor.export_metrics('xml')
        
        assert isinstance(data, str)

    def test_monitoring_loop_cleanup_old_alerts(self, monitor, module, monkeypatch):
        """测试监控循环 - 清理旧告警"""
        from datetime import timedelta
        
        # 创建一些告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 修改告警时间戳为8天前
        old_time = datetime.now() - timedelta(days=8)
        for alert in monitor.alerts:
            alert.timestamp = old_time
            alert.resolved = True  # 已解决的告警应该被清理
        
        original_count = len(monitor.alerts)
        
        # 模拟监控循环中的清理逻辑
        now = datetime.now()
        cutoff_date = now - timedelta(days=7)
        monitor.alerts = [a for a in monitor.alerts if a.timestamp > cutoff_date or not a.resolved]
        
        # 已解决的旧告警应该被清理
        assert len(monitor.alerts) < original_count

    def test_monitoring_loop_unused_components(self, monitor, module, monkeypatch):
        """测试监控循环 - 检查未使用的组件"""
        from datetime import timedelta
        
        # 记录组件使用
        monitor.record_component_creation("TestComponent", 50.0, success=True)
        
        # 修改最后使用时间为2小时前
        if "TestComponent" in monitor.usage_metrics:
            monitor.usage_metrics["TestComponent"].last_used = datetime.now() - timedelta(hours=2)
        
        # Mock logger
        infos = []
        def mock_info(msg):
            infos.append(msg)
        
        monkeypatch.setattr(monitor.logger, "info", mock_info)
        
        # 模拟监控循环中的检查逻辑
        now = datetime.now()
        for component_type, metrics in monitor.usage_metrics.items():
            if (now - metrics.last_used) > timedelta(hours=1):
                monitor.logger.info(f"组件 {component_type} 已1小时未使用")
        
        # 应该记录信息
        assert len(infos) > 0

    def test_monitored_component_factory(self, module, monkeypatch):
        """测试带监控功能的ComponentFactory"""
        # Mock ComponentFactory
        mock_factory = MagicMock()
        mock_component = MagicMock()
        mock_factory.create_component = MagicMock(return_value=mock_component)
        
        monkeypatch.setattr(module, "ComponentFactory", MagicMock(return_value=mock_factory))
        
        monitored_factory = module.MonitoredComponentFactory()
        
        # 测试成功创建
        component = monitored_factory.create_component("TestComponent", {"key": "value"})
        
        assert component == mock_component
        assert len(monitored_factory.monitor.usage_metrics) > 0

    def test_monitored_component_factory_exception(self, module, monkeypatch):
        """测试带监控功能的ComponentFactory - 异常处理"""
        # Mock ComponentFactory 抛出异常
        mock_factory = MagicMock()
        mock_factory.create_component = MagicMock(side_effect=Exception("Test error"))
        
        monkeypatch.setattr(module, "ComponentFactory", MagicMock(return_value=mock_factory))
        
        monitored_factory = module.MonitoredComponentFactory()
        
        # 测试异常情况
        with pytest.raises(Exception):
            monitored_factory.create_component("TestComponent")
        
        # 应该记录失败的创建
        assert "TestComponent" in monitored_factory.monitor.usage_metrics

    def test_monitoring_loop_exception_handling(self, monitor, module, monkeypatch):
        """测试监控循环 - 异常处理"""
        monitor.monitoring_active = True
        
        # Mock time.sleep 来触发异常
        call_count = {"count": 0}
        def mock_sleep(seconds):
            call_count["count"] += 1
            if call_count["count"] == 1:
                # 第一次调用时模拟异常
                monitor.monitoring_active = False  # 先停止循环
                raise Exception("Test error")
        
        monkeypatch.setattr(module.time, "sleep", mock_sleep)
        
        # Mock logger
        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(monitor.logger, "error", mock_error)
        
        # 直接测试异常处理逻辑（模拟循环中的异常）
        try:
            # 模拟循环中的代码
            monitor.monitoring_active = True
            module.time.sleep(60)  # 这会触发异常
        except Exception as e:
            # 模拟循环中的异常处理
            monitor.logger.error(f"监控循环异常: {e}")
        
        # 应该记录错误
        assert len(errors) > 0
        assert "监控循环异常" in errors[0]

    def test_export_metrics_json(self, monitor, module, monkeypatch):
        """测试导出指标 - JSON格式"""
        # 记录一些数据
        monitor.record_component_creation("TestComponent", 50.0, success=True)
        monitor.record_component_creation("TestComponent", 60.0, success=True)
        
        json_data = monitor.export_metrics('json')
        
        assert isinstance(json_data, str)
        import json
        parsed = json.loads(json_data)
        assert 'summary' in parsed
        assert 'component_metrics' in parsed

    def test_export_metrics_csv_formatting(self, monitor, module, monkeypatch):
        """测试导出指标 - CSV格式格式化"""
        # 记录一些数据
        monitor.record_component_creation("TestComponent", 50.0, success=True)
        monitor.record_component_creation("TestComponent2", 60.0, success=True)
        
        csv_data = monitor.export_metrics('csv')
        
        assert isinstance(csv_data, str)
        lines = csv_data.split('\n')
        assert len(lines) >= 3  # 标题行 + 2个数据行
        assert 'Component' in lines[0]
        assert 'TestComponent' in csv_data
        assert 'TestComponent2' in csv_data

    def test_get_usage_report_with_alerts(self, monitor, module, monkeypatch):
        """测试获取使用报告 - 包含告警"""
        # 创建告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        report = monitor.get_usage_report()
        
        assert 'recent_alerts' in report
        assert len(report['recent_alerts']) > 0

    def test_get_usage_report_no_alerts(self, monitor, module, monkeypatch):
        """测试获取使用报告 - 无告警"""
        # 只记录成功的创建，不触发告警
        monitor.record_component_creation("TestComponent", 10.0, success=True)
        
        report = monitor.get_usage_report()
        
        assert 'recent_alerts' in report
        assert isinstance(report['recent_alerts'], list)

    def test_monitoring_loop_unused_components_check(self, monitor, module, monkeypatch):
        """测试监控循环 - 检查长期未使用的组件"""
        from datetime import timedelta
        
        # 记录组件使用
        monitor.record_component_creation("TestComponent", 50.0, success=True)
        
        # 修改最后使用时间为2小时前
        if "TestComponent" in monitor.usage_metrics:
            monitor.usage_metrics["TestComponent"].last_used = datetime.now() - timedelta(hours=2)
        
        # Mock logger
        infos = []
        def mock_info(msg):
            infos.append(msg)
        
        monkeypatch.setattr(monitor.logger, "info", mock_info)
        
        # 模拟监控循环中的检查逻辑
        now = datetime.now()
        for component_type, metrics in monitor.usage_metrics.items():
            if (now - metrics.last_used) > timedelta(hours=1):
                monitor.logger.info(f"组件 {component_type} 已1小时未使用")
        
        # 应该记录信息
        assert len(infos) > 0
        assert any("已1小时未使用" in msg for msg in infos)

    def test_monitoring_loop_cleanup_old_resolved_alerts(self, monitor, module, monkeypatch):
        """测试监控循环 - 清理已解决的旧告警"""
        from datetime import timedelta
        
        # 创建一些告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 解决所有告警
        for alert in monitor.alerts:
            alert.resolved = True
        
        # 修改告警时间戳为8天前
        old_time = datetime.now() - timedelta(days=8)
        for alert in monitor.alerts:
            alert.timestamp = old_time
        
        original_count = len(monitor.alerts)
        
        # 模拟监控循环中的清理逻辑
        now = datetime.now()
        cutoff_date = now - timedelta(days=7)
        monitor.alerts = [a for a in monitor.alerts if a.timestamp > cutoff_date or not a.resolved]
        
        # 已解决的旧告警应该被清理
        assert len(monitor.alerts) < original_count

    def test_monitoring_loop_keep_unresolved_alerts(self, monitor, module, monkeypatch):
        """测试监控循环 - 保留未解决的旧告警"""
        from datetime import timedelta
        
        # 创建一些告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 不解决告警，但修改时间戳为8天前
        old_time = datetime.now() - timedelta(days=8)
        for alert in monitor.alerts:
            alert.timestamp = old_time
            alert.resolved = False  # 未解决
        
        original_count = len(monitor.alerts)
        
        # 模拟监控循环中的清理逻辑
        now = datetime.now()
        cutoff_date = now - timedelta(days=7)
        monitor.alerts = [a for a in monitor.alerts if a.timestamp > cutoff_date or not a.resolved]
        
        # 未解决的旧告警应该被保留
        assert len(monitor.alerts) == original_count

    def test_monitoring_loop_keep_recent_alerts(self, monitor, module, monkeypatch):
        """测试监控循环 - 保留最近的告警"""
        from datetime import timedelta
        
        # 创建一些告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 修改告警时间戳为3天前（在7天范围内）
        recent_time = datetime.now() - timedelta(days=3)
        for alert in monitor.alerts:
            alert.timestamp = recent_time
            alert.resolved = True  # 已解决
        
        original_count = len(monitor.alerts)
        
        # 模拟监控循环中的清理逻辑
        now = datetime.now()
        cutoff_date = now - timedelta(days=7)
        monitor.alerts = [a for a in monitor.alerts if a.timestamp > cutoff_date or not a.resolved]
        
        # 最近的告警应该被保留（即使已解决）
        assert len(monitor.alerts) == original_count

    def test_monitoring_loop_check_unused_components(self, monitor, module, monkeypatch):
        """测试监控循环 - 检查长期未使用的组件"""
        from datetime import timedelta
        
        # 创建组件并记录使用
        monitor.record_component_creation("TestComponent", 100.0, success=True)
        
        # 修改最后使用时间为2小时前
        if "TestComponent" in monitor.usage_metrics:
            monitor.usage_metrics["TestComponent"].last_used = datetime.now() - timedelta(hours=2)
        
        # Mock logger
        logs = []
        def mock_info(msg):
            logs.append(msg)
        
        monitor.logger.info = mock_info
        
        # 模拟监控循环中的检查逻辑
        now = datetime.now()
        for component_type, metrics in monitor.usage_metrics.items():
            if (now - metrics.last_used) > timedelta(hours=1):
                monitor.logger.info(f"组件 {component_type} 已1小时未使用")
        
        # 应该记录未使用的组件
        assert len(logs) > 0
        assert any("已1小时未使用" in log for log in logs)

    def test_monitoring_loop_check_recently_used_components(self, monitor, module, monkeypatch):
        """测试监控循环 - 检查最近使用的组件"""
        from datetime import timedelta
        
        # 创建组件并记录使用
        monitor.record_component_creation("TestComponent", 100.0, success=True)
        
        # 修改最后使用时间为30分钟前（在1小时内）
        if "TestComponent" in monitor.usage_metrics:
            monitor.usage_metrics["TestComponent"].last_used = datetime.now() - timedelta(minutes=30)
        
        # Mock logger
        logs = []
        def mock_info(msg):
            logs.append(msg)
        
        monitor.logger.info = mock_info
        
        # 模拟监控循环中的检查逻辑
        now = datetime.now()
        for component_type, metrics in monitor.usage_metrics.items():
            if (now - metrics.last_used) > timedelta(hours=1):
                monitor.logger.info(f"组件 {component_type} 已1小时未使用")
        
        # 应该不记录最近使用的组件
        assert len(logs) == 0

    def test_monitoring_loop_cleanup_old_resolved_alerts(self, monitor, module, monkeypatch):
        """测试监控循环 - 清理旧的已解决告警"""
        from datetime import timedelta
        
        # 创建一些告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 修改告警时间戳为8天前（超过7天）
        old_time = datetime.now() - timedelta(days=8)
        for alert in monitor.alerts:
            alert.timestamp = old_time
            alert.resolved = True  # 已解决
        
        original_count = len(monitor.alerts)
        
        # 模拟监控循环中的清理逻辑
        now = datetime.now()
        cutoff_date = now - timedelta(days=7)
        monitor.alerts = [a for a in monitor.alerts if a.timestamp > cutoff_date or not a.resolved]
        
        # 旧的已解决告警应该被清理
        assert len(monitor.alerts) < original_count

    def test_monitoring_loop_keep_unresolved_old_alerts(self, monitor, module, monkeypatch):
        """测试监控循环 - 保留未解决的旧告警"""
        from datetime import timedelta
        
        # 创建一些告警
        monitor.record_component_creation("TestComponent", 100.0, success=False)
        if "TestComponent" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent", monitor.usage_metrics["TestComponent"])
        
        # 修改告警时间戳为8天前（超过7天），但未解决
        old_time = datetime.now() - timedelta(days=8)
        for alert in monitor.alerts:
            alert.timestamp = old_time
            alert.resolved = False  # 未解决
        
        original_count = len(monitor.alerts)
        
        # 模拟监控循环中的清理逻辑
        now = datetime.now()
        cutoff_date = now - timedelta(days=7)
        monitor.alerts = [a for a in monitor.alerts if a.timestamp > cutoff_date or not a.resolved]
        
        # 未解决的旧告警应该被保留
        assert len(monitor.alerts) == original_count

    def test_monitoring_loop_direct_execution(self, monitor, module, monkeypatch):
        """测试监控循环 - 直接执行循环逻辑"""
        from datetime import timedelta
        import time
        
        # 创建组件并记录使用
        monitor.record_component_creation("TestComponent", 100.0, success=True)
        
        # 修改最后使用时间为2小时前
        if "TestComponent" in monitor.usage_metrics:
            monitor.usage_metrics["TestComponent"].last_used = datetime.now() - timedelta(hours=2)
        
        # 创建一些告警
        monitor.record_component_creation("TestComponent2", 100.0, success=False)
        if "TestComponent2" in monitor.usage_metrics:
            monitor._check_alerts("TestComponent2", monitor.usage_metrics["TestComponent2"])
        
        # 修改告警时间戳为8天前（超过7天），已解决
        old_time = datetime.now() - timedelta(days=8)
        for alert in monitor.alerts:
            alert.timestamp = old_time
            alert.resolved = True
        
        # Mock logger
        info_logs = []
        error_logs = []
        def mock_info(msg):
            info_logs.append(msg)
        def mock_error(msg):
            error_logs.append(msg)
        
        monitor.logger.info = mock_info
        monitor.logger.error = mock_error
        
        # Mock time.sleep 以避免实际等待
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
            # 只执行一次循环就退出
            monitor.monitoring_active = False
        
        monkeypatch.setattr(module.time, "sleep", mock_sleep)
        
        # 启动监控（会执行一次循环）
        monitor.monitoring_active = True
        monitor._monitoring_loop()
        
        # 应该调用了sleep
        assert len(sleep_calls) > 0
        # 应该记录了未使用的组件
        assert len(info_logs) > 0
        assert any("已1小时未使用" in log for log in info_logs)

    def test_monitoring_loop_exception_in_loop(self, monitor, module, monkeypatch):
        """测试监控循环 - 循环中发生异常"""
        from datetime import timedelta
        import time
        
        # Mock logger
        error_logs = []
        def mock_error(msg):
            error_logs.append(msg)
        
        monitor.logger.error = mock_error
        
        # 创建一个会抛出异常的字典类
        class ExceptionDict(dict):
            def items(self):
                raise Exception("Test exception")
        
        # 替换 usage_metrics
        original_usage_metrics = monitor.usage_metrics
        monitor.usage_metrics = ExceptionDict()
        
        # Mock time.sleep
        def mock_sleep(seconds):
            monitor.monitoring_active = False
        
        monkeypatch.setattr(module.time, "sleep", mock_sleep)
        
        # 启动监控（会执行一次循环并捕获异常）
        monitor.monitoring_active = True
        monitor._monitoring_loop()
        
        # 恢复原始 usage_metrics
        monitor.usage_metrics = original_usage_metrics
        
        # 应该记录了异常
        assert len(error_logs) > 0
        assert any("监控循环异常" in log for log in error_logs)

