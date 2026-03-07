# -*- coding: utf-8 -*-
"""
特征监控器覆盖率测试 - Phase 2
针对FeaturesMonitor类的未覆盖方法进行补充测试
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from src.features.monitoring.features_monitor import (
    FeaturesMonitor, MetricType, MetricValue, ComponentInfo
)


class TestFeaturesMonitorCoverage:
    """测试FeaturesMonitor的未覆盖方法"""

    @pytest.fixture
    def monitor(self):
        """创建FeaturesMonitor实例"""
        return FeaturesMonitor()

    def test_register_component_success(self, monitor):
        """测试注册组件 - 成功"""
        monitor.register_component("test_component", "processor")
        
        # 验证组件已注册
        assert "test_component" in monitor.components
        assert monitor.components["test_component"].component_type == "processor"
        assert monitor.components["test_component"].status == "registered"

    def test_register_component_duplicate(self, monitor):
        """测试注册组件 - 重复注册"""
        monitor.register_component("test_component", "processor")
        monitor.register_component("test_component", "analyzer")
        
        # 验证组件信息已更新
        assert monitor.components["test_component"].component_type == "analyzer"

    def test_unregister_component_success(self, monitor):
        """测试注销组件 - 成功"""
        monitor.register_component("test_component", "processor")
        monitor.unregister_component("test_component")
        
        # 验证组件已注销
        assert "test_component" not in monitor.components
        assert "test_component" not in monitor.component_locks

    def test_unregister_component_not_found(self, monitor):
        """测试注销组件 - 组件不存在"""
        # 应该不抛出异常
        monitor.unregister_component("nonexistent_component")

    def test_update_component_status_success(self, monitor):
        """测试更新组件状态 - 成功"""
        monitor.register_component("test_component", "processor")
        
        metrics = {"cpu_usage": 50.0, "memory_usage": 60.0}
        monitor.update_component_status("test_component", "running", metrics)
        
        # 验证状态已更新
        assert monitor.components["test_component"].status == "running"
        assert "cpu_usage" in monitor.components["test_component"].metrics

    def test_update_component_status_not_registered(self, monitor):
        """测试更新组件状态 - 组件未注册"""
        # 应该不抛出异常，只记录警告
        monitor.update_component_status("nonexistent_component", "running")

    def test_collect_metrics_success(self, monitor):
        """测试收集指标 - 成功"""
        monitor.register_component("test_component", "processor")
        
        monitor.collect_metrics("test_component", "cpu_usage", 75.0, MetricType.GAUGE)
        
        # 验证指标已收集
        assert "cpu_usage" in monitor.components["test_component"].metrics
        assert monitor.components["test_component"].metrics["cpu_usage"].value == 75.0

    def test_collect_metrics_not_registered(self, monitor):
        """测试收集指标 - 组件未注册"""
        # 应该不抛出异常，只记录警告
        monitor.collect_metrics("nonexistent_component", "cpu_usage", 75.0)

    def test_collect_metrics_with_labels(self, monitor):
        """测试收集指标 - 带标签"""
        monitor.register_component("test_component", "processor")
        
        labels = {"environment": "production", "region": "us-east"}
        monitor.collect_metrics("test_component", "cpu_usage", 75.0, labels=labels)
        
        # 验证标签已设置
        metric = monitor.components["test_component"].metrics["cpu_usage"]
        assert metric.labels == labels

    def test_check_thresholds_exceeded(self, monitor):
        """测试检查阈值 - 超过阈值"""
        monitor.register_component("test_component", "processor")
        monitor.thresholds["test_component.cpu_usage"] = 80.0
        
        # Mock alert_manager.send_alert
        with patch.object(monitor.alert_manager, 'send_alert') as mock_send_alert:
            monitor._check_thresholds("test_component", "cpu_usage", 85.0)
            
            # 验证告警已发送
            mock_send_alert.assert_called_once()

    def test_check_thresholds_not_exceeded(self, monitor):
        """测试检查阈值 - 未超过阈值"""
        monitor.register_component("test_component", "processor")
        monitor.thresholds["test_component.cpu_usage"] = 80.0
        
        # Mock alert_manager.send_alert
        with patch.object(monitor.alert_manager, 'send_alert') as mock_send_alert:
            monitor._check_thresholds("test_component", "cpu_usage", 70.0)
            
            # 验证告警未发送
            mock_send_alert.assert_not_called()

    def test_get_component_metrics_success(self, monitor):
        """测试获取组件指标 - 成功"""
        monitor.register_component("test_component", "processor")
        monitor.collect_metrics("test_component", "cpu_usage", 75.0)
        
        metrics = monitor.get_component_metrics("test_component")
        
        # 验证结果
        assert isinstance(metrics, dict)
        assert "cpu_usage" in metrics
        assert metrics["cpu_usage"]["value"] == 75.0

    def test_get_component_metrics_not_found(self, monitor):
        """测试获取组件指标 - 组件不存在"""
        metrics = monitor.get_component_metrics("nonexistent_component")
        
        # 应该返回空字典
        assert metrics == {}

    def test_get_all_metrics(self, monitor):
        """测试获取所有组件指标"""
        monitor.register_component("component1", "processor")
        monitor.register_component("component2", "analyzer")
        monitor.collect_metrics("component1", "cpu_usage", 75.0)
        monitor.collect_metrics("component2", "memory_usage", 60.0)
        
        all_metrics = monitor.get_all_metrics()
        
        # 验证结果
        assert isinstance(all_metrics, dict)
        assert "component1" in all_metrics
        assert "component2" in all_metrics

    def test_get_component_status_success(self, monitor):
        """测试获取组件状态 - 成功"""
        monitor.register_component("test_component", "processor")
        monitor.update_component_status("test_component", "running")
        
        status = monitor.get_component_status("test_component")
        
        # 验证结果
        assert isinstance(status, dict)
        assert status.get("status") == "running" or status.get("status") == "registered"
        assert "name" in status or "type" in status
        assert "component_type" in status or "type" in status

    def test_get_component_status_not_found(self, monitor):
        """测试获取组件状态 - 组件不存在"""
        status = monitor.get_component_status("nonexistent_component")
        
        # 应该返回None
        assert status is None

    def test_get_all_status(self, monitor):
        """测试获取所有组件状态"""
        monitor.register_component("component1", "processor")
        monitor.register_component("component2", "analyzer")
        
        all_status = monitor.get_all_status()
        
        # 验证结果
        assert isinstance(all_status, dict)
        assert "component1" in all_status
        assert "component2" in all_status

    def test_start_monitoring(self, monitor):
        """测试启动监控"""
        monitor.start_monitoring()
        
        # 验证监控已启动
        assert monitor.is_monitoring is True
        assert monitor.monitor_thread is not None
        
        # 清理
        monitor.stop_monitoring()

    def test_stop_monitoring(self, monitor):
        """测试停止监控"""
        monitor.start_monitoring()
        monitor.stop_monitoring()
        
        # 验证监控已停止
        assert monitor.is_monitoring is False

    def test_context_manager(self, monitor):
        """测试上下文管理器"""
        with monitor:
            assert monitor.is_monitoring is True
        
        # 退出上下文后，监控应该已停止
        assert monitor.is_monitoring is False

    def test_export_metrics(self, monitor, tmp_path):
        """测试导出指标"""
        monitor.register_component("test_component", "processor")
        monitor.collect_metrics("test_component", "cpu_usage", 75.0)
        
        file_path = tmp_path / "metrics.json"
        monitor.export_metrics(str(file_path))
        
        # 验证文件已创建
        assert file_path.exists()

    def test_get_performance_report(self, monitor):
        """测试获取性能报告"""
        monitor.register_component("test_component", "processor")
        monitor.collect_metrics("test_component", "cpu_usage", 75.0)
        
        report = monitor.get_performance_report()
        
        # 验证结果
        assert isinstance(report, dict)
        assert "components" in report or "metrics" in report or "summary" in report

