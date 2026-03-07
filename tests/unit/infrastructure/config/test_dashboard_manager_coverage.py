"""
监控面板管理器测试用例 - 提升覆盖率到80%+
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.infrastructure.config.monitoring.dashboard_manager import UnifiedMonitoringManager
from src.infrastructure.config.monitoring.dashboard_models import (
    MonitoringConfig, PerformanceMetrics, SystemResources, AlertSeverity
)


class TestUnifiedMonitoringManagerCoverage:
    """UnifiedMonitoringManager覆盖率测试"""

    def setup_method(self):
        """测试前准备"""
        self.config = MonitoringConfig(
            enabled=True,
            collection_interval=15,
            retention_days=30,
            alerting_enabled=True
        )
        self.manager = UnifiedMonitoringManager(self.config)

    def test_unified_monitoring_manager_init(self):
        """测试UnifiedMonitoringManager初始化"""
        assert self.manager.config == self.config
        assert self.manager._metrics_collector is None
        assert self.manager._alert_manager is None
        assert isinstance(self.manager._performance_history, list)
        assert isinstance(self.manager._system_resources_history, list)
        assert self.manager._max_history_size == 10000
        assert 'start_time' in self.manager._stats
        assert 'total_operations' in self.manager._stats

    def test_unified_monitoring_manager_init_default_config(self):
        """测试使用默认配置初始化"""
        manager = UnifiedMonitoringManager()
        assert isinstance(manager.config, MonitoringConfig)

    def test_set_metrics_collector(self):
        """测试设置指标收集器"""
        mock_collector = Mock()
        
        self.manager.set_metrics_collector(mock_collector)
        assert self.manager._metrics_collector == mock_collector

    def test_set_alert_manager(self):
        """测试设置告警管理器"""
        mock_manager = StandardMockBuilder.create_config_mock()
        
        self.manager.set_alert_manager(mock_manager)
        assert self.manager._alert_manager == mock_manager

    def test_start_monitoring_enabled(self):
        """测试启动监控（启用状态）"""
        mock_collector = Mock()
        mock_alert_manager = Mock()
        
        self.manager.set_metrics_collector(mock_collector)
        self.manager.set_alert_manager(mock_alert_manager)
        
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            self.manager.start_monitoring()
            
            mock_collector.start_collection.assert_called_once()
            mock_logger.info.assert_any_call("指标收集已启动")
            mock_logger.info.assert_any_call("告警管理已启用")
            mock_logger.info.assert_any_call("统一监控管理器已启动")

    def test_start_monitoring_disabled(self):
        """测试启动监控（禁用状态）"""
        self.config.enabled = False
        
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            self.manager.start_monitoring()
            mock_logger.info.assert_called_with("监控已禁用")

    def test_start_monitoring_alerting_disabled(self):
        """测试启动监控（告警禁用状态）"""
        self.config.alerting_enabled = False
        mock_collector = Mock()
        mock_alert_manager = Mock()
        
        self.manager.set_metrics_collector(mock_collector)
        self.manager.set_alert_manager(mock_alert_manager)
        
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            self.manager.start_monitoring()
            mock_collector.start_collection.assert_called_once()

    def test_stop_monitoring(self):
        """测试停止监控"""
        mock_collector = Mock()
        self.manager.set_metrics_collector(mock_collector)
        
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            self.manager.stop_monitoring()
            
            mock_collector.stop_collection.assert_called_once()
            mock_logger.info.assert_any_call("指标收集已停止")
            mock_logger.info.assert_any_call("统一监控管理器已停止")

    def test_stop_monitoring_no_collector(self):
        """测试停止监控（无收集器）"""
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            self.manager.stop_monitoring()
            mock_logger.info.assert_called_with("统一监控管理器已停止")

    def test_record_operation_success(self):
        """测试记录成功操作"""
        mock_collector = Mock()
        mock_alert_manager = Mock()
        self.manager.set_metrics_collector(mock_collector)
        self.manager.set_alert_manager(mock_alert_manager)
        
        initial_ops = self.manager._stats['total_operations']
        initial_success = self.manager._stats['successful_operations']
        
        self.manager.record_operation(
            operation_type="test_op",
            duration=1.5,
            success=True,
            metadata={"test": "data"}
        )
        
        assert self.manager._stats['total_operations'] == initial_ops + 1
        assert self.manager._stats['successful_operations'] == initial_success + 1
        assert len(self.manager._performance_history) == 1
        
        metric = self.manager._performance_history[0]
        assert metric.operation_type == "test_op"
        assert metric.duration == 1.5
        assert metric.success is True
        assert metric.metadata == {"test": "data"}
        
        mock_collector.record_operation.assert_called_once_with(
            "test_op", 1.5, True, {"test": "data"}
        )

    def test_record_operation_failure(self):
        """测试记录失败操作"""
        mock_collector = Mock()
        mock_alert_manager = Mock()
        self.manager.set_metrics_collector(mock_collector)
        self.manager.set_alert_manager(mock_alert_manager)
        
        initial_failed = self.manager._stats['failed_operations']
        
        self.manager.record_operation(
            operation_type="failed_op",
            duration=2.0,
            success=False,
            error_message="Test error"
        )
        
        assert self.manager._stats['failed_operations'] == initial_failed + 1
        
        metric = self.manager._performance_history[0]
        assert metric.success is False
        assert metric.error_message == "Test error"

    def test_record_operation_history_limit(self):
        """测试操作历史记录限制"""
        self.manager._max_history_size = 2
        
        # 记录3个操作，应该只保留最新的2个
        for i in range(3):
            self.manager.record_operation(
                operation_type=f"op_{i}",
                duration=1.0,
                success=True
            )
        
        assert len(self.manager._performance_history) == 2
        # 第一个操作应该被移除
        assert self.manager._performance_history[0].operation_type == "op_1"
        assert self.manager._performance_history[1].operation_type == "op_2"

    def test_record_operation_no_collector(self):
        """测试记录操作（无收集器）"""
        self.manager.record_operation(
            operation_type="test_op",
            duration=1.0,
            success=True
        )
        
        assert len(self.manager._performance_history) == 1

    def test_record_system_resources(self):
        """测试记录系统资源"""
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        disk_usage = {"/": 50.0, "/tmp": 30.0}
        network_io = {"bytes_sent": 1000, "bytes_recv": 2000}
        load_average = (1.0, 2.0, 3.0)
        
        self.manager.record_system_resources(
            cpu_percent=75.0,
            memory_percent=60.0,
            disk_usage=disk_usage,
            network_io=network_io,
            load_average=load_average
        )
        
        assert len(self.manager._system_resources_history) == 1
        
        resources = self.manager._system_resources_history[0]
        assert resources.cpu_percent == 75.0
        assert resources.memory_percent == 60.0
        assert resources.disk_usage == disk_usage
        assert resources.network_io == network_io
        assert resources.load_average == load_average

    def test_record_system_resources_history_limit(self):
        """测试系统资源历史记录限制"""
        self.manager._max_history_size = 2
        
        for i in range(3):
            self.manager.record_system_resources(
                cpu_percent=float(i * 10),
                memory_percent=50.0,
                disk_usage={"/": 30.0},
                network_io={"bytes_sent": 1000}
            )
        
        assert len(self.manager._system_resources_history) == 2
        assert self.manager._system_resources_history[0].cpu_percent == 10.0
        assert self.manager._system_resources_history[1].cpu_percent == 20.0

    def test_check_alerts_no_manager(self):
        """测试检查告警（无告警管理器）"""
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            operation_type="test",
            duration=1.0,
            success=False
        )
        
        # 不应该抛出异常
        self.manager._check_alerts(metric)

    def test_check_alerts_alerting_disabled(self):
        """测试检查告警（告警禁用）"""
        self.config.alerting_enabled = False
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            operation_type="test",
            duration=1.0,
            success=False
        )
        
        self.manager._check_alerts(metric)
        mock_alert_manager.create_alert.assert_not_called()

    def test_check_alerts_operation_failure(self):
        """测试检查操作失败告警"""
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            operation_type="test_op",
            duration=2.0,
            success=False,
            error_message="Test error"
        )
        
        self.manager._check_alerts(metric)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "Operation Failed: test_op" in call_args[1]["name"]
        assert call_args[1]["severity"] == AlertSeverity.ERROR

    def test_check_alerts_operation_timeout(self):
        """测试检查操作超时告警"""
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            operation_type="slow_op",
            duration=35.0,  # 超过30秒阈值
            success=True
        )
        
        self.manager._check_alerts(metric)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "Operation Timeout: slow_op" in call_args[1]["name"]
        assert call_args[1]["severity"] == AlertSeverity.WARNING
        assert call_args[1]["value"] == 35.0

    def test_check_resource_alerts_no_manager(self):
        """测试检查资源告警（无告警管理器）"""
        resources = SystemResources(
            timestamp=datetime.now(),
            cpu_percent=95.0,
            memory_percent=90.0,
            disk_usage={"/": 98.0},
            network_io={}
        )
        
        # 不应该抛出异常
        self.manager._check_resource_alerts(resources)

    def test_check_resource_alerts_alerting_disabled(self):
        """测试检查资源告警（告警禁用）"""
        self.config.alerting_enabled = False
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        resources = SystemResources(
            timestamp=datetime.now(),
            cpu_percent=95.0,
            memory_percent=90.0,
            disk_usage={"/": 50.0},
            network_io={}
        )
        
        self.manager._check_resource_alerts(resources)
        mock_alert_manager.create_alert.assert_not_called()

    def test_check_resource_alerts_cpu_high(self):
        """测试检查CPU使用率高告警"""
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        resources = SystemResources(
            timestamp=datetime.now(),
            cpu_percent=95.0,  # 超过90%阈值
            memory_percent=50.0,
            disk_usage={"/": 50.0},
            network_io={}
        )
        
        self.manager._check_resource_alerts(resources)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "High CPU Usage" in call_args[1]["name"]
        assert call_args[1]["value"] == 95.0

    def test_check_resource_alerts_memory_high(self):
        """测试检查内存使用率高告警"""
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        resources = SystemResources(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=90.0,  # 超过85%阈值
            disk_usage={"/": 50.0},
            network_io={}
        )
        
        self.manager._check_resource_alerts(resources)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "High Memory Usage" in call_args[1]["name"]
        assert call_args[1]["value"] == 90.0

    def test_check_resource_alerts_disk_high(self):
        """测试检查磁盘使用率高告警"""
        mock_alert_manager = Mock()
        self.manager.set_alert_manager(mock_alert_manager)
        
        resources = SystemResources(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=50.0,
            disk_usage={"/": 98.0, "/tmp": 50.0},  # / 超过95%阈值
            network_io={}
        )
        
        self.manager._check_resource_alerts(resources)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "High Disk Usage: /" in call_args[1]["name"]
        assert call_args[1]["severity"] == AlertSeverity.ERROR

    def test_get_performance_metrics_all(self):
        """测试获取所有性能指标"""
        # 添加一些测试数据
        for i in range(3):
            self.manager.record_operation(f"op_{i}", 1.0, True)
        
        metrics = self.manager.get_performance_metrics()
        assert len(metrics) == 3

    def test_get_performance_metrics_filtered_by_type(self):
        """测试按操作类型过滤性能指标"""
        self.manager.record_operation("type_a", 1.0, True)
        self.manager.record_operation("type_b", 2.0, True)
        self.manager.record_operation("type_a", 1.5, False)
        
        metrics = self.manager.get_performance_metrics(operation_type="type_a")
        assert len(metrics) == 2
        assert all(m.operation_type == "type_a" for m in metrics)

    def test_get_performance_metrics_filtered_by_time(self):
        """测试按时间范围过滤性能指标"""
        # 添加一些测试数据
        self.manager.record_operation("test_op", 1.0, True)
        
        # 获取最近1小时的数据（应该包含刚添加的）
        time_range = timedelta(hours=1)
        metrics = self.manager.get_performance_metrics(time_range=time_range)
        assert len(metrics) == 1
        
        # 获取过去很短时间的数据（在实际测试中可能由于执行时间而包含数据）
        # 这里我们测试功能是工作的，只要时间过滤逻辑被执行了
        time_range = timedelta(milliseconds=1)
        metrics = self.manager.get_performance_metrics(time_range=time_range)
        # 由于测试执行时间可能超过1毫秒，我们只验证方法不抛异常
        assert isinstance(metrics, list)

    def test_get_system_resources_all(self):
        """测试获取所有系统资源"""
        # 添加一些测试数据
        for i in range(3):
            self.manager.record_system_resources(
                cpu_percent=float(i * 10),
                memory_percent=50.0,
                disk_usage={"/": 30.0},
                network_io={}
            )
        
        resources = self.manager.get_system_resources()
        assert len(resources) == 3

    def test_get_system_resources_filtered_by_time(self):
        """测试按时间范围过滤系统资源"""
        self.manager.record_system_resources(
            cpu_percent=50.0,
            memory_percent=50.0,
            disk_usage={"/": 30.0},
            network_io={}
        )
        
        # 获取最近1小时的数据
        time_range = timedelta(hours=1)
        resources = self.manager.get_system_resources(time_range=time_range)
        assert len(resources) == 1

    def test_get_statistics_basic(self):
        """测试获取基本统计信息"""
        # 记录一些操作
        self.manager.record_operation("success_op", 1.0, True)
        self.manager.record_operation("failed_op", 2.0, False)
        
        stats = self.manager.get_statistics()
        
        assert 'uptime' in stats
        assert 'start_time' in stats
        assert stats['total_operations'] == 2
        assert stats['successful_operations'] == 1
        assert stats['failed_operations'] == 1
        assert stats['success_rate'] == 0.5

    def test_get_statistics_no_operations(self):
        """测试获取统计信息（无操作）"""
        stats = self.manager.get_statistics()
        assert stats['total_operations'] == 0
        assert stats['success_rate'] == 0.0

    def test_get_statistics_with_alert_manager(self):
        """测试获取统计信息（包含告警统计）"""
        mock_alert_manager = Mock()
        mock_alert_manager.get_alerts_summary.return_value = {
            'total': 10,
            'active': 5,
            'resolved': 5
        }
        self.manager.set_alert_manager(mock_alert_manager)
        
        stats = self.manager.get_statistics()
        
        assert 'alerts_total' in stats
        assert 'alerts_active' in stats
        assert 'alerts_resolved' in stats
        assert stats['alerts_total'] == 10

    def test_cleanup_old_data(self):
        """测试清理过期数据"""
        mock_alert_manager = Mock()
        mock_alert_manager.clear_resolved_alerts.return_value = 5
        self.manager.set_alert_manager(mock_alert_manager)
        
        # 添加一些测试数据
        for i in range(3):
            self.manager.record_operation(f"op_{i}", 1.0, True)
            self.manager.record_system_resources(
                cpu_percent=50.0,
                memory_percent=50.0,
                disk_usage={"/": 30.0},
                network_io={}
            )
        
        cleaned_count = self.manager.cleanup_old_data(max_age_days=0)  # 清理所有数据
        
        # 由于数据是刚刚创建的，应该全部被清理
        assert cleaned_count > 0
        mock_alert_manager.clear_resolved_alerts.assert_called_once_with(0)

    def test_cleanup_old_data_no_alert_manager(self):
        """测试清理过期数据（无告警管理器）"""
        # 添加一些测试数据
        self.manager.record_operation("test_op", 1.0, True)
        
        cleaned_count = self.manager.cleanup_old_data(max_age_days=0)
        assert cleaned_count >= 0  # 应该不会抛出异常
