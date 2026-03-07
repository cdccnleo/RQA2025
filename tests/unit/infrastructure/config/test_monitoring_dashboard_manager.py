"""测试monitoring/dashboard_manager模块"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

try:
    from src.infrastructure.config.monitoring.dashboard_manager import UnifiedMonitoringManager
except ImportError:
    UnifiedMonitoringManager = None

try:
    # 尝试导入相关依赖类
    from src.infrastructure.config.monitoring.dashboard_alerts import AlertManager
    from src.infrastructure.config.monitoring.dashboard_collectors import MetricsCollector
    from src.infrastructure.config.monitoring.dashboard_models import (
        MonitoringConfig, PerformanceMetrics, SystemResources, AlertSeverity
    )
except ImportError:
    AlertManager = None
    MetricsCollector = None
    MonitoringConfig = None
    PerformanceMetrics = None
    SystemResources = None
    AlertSeverity = None


class TestUnifiedMonitoringManager:
    """测试UnifiedMonitoringManager类"""
    
    def setup_method(self):
        """测试前准备"""
        if UnifiedMonitoringManager is None:
            pytest.skip("UnifiedMonitoringManager导入失败，跳过测试")

    def test_initialization_default_config(self):
        """测试使用默认配置初始化"""
        manager = UnifiedMonitoringManager()
        
        assert manager.config is not None
        assert manager._metrics_collector is None
        assert manager._alert_manager is None
        assert manager._performance_history == []
        assert manager._system_resources_history == []
        assert manager._max_history_size == 10000
        assert manager._stats['total_operations'] == 0
        assert manager._stats['successful_operations'] == 0
        assert manager._stats['failed_operations'] == 0
        assert manager._stats['alerts_triggered'] == 0

    def test_initialization_custom_config(self):
        """测试使用自定义配置初始化"""
        if MonitoringConfig is not None:
            config = MonitoringConfig()
            manager = UnifiedMonitoringManager(config)
            assert manager.config == config
        else:
            # 如果不能导入MonitoringConfig，用Mock测试
            config_mock = StandardMockBuilder.create_config_mock()
            manager = UnifiedMonitoringManager(config_mock)
            assert manager.config == config_mock

    def test_set_metrics_collector(self):
        """测试设置指标收集器"""
        manager = UnifiedMonitoringManager()
        
        # 使用MagicMock来模拟MetricsCollector，因为它是抽象类
        collector = MagicMock(spec=MetricsCollector)
        collector.collect_config_metrics.return_value = {}
        collector.collect_system_metrics.return_value = {}
        
        manager.set_metrics_collector(collector)
        assert manager._metrics_collector == collector

    def test_set_alert_manager(self):
        """测试设置告警管理器"""
        manager = UnifiedMonitoringManager()
        
        # 使用MagicMock来模拟AlertManager，因为它是抽象类
        alert_manager = MagicMock(spec=AlertManager)
        alert_manager.create_alert.return_value = True
        alert_manager.acknowledge_alert.return_value = True
        alert_manager.resolve_alert.return_value = True
        
        manager.set_alert_manager(alert_manager)
        assert manager._alert_manager == alert_manager

    def test_start_monitoring_disabled_config(self):
        """测试启动监控但配置被禁用"""
        manager = UnifiedMonitoringManager()
        
        # 禁用配置
        manager.config.enabled = False
        
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            manager.start_monitoring()
            mock_logger.info.assert_called_with("监控已禁用")

    def test_start_monitoring_enabled_config(self):
        """测试启动监控且配置启用"""
        manager = UnifiedMonitoringManager()
        manager.config.enabled = True
        
        # 设置mock收集器
        mock_collector = MagicMock()
        manager.set_metrics_collector(mock_collector)
        
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            manager.start_monitoring()
            
            mock_collector.start_collection.assert_called_once()
            mock_logger.info.assert_any_call("指标收集已启动")
            mock_logger.info.assert_any_call("统一监控管理器已启动")

    def test_stop_monitoring(self):
        """测试停止监控"""
        manager = UnifiedMonitoringManager()
        
        # 设置mock收集器
        mock_collector = MagicMock()
        manager.set_metrics_collector(mock_collector)
        
        with patch('src.infrastructure.config.monitoring.dashboard_manager.logger') as mock_logger:
            manager.stop_monitoring()
            
            mock_collector.stop_collection.assert_called_once()
            mock_logger.info.assert_any_call("指标收集已停止")
            mock_logger.info.assert_any_call("统一监控管理器已停止")

    def test_record_operation_success(self):
        """测试记录成功操作"""
        manager = UnifiedMonitoringManager()
        
        # 设置mock收集器
        mock_collector = MagicMock()
        manager.set_metrics_collector(mock_collector)
        
        manager.record_operation(
            operation_type="test_operation",
            duration=1.5,
            success=True,
            metadata={"test": "data"}
        )
        
        # 检查统计更新
        assert manager._stats['total_operations'] == 1
        assert manager._stats['successful_operations'] == 1
        assert manager._stats['failed_operations'] == 0
        
        # 检查历史记录
        assert len(manager._performance_history) == 1
        performance_metric = manager._performance_history[0]
        assert performance_metric.operation_type == "test_operation"
        assert performance_metric.duration == 1.5
        assert performance_metric.success is True
        
        # 检查收集器调用
        mock_collector.record_operation.assert_called_once_with(
            "test_operation", 1.5, True, {"test": "data"}
        )

    def test_record_operation_failure(self):
        """测试记录失败操作"""
        manager = UnifiedMonitoringManager()
        
        manager.record_operation(
            operation_type="failed_operation",
            duration=2.0,
            success=False,
            error_message="Test error",
            metadata={"error": "details"}
        )
        
        # 检查统计更新
        assert manager._stats['total_operations'] == 1
        assert manager._stats['successful_operations'] == 0
        assert manager._stats['failed_operations'] == 1
        
        # 检查历史记录
        assert len(manager._performance_history) == 1
        performance_metric = manager._performance_history[0]
        assert performance_metric.success is False
        assert performance_metric.error_message == "Test error"

    def test_record_operation_history_limit(self):
        """测试操作历史记录限制"""
        manager = UnifiedMonitoringManager()
        manager._max_history_size = 3  # 设置较小的限制
        
        # 记录超过限制的操作次数
        for i in range(5):
            manager.record_operation(f"operation_{i}", 1.0, True)
        
        # 应该只保留最后3条记录
        assert len(manager._performance_history) == 3
        assert manager._performance_history[0].operation_type == "operation_2"
        assert manager._performance_history[-1].operation_type == "operation_4"

    def test_record_system_resources(self):
        """测试记录系统资源"""
        manager = UnifiedMonitoringManager()
        
        resources_data = {
            'cpu_percent': 75.5,
            'memory_percent': 60.0,
            'disk_usage': {'/': 80.0, '/tmp': 30.0},
            'network_io': {'bytes_sent': 1000, 'bytes_recv': 2000},
            'load_average': (1.5, 2.0, 1.8)
        }
        
        manager.record_system_resources(**resources_data)
        
        assert len(manager._system_resources_history) == 1
        system_resource = manager._system_resources_history[0]
        assert system_resource.cpu_percent == 75.5
        assert system_resource.memory_percent == 60.0
        assert system_resource.disk_usage == {'/': 80.0, '/tmp': 30.0}

    def test_record_system_resources_history_limit(self):
        """测试系统资源历史记录限制"""
        manager = UnifiedMonitoringManager()
        manager._max_history_size = 2
        
        # 记录超过限制的资源数据
        for i in range(4):
            manager.record_system_resources(
                cpu_percent=i * 25,
                memory_percent=i * 20,
                disk_usage={'/': 50.0},
                network_io={'bytes_sent': i * 100}
            )
        
        assert len(manager._system_resources_history) == 2

    @patch('src.infrastructure.config.monitoring.dashboard_manager.datetime')
    def test_get_performance_metrics_all(self, mock_datetime):
        """测试获取所有性能指标"""
        # 设置固定的当前时间
        now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        
        manager = UnifiedMonitoringManager()
        
        # 添加一些测试数据
        manager.record_operation("op1", 1.0, True)
        manager.record_operation("op2", 2.0, True)
        manager.record_operation("op1", 1.5, False)
        
        metrics = manager.get_performance_metrics()
        assert len(metrics) == 3

    @patch('src.infrastructure.config.monitoring.dashboard_manager.datetime')
    def test_get_performance_metrics_filtered_by_type(self, mock_datetime):
        """测试按操作类型过滤性能指标"""
        now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        
        manager = UnifiedMonitoringManager()
        
        # 添加不同类型的操作
        manager.record_operation("type1", 1.0, True)
        manager.record_operation("type2", 2.0, True)
        manager.record_operation("type1", 1.5, True)
        
        metrics = manager.get_performance_metrics(operation_type="type1")
        assert len(metrics) == 2
        assert all(m.operation_type == "type1" for m in metrics)

    @patch('src.infrastructure.config.monitoring.dashboard_manager.datetime')
    def test_get_performance_metrics_filtered_by_time(self, mock_datetime):
        """测试按时间范围过滤性能指标"""
        now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        
        manager = UnifiedMonitoringManager()
        
        # 由于我们使用固定的时间，所有记录的指标都在同一时间
        manager.record_operation("test", 1.0, True)
        
        # 使用1小时的时间范围
        time_range = timedelta(hours=1)
        metrics = manager.get_performance_metrics(time_range=time_range)
        assert len(metrics) == 1

    def test_get_system_resources_all(self):
        """测试获取所有系统资源"""
        manager = UnifiedMonitoringManager()
        
        # 添加一些测试数据
        manager.record_system_resources(50.0, 40.0, {'/': 60.0}, {'sent': 100})
        manager.record_system_resources(60.0, 50.0, {'/': 70.0}, {'sent': 200})
        
        resources = manager.get_system_resources()
        assert len(resources) == 2

    @patch('src.infrastructure.config.monitoring.dashboard_manager.datetime')
    def test_get_system_resources_filtered_by_time(self, mock_datetime):
        """测试按时间范围过滤系统资源"""
        now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        
        manager = UnifiedMonitoringManager()
        
        manager.record_system_resources(50.0, 40.0, {'/': 60.0}, {'sent': 100})
        
        time_range = timedelta(hours=1)
        resources = manager.get_system_resources(time_range=time_range)
        assert len(resources) == 1

    @patch('src.infrastructure.config.monitoring.dashboard_manager.datetime')
    def test_get_statistics(self, mock_datetime):
        """测试获取统计信息"""
        start_time = datetime(2023, 1, 1, 10, 0, 0)
        now = datetime(2023, 1, 1, 12, 0, 0)
        
        manager = UnifiedMonitoringManager()
        manager._stats = {
            'start_time': start_time,
            'total_operations': 10,
            'successful_operations': 8,
            'failed_operations': 2,
            'alerts_triggered': 1
        }
        manager._alert_manager = None
        
        mock_datetime.now.return_value = now
        
        stats = manager.get_statistics()
        
        assert stats['total_operations'] == 10
        assert stats['successful_operations'] == 8
        assert stats['failed_operations'] == 2
        assert stats['uptime'] == 7200  # 2小时
        assert stats['success_rate'] == 0.8

    def test_get_statistics_with_alert_manager(self):
        """测试带有告警管理器的统计信息"""
        manager = UnifiedMonitoringManager()
        
        # Mock告警管理器
        mock_alert_manager = MagicMock()
        mock_alert_manager.get_alerts_summary.return_value = {
            'total': 5,
            'active': 2,
            'resolved': 3
        }
        manager._alert_manager = mock_alert_manager
        
        stats = manager.get_statistics()
        
        assert stats['alerts_total'] == 5
        assert stats['alerts_active'] == 2
        assert stats['alerts_resolved'] == 3
        mock_alert_manager.get_alerts_summary.assert_called_once()

    @patch('src.infrastructure.config.monitoring.dashboard_manager.datetime')
    def test_cleanup_old_data(self, mock_datetime):
        """测试清理过期数据"""
        now = datetime(2023, 1, 31, 12, 0, 0)
        mock_datetime.now.return_value = now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        manager = UnifiedMonitoringManager()
        
        # 设置一些历史数据（由于我们无法直接创建PerformanceMetrics和SystemResources对象，
        # 我们mock这些对象）
        old_performance = MagicMock()
        old_performance.timestamp = datetime(2023, 1, 1)  # 30天前
        
        recent_performance = MagicMock()
        recent_performance.timestamp = datetime(2023, 1, 25)  # 6天前
        
        manager._performance_history = [old_performance, recent_performance]
        manager._system_resources_history = [old_performance]  # 用一个相同的mock
        
        mock_alert_manager = MagicMock()
        mock_alert_manager.clear_resolved_alerts.return_value = 1
        manager._alert_manager = mock_alert_manager
        
        cleaned_count = manager.cleanup_old_data(max_age_days=7)
        
        # 应该清理掉30天前的数据，保留6天前的数据
        assert cleaned_count >= 2  # 至少清理了2条记录
        assert len(manager._performance_history) == 1
        assert len(manager._system_resources_history) == 0
        mock_alert_manager.clear_resolved_alerts.assert_called_once_with(7)

    def test_check_alerts_with_alert_manager(self):
        """测试检查告警"""
        manager = UnifiedMonitoringManager()
        
        # Mock告警管理器和配置
        mock_alert_manager = MagicMock()
        manager.set_alert_manager(mock_alert_manager)
        manager.config.alerting_enabled = True
        
        # 创建mock性能指标
        mock_metric = MagicMock()
        mock_metric.success = False
        mock_metric.operation_type = "test_operation"
        mock_metric.error_message = "Test error"
        mock_metric.duration = 1.5
        
        # 测试失败操作的告警
        manager._check_alerts(mock_metric)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "Operation Failed" in call_args[1]['name']

    def test_check_alerts_operation_timeout(self):
        """测试操作超时告警"""
        manager = UnifiedMonitoringManager()
        
        mock_alert_manager = MagicMock()
        manager.set_alert_manager(mock_alert_manager)
        manager.config.alerting_enabled = True
        
        # 创建超时的性能指标
        mock_metric = MagicMock()
        mock_metric.success = True
        mock_metric.operation_type = "slow_operation"
        mock_metric.duration = 35.0  # 超过30秒阈值
        mock_metric.error_message = None
        
        manager._check_alerts(mock_metric)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "Operation Timeout" in call_args[1]['name']

    def test_check_resource_alerts_cpu(self):
        """测试CPU资源告警"""
        manager = UnifiedMonitoringManager()
        
        mock_alert_manager = MagicMock()
        manager.set_alert_manager(mock_alert_manager)
        manager.config.alerting_enabled = True
        
        # 创建高CPU使用率的资源记录
        mock_resource = MagicMock()
        mock_resource.cpu_percent = 95.0
        mock_resource.memory_percent = 50.0
        mock_resource.disk_usage = {'/': 70.0}
        
        manager._check_resource_alerts(mock_resource)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert call_args[1]['name'] == "High CPU Usage"

    def test_check_resource_alerts_memory(self):
        """测试内存资源告警"""
        manager = UnifiedMonitoringManager()
        
        mock_alert_manager = MagicMock()
        manager.set_alert_manager(mock_alert_manager)
        manager.config.alerting_enabled = True
        
        # 创建高内存使用率的资源记录
        mock_resource = MagicMock()
        mock_resource.cpu_percent = 50.0
        mock_resource.memory_percent = 90.0
        mock_resource.disk_usage = {'/': 70.0}
        
        manager._check_resource_alerts(mock_resource)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert call_args[1]['name'] == "High Memory Usage"

    def test_check_resource_alerts_disk(self):
        """测试磁盘资源告警"""
        manager = UnifiedMonitoringManager()
        
        mock_alert_manager = MagicMock()
        manager.set_alert_manager(mock_alert_manager)
        manager.config.alerting_enabled = True
        
        # 创建高磁盘使用率的资源记录
        mock_resource = MagicMock()
        mock_resource.cpu_percent = 50.0
        mock_resource.memory_percent = 50.0
        mock_resource.disk_usage = {'/': 96.0}
        
        manager._check_resource_alerts(mock_resource)
        
        mock_alert_manager.create_alert.assert_called_once()
        call_args = mock_alert_manager.create_alert.call_args
        assert "High Disk Usage" in call_args[1]['name']
        assert '/' in call_args[1]['name']
