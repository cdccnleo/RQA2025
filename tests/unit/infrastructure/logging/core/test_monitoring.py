"""
Log System Monitoring 单元测试

测试日志系统监控、指标收集和健康检查功能。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time
import threading
from datetime import datetime, timedelta

from src.infrastructure.logging.core.monitoring import (
    LogSystemMetrics,
    MetricsCollector,
    HealthChecker,
    AlertManager,
    LogSystemMonitor,
    LoggingMonitor,
)
from src.infrastructure.logging.core.interfaces import LogLevel


class TestLogSystemMetrics:
    """测试日志系统指标数据类"""

    def test_init_default(self):
        """测试默认初始化"""
        metrics = LogSystemMetrics()

        assert metrics.total_logs_processed == 0
        assert metrics.logs_per_second == 0.0
        assert metrics.error_count == 0
        assert metrics.warning_count == 0
        assert metrics.average_processing_time == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.uptime_seconds == 0.0
        assert metrics.active_loggers == 0
        assert metrics.queued_logs == 0
        assert isinstance(metrics.last_health_check, datetime)

    def test_init_with_values(self):
        """测试带值初始化"""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        metrics = LogSystemMetrics(
            total_logs_processed=1000,
            logs_per_second=50.5,
            error_count=10,
            warning_count=25,
            average_processing_time=0.002,
            memory_usage_mb=128.5,
            uptime_seconds=3600.0,
            active_loggers=5,
            queued_logs=3,
            last_health_check=custom_time
        )

        assert metrics.total_logs_processed == 1000
        assert metrics.logs_per_second == 50.5
        assert metrics.error_count == 10
        assert metrics.warning_count == 25
        assert metrics.average_processing_time == 0.002
        assert metrics.memory_usage_mb == 128.5
        assert metrics.uptime_seconds == 3600.0
        assert metrics.active_loggers == 5
        assert metrics.queued_logs == 3
        assert metrics.last_health_check == custom_time


class TestMetricsCollector:
    """测试指标收集器"""

    @pytest.fixture
    def metrics_collector(self):
        """创建指标收集器实例"""
        return MetricsCollector()

    def test_init(self, metrics_collector):
        """测试初始化"""
        assert isinstance(metrics_collector._metrics, LogSystemMetrics)
        assert isinstance(metrics_collector._start_time, float)
        assert hasattr(metrics_collector, '_lock')

    def test_record_log_processed_info(self, metrics_collector):
        """测试记录INFO级别日志处理"""
        initial_count = metrics_collector._metrics.total_logs_processed

        metrics_collector.record_log_processed(LogLevel.INFO, 0.001)

        assert metrics_collector._metrics.total_logs_processed == initial_count + 1
        # INFO级别不应该增加错误或警告计数
        assert metrics_collector._metrics.error_count == 0
        assert metrics_collector._metrics.warning_count == 0

    def test_record_log_processed_warning(self, metrics_collector):
        """测试记录WARNING级别日志处理"""
        metrics_collector.record_log_processed(LogLevel.WARNING, 0.002)

        assert metrics_collector._metrics.warning_count == 1
        assert metrics_collector._metrics.error_count == 0

    def test_record_log_processed_error(self, metrics_collector):
        """测试记录ERROR级别日志处理"""
        metrics_collector.record_log_processed(LogLevel.ERROR, 0.003)
        metrics_collector.record_log_processed(LogLevel.CRITICAL, 0.004)

        # CRITICAL不计入error_count，只有ERROR才计入
        assert metrics_collector._metrics.error_count == 1

    def test_record_multiple_logs(self, metrics_collector):
        """测试记录多个日志的统计"""
        # 记录多个不同级别的日志
        logs = [
            (LogLevel.INFO, 0.001),
            (LogLevel.WARNING, 0.002),
            (LogLevel.ERROR, 0.003),
            (LogLevel.INFO, 0.001),
            (LogLevel.WARNING, 0.002),
        ]

        for level, processing_time in logs:
            metrics_collector.record_log_processed(level, processing_time)

        metrics = metrics_collector._metrics
        assert metrics.total_logs_processed == 5
        assert metrics.warning_count == 2
        assert metrics.error_count == 1

    def test_update_active_loggers(self, metrics_collector):
        """测试更新活跃日志器数量"""
        metrics_collector.update_active_loggers(5)
        assert metrics_collector._metrics.active_loggers == 5

        metrics_collector.update_active_loggers(3)
        assert metrics_collector._metrics.active_loggers == 3

    def test_update_queue_size(self, metrics_collector):
        """测试更新队列大小"""
        metrics_collector.update_queue_size(10)
        assert metrics_collector._metrics.queued_logs == 10

        metrics_collector.update_queue_size(0)
        assert metrics_collector._metrics.queued_logs == 0

    def test_get_metrics(self, metrics_collector):
        """测试获取指标"""
        # 添加一些数据
        metrics_collector.record_log_processed(LogLevel.INFO, 0.001)
        metrics_collector.update_active_loggers(2)
        metrics_collector.update_queue_size(5)

        metrics = metrics_collector.get_metrics()

        assert isinstance(metrics, LogSystemMetrics)
        assert metrics.total_logs_processed == 1
        assert metrics.active_loggers == 2
        assert metrics.queued_logs == 5

    def test_get_raw_metrics(self, metrics_collector):
        """测试获取原始指标数据"""
        raw_metrics = metrics_collector.get_raw_metrics()

        assert isinstance(raw_metrics, LogSystemMetrics)
        assert hasattr(raw_metrics, 'total_logs_processed')
        assert hasattr(raw_metrics, 'active_loggers')
        assert hasattr(raw_metrics, 'queued_logs')

    def test_calculate_logs_per_second(self, metrics_collector):
        """测试计算每秒日志处理量"""
        # 记录一些日志
        for i in range(10):
            metrics_collector.record_log_processed(LogLevel.INFO, 0.001)

        # 手动设置一些时间流逝来测试计算
        # 注意：实际实现可能需要更复杂的时间计算逻辑
        metrics = metrics_collector.get_metrics()
        assert metrics.total_logs_processed == 10

    def test_thread_safety(self, metrics_collector):
        """测试线程安全性"""
        results = []

        def worker():
            for i in range(100):
                metrics_collector.record_log_processed(LogLevel.INFO, 0.001)
                results.append(metrics_collector._metrics.total_logs_processed)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证最终计数正确
        final_metrics = metrics_collector.get_metrics()
        assert final_metrics.total_logs_processed == 500  # 5个线程 * 100个日志


class TestHealthChecker:
    """测试健康检查器"""

    @pytest.fixture
    def health_checker(self, mock_metrics_collector):
        """创建健康检查器实例"""
        return HealthChecker(mock_metrics_collector)

    @pytest.fixture
    def mock_metrics_collector(self):
        """创建Mock指标收集器"""
        collector = Mock()
        collector.get_metrics.return_value = LogSystemMetrics(
            total_logs_processed=1000,
            error_count=5,
            warning_count=10,
            average_processing_time=0.005,
            memory_usage_mb=150.0,
            uptime_seconds=3600.0,
            active_loggers=3,
            queued_logs=2
        )
        return collector

    def test_init(self, health_checker, mock_metrics_collector):
        """测试初始化"""
        assert health_checker.metrics_collector == mock_metrics_collector

    def test_check_health_healthy(self, health_checker, mock_metrics_collector):
        """测试健康状态检查"""
        result = health_checker.check_health()

        assert result["status"] == "healthy"
        assert len(result["issues"]) == 0
        assert "timestamp" in result
        assert "metrics" in result

    def test_check_health_error_rate_high(self, health_checker, mock_metrics_collector):
        """测试高错误率情况"""
        # 设置高错误率：60个错误，1000个日志 = 6% (>5%触发警告)
        mock_metrics_collector.get_metrics.return_value.error_count = 60

        result = health_checker.check_health()

        assert result["status"] == "warning"
        assert len(result["issues"]) > 0

    def test_check_health_performance_poor(self, health_checker, mock_metrics_collector):
        """测试性能差的情况"""
        # 设置慢的平均处理时间 (>0.5秒触发警告)
        mock_metrics_collector.get_metrics.return_value.average_processing_time = 0.8

        result = health_checker.check_health()

        assert result["status"] == "warning"
        assert len(result["issues"]) > 0

    def test_check_health_queue_backlog(self, health_checker, mock_metrics_collector):
        """测试队列积压情况"""
        # 设置大的队列积压 (>500触发警告)
        mock_metrics_collector.get_metrics.return_value.queued_logs = 600

        result = health_checker.check_health()

        assert result["status"] == "warning"
        assert len(result["issues"]) > 0

    def test_check_health_memory_high(self, health_checker, mock_metrics_collector):
        """测试内存使用高的情况"""
        # 设置高的内存使用 (>200MB触发警告)
        mock_metrics_collector.get_metrics.return_value.memory_usage_mb = 300.0

        result = health_checker.check_health()

        assert result["status"] == "warning"
        assert len(result["issues"]) > 0

    def test_check_health_critical(self, health_checker, mock_metrics_collector):
        """测试严重问题的情况"""
        # 设置多个严重问题
        mock_metrics_collector.get_metrics.return_value.error_count = 100  # 10%错误率
        mock_metrics_collector.get_metrics.return_value.queued_logs = 200  # 严重积压

        result = health_checker.check_health()

        assert result["status"] == "critical"
        assert len(result["issues"]) > 0




class TestLogSystemMonitor:
    """测试日志系统监控器"""

    @pytest.fixture
    def log_system_monitor(self):
        """创建日志系统监控器实例"""
        return LogSystemMonitor()

    def test_init(self, log_system_monitor):
        """测试初始化"""
        # 基本的初始化检查
        assert hasattr(log_system_monitor, '_metrics_collector')
        assert hasattr(log_system_monitor, '_health_checker')
        assert hasattr(log_system_monitor, '_alert_manager')

    def test_basic_functionality(self, log_system_monitor):
        """测试基本功能"""
        # 测试可以实例化
        assert log_system_monitor is not None

        # 测试有基本方法
        assert hasattr(log_system_monitor, 'record_log_processed')
        assert hasattr(log_system_monitor, 'get_metrics')
        assert hasattr(log_system_monitor, 'get_health_status')


class TestLoggingMonitor:
    """测试日志监控器"""

    @pytest.fixture
    def logging_monitor(self):
        """创建日志监控器实例"""
        return LoggingMonitor()

    def test_init(self, logging_monitor):
        """测试初始化"""
        assert logging_monitor is not None

    def test_basic_operations(self, logging_monitor):
        """测试基本操作"""
        # 测试记录日志处理
        logging_monitor.record_log_processed(LogLevel.INFO, 0.001)
        logging_monitor.record_log_processed(LogLevel.ERROR, 0.002)

        # 测试获取指标
        metrics = logging_monitor.get_metrics()
        assert hasattr(metrics, 'total_logs_processed')
        assert hasattr(metrics, 'error_count')

        # 测试获取健康状态
        health = logging_monitor.get_health_status()
        assert isinstance(health, dict)


class TestAlertManager:
    """测试告警管理器"""

    @pytest.fixture
    def metrics_collector(self):
        """创建指标收集器实例"""
        return MetricsCollector()

    @pytest.fixture
    def alert_manager(self, metrics_collector):
        """创建告警管理器实例"""
        return AlertManager(metrics_collector)

    def test_init(self, alert_manager):
        """测试初始化"""
        assert alert_manager.metrics_collector is not None
        assert alert_manager._alert_callbacks == []
        assert hasattr(alert_manager, '_lock')

    def test_add_alert_callback(self, alert_manager):
        """测试添加告警回调"""
        callback1 = Mock()
        callback2 = Mock()

        alert_manager.add_alert_callback(callback1)
        alert_manager.add_alert_callback(callback2)

        assert len(alert_manager._alert_callbacks) == 2
        assert callback1 in alert_manager._alert_callbacks
        assert callback2 in alert_manager._alert_callbacks

    def test_check_alerts_no_alerts(self, alert_manager):
        """测试检查告警 - 无告警"""
        # 创建具有正常值的指标
        alert_manager.metrics_collector._metrics.total_logs_processed = 100
        alert_manager.metrics_collector._metrics.error_count = 1
        alert_manager.metrics_collector._metrics.warning_count = 1
        alert_manager.metrics_collector._metrics.average_processing_time = 0.1
        alert_manager.metrics_collector._metrics.queued_logs = 10

        # 添加回调
        callback = Mock()
        alert_manager.add_alert_callback(callback)

        # 检查告警
        alert_manager.check_alerts()

        # 回调不应该被调用
        callback.assert_not_called()

    @patch('src.infrastructure.logging.core.monitoring.MetricsCollector')
    def test_check_alerts_with_alerts(self, mock_metrics_collector_class, alert_manager):
        """测试检查告警 - 有告警"""
        mock_metrics = Mock()
        mock_metrics.total_logs_processed = 100
        mock_metrics.error_count = 20  # 20%错误率
        mock_metrics.warning_count = 0
        mock_metrics.average_processing_time = 0.1
        mock_metrics.queued_logs = 10

        mock_collector = Mock()
        mock_collector.get_metrics.return_value = mock_metrics
        alert_manager.metrics_collector = mock_collector

        # 添加回调
        callback = Mock()
        alert_manager.add_alert_callback(callback)

        # 检查告警
        alert_manager.check_alerts()

        # 回调应该被调用
        callback.assert_called()

    def test_collect_all_alerts(self, alert_manager):
        """测试收集所有告警"""
        mock_metrics = Mock()
        mock_metrics.total_logs_processed = 100
        mock_metrics.error_count = 20
        mock_metrics.warning_count = 0
        mock_metrics.average_processing_time = 2.0  # 超过阈值
        mock_metrics.queued_logs = 1000  # 超过阈值

        alerts = alert_manager._collect_all_alerts(mock_metrics)

        assert len(alerts) > 0

    def test_check_error_rate_alerts_high(self, alert_manager):
        """测试错误率告警 - 高错误率"""
        mock_metrics = Mock()
        mock_metrics.total_logs_processed = 100
        mock_metrics.error_count = 20
        mock_metrics.warning_count = 0

        alerts = alert_manager._check_error_rate_alerts(mock_metrics)

        assert len(alerts) > 0
        assert "错误率严重超标" in alerts[0]["message"]

    def test_check_performance_alerts_slow(self, alert_manager):
        """测试性能告警 - 处理慢"""
        mock_metrics = Mock()
        mock_metrics.average_processing_time = 2.1  # 超过2.0阈值

        alerts = alert_manager._check_performance_alerts(mock_metrics)

        assert len(alerts) > 0
        assert "处理性能严重下降" in alerts[0]["message"]

    def test_check_queue_backlog_alerts_high(self, alert_manager):
        """测试队列积压告警 - 高积压"""
        mock_metrics = Mock()
        mock_metrics.queued_logs = 2500  # 超过2000阈值

        alerts = alert_manager._check_queue_backlog_alerts(mock_metrics)

        assert len(alerts) > 0
        assert "队列积压严重" in alerts[0]["message"]

    def test_trigger_alert_callbacks(self, alert_manager):
        """测试触发告警回调"""
        callback1 = Mock()
        callback2 = Mock()

        alert_manager.add_alert_callback(callback1)
        alert_manager.add_alert_callback(callback2)

        alerts = [{"type": "error_rate", "message": "测试告警"}]

        alert_manager._trigger_alert_callbacks(alerts)

        callback1.assert_called_once_with(alerts[0])
        callback2.assert_called_once_with(alerts[0])

    def test_trigger_alert_callbacks_exception_handling(self, alert_manager):
        """测试触发告警回调 - 异常处理"""
        callback1 = Mock(side_effect=Exception("Callback failed"))
        callback2 = Mock()

        alert_manager.add_alert_callback(callback1)
        alert_manager.add_alert_callback(callback2)

        alerts = [{"type": "error_rate", "message": "测试告警"}]

        # 不应该抛出异常
        alert_manager._trigger_alert_callbacks(alerts)

        # 第二个回调仍然会被调用
        callback2.assert_called_once()


class TestLogSystemMonitorExtended:
    """测试日志系统监控器的扩展功能"""

    @pytest.fixture
    def log_system_monitor(self):
        """创建日志系统监控器实例"""
        return LogSystemMonitor()

    def test_record_log_processed_with_alert_check(self, log_system_monitor):
        """测试记录日志处理时检查告警"""
        # 这个方法应该调用告警检查
        log_system_monitor.record_log_processed(LogLevel.ERROR, 0.1)

        # 验证指标被记录
        metrics = log_system_monitor.get_metrics()
        assert metrics.total_logs_processed >= 1

    def test_update_active_loggers(self, log_system_monitor):
        """测试更新活跃日志器数量"""
        log_system_monitor.update_active_loggers(5)

        metrics = log_system_monitor.get_metrics()
        assert metrics.active_loggers == 5

    def test_update_queue_size(self, log_system_monitor):
        """测试更新队列大小"""
        log_system_monitor.update_queue_size(100)

        metrics = log_system_monitor.get_metrics()
        assert metrics.queued_logs == 100

    def test_add_alert_callback(self, log_system_monitor):
        """测试添加告警回调"""
        callback = Mock()
        log_system_monitor.add_alert_callback(callback)

        # 验证回调被添加到告警管理器
        assert callback in log_system_monitor._alert_manager._alert_callbacks

    def test_get_health_status(self, log_system_monitor):
        """测试获取健康状态"""
        health = log_system_monitor.get_health_status()

        assert isinstance(health, dict)
        assert "status" in health
        assert "timestamp" in health

    def test_get_metrics(self, log_system_monitor):
        """测试获取指标"""
        metrics = log_system_monitor.get_metrics()

        assert isinstance(metrics, LogSystemMetrics)
        assert hasattr(metrics, 'total_logs_processed')

    def test_shutdown(self, log_system_monitor):
        """测试关闭监控器"""
        # 记录一些数据
        log_system_monitor.record_log_processed(LogLevel.INFO)

        # 关闭
        log_system_monitor.shutdown()

        # 验证监控线程停止
        assert not log_system_monitor._monitoring_active

    def test_monitoring_loop_functionality(self, log_system_monitor):
        """测试监控循环功能"""
        # 等待监控循环执行几次
        time.sleep(0.1)

        # 应该有运行时间
        metrics = log_system_monitor.get_metrics()
        assert metrics.uptime_seconds > 0