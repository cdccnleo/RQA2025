"""
测试性能监控器

覆盖 performance_monitor.py 中的 PerformanceMonitor 类
"""

import time
import threading
from unittest.mock import Mock, patch
from src.infrastructure.logging.monitors.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor:
    """PerformanceMonitor 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            assert monitor.name == "performance_monitor"
            assert monitor.sample_interval == 1.0
            assert monitor.retention_period == 3600
            assert monitor._running == True
            assert hasattr(monitor, '_lock')
            assert monitor._stats['total_logs'] == 0
            assert monitor._stats['error_logs'] == 0
            assert monitor._stats['warning_logs'] == 0

    def test_init_custom(self):
        """测试自定义初始化"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor(
                name="custom_monitor",
                sample_interval=2.0,
                retention_period=7200
            )

            assert monitor.name == "custom_monitor"
            assert monitor.sample_interval == 2.0
            assert monitor.retention_period == 7200

    def test_record_log_basic(self):
        """测试基本日志记录"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor.record_log("INFO", "Test message")

            assert monitor._stats['total_logs'] == 1
            assert monitor._stats['error_logs'] == 0
            assert monitor._stats['warning_logs'] == 0

    def test_record_log_error(self):
        """测试错误日志记录"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor.record_log("ERROR", "Error message")
            monitor.record_log("CRITICAL", "Critical message")

            assert monitor._stats['total_logs'] == 2
            assert monitor._stats['error_logs'] == 2
            assert monitor._stats['warning_logs'] == 0

    def test_record_log_warning(self):
        """测试警告日志记录"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor.record_log("WARNING", "Warning message")

            assert monitor._stats['total_logs'] == 1
            assert monitor._stats['error_logs'] == 0
            assert monitor._stats['warning_logs'] == 1

    def test_record_log_with_response_time(self):
        """测试带响应时间的日志记录"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor.record_log("INFO", "Test message", response_time=0.5)

            assert len(monitor._metrics['response_times']) == 1
            assert monitor._metrics['response_times'][0] == 500  # 转换为ms

    def test_record_log_with_queue_size(self):
        """测试带队列大小的日志记录"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor.record_log("INFO", "Test message", queue_size=100)

            assert len(monitor._metrics['queue_sizes']) == 1
            assert monitor._metrics['queue_sizes'][0] == 100

    def test_get_throughput(self):
        """测试吞吐量获取"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            # 模拟经过的时间和日志数量
            monitor._stats['start_time'] = time.time() - 10  # 10秒前开始
            monitor._stats['total_logs'] = 100  # 100个日志

            throughput = monitor.get_throughput()
            assert abs(throughput - 10.0) < 0.01  # 允许小误差

    def test_get_throughput_zero_time(self):
        """测试零时间情况下的吞吐量"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            # 模拟刚刚开始
            monitor._stats['start_time'] = time.time()
            monitor._stats['total_logs'] = 50

            throughput = monitor.get_throughput()
            assert throughput >= 0  # 应该返回非负值

    def test_get_error_rate(self):
        """测试错误率获取"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor._stats['total_logs'] = 100
            monitor._stats['error_logs'] = 5

            error_rate = monitor.get_error_rate()
            assert error_rate == 5.0  # 5%

    def test_get_error_rate_zero_logs(self):
        """测试零日志情况下的错误率"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor._stats['total_logs'] = 0
            monitor._stats['error_logs'] = 0

            error_rate = monitor.get_error_rate()
            assert error_rate == 0.0

    @patch('psutil.Process')
    def test_get_memory_usage(self, mock_process_class):
        """测试内存使用获取"""
        with patch('threading.Thread'):
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 100 * 1024 * 1024  # 100MB in bytes
            mock_process.memory_info.return_value = mock_memory_info
            mock_process_class.return_value = mock_process

            monitor = PerformanceMonitor()

            memory_usage = monitor.get_memory_usage()
            assert memory_usage == 100.0  # 100MB

    @patch('psutil.Process')
    def test_get_memory_usage_error(self, mock_process_class):
        """测试内存使用获取错误处理"""
        with patch('threading.Thread'):
            mock_process = Mock()
            mock_process.memory_info.side_effect = Exception("Memory info error")
            mock_process_class.return_value = mock_process

            monitor = PerformanceMonitor()

            memory_usage = monitor.get_memory_usage()
            assert memory_usage == 0.0

    @patch('psutil.Process')
    def test_get_cpu_usage(self, mock_process_class):
        """测试CPU使用率获取"""
        with patch('threading.Thread'):
            mock_process = Mock()
            mock_process.cpu_percent.return_value = 45.5
            mock_process_class.return_value = mock_process

            monitor = PerformanceMonitor()

            cpu_usage = monitor.get_cpu_usage()
            assert cpu_usage == 45.5

    @patch('psutil.Process')
    def test_get_cpu_usage_error(self, mock_process_class):
        """测试CPU使用率获取错误处理"""
        with patch('threading.Thread'):
            mock_process = Mock()
            mock_process.cpu_percent.side_effect = Exception("CPU info error")
            mock_process_class.return_value = mock_process

            monitor = PerformanceMonitor()

            cpu_usage = monitor.get_cpu_usage()
            assert cpu_usage == 0.0

    def test_get_average_response_time(self):
        """测试平均响应时间获取"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor._metrics['response_times'] = [100, 200, 300]  # ms

            avg_time = monitor.get_average_response_time()
            assert avg_time == 200.0

    def test_get_average_response_time_empty(self):
        """测试空响应时间列表的平均值"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            monitor._metrics['response_times'] = []

            avg_time = monitor.get_average_response_time()
            assert avg_time == 0.0

    def test_get_performance_metrics(self):
        """测试性能指标获取"""
        with patch('threading.Thread'):
            with patch.object(PerformanceMonitor, 'get_memory_usage', return_value=150.0):
                with patch.object(PerformanceMonitor, 'get_cpu_usage', return_value=25.0):
                    monitor = PerformanceMonitor()

                    monitor._stats['total_logs'] = 1000
                    monitor._stats['error_logs'] = 50
                    monitor._stats['warning_logs'] = 25
                    monitor._stats['start_time'] = time.time() - 100  # 100秒前开始

                    metrics = monitor.get_performance_metrics()

                    assert 'throughput' in metrics
                    assert 'error_rate' in metrics
                    assert 'memory_usage' in metrics
                    assert 'cpu_usage' in metrics
                    assert 'avg_response_time' in metrics
                    assert metrics['total_logs'] == 1000
                    assert metrics['error_logs'] == 50
                    assert metrics['warning_logs'] == 25
                    assert 'uptime' in metrics

    def test_get_historical_data(self):
        """测试历史数据获取"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            # 设置测试数据
            current_time = time.time()
            monitor._timestamps = [
                current_time - 7200,  # 2小时前
                current_time - 3600,  # 1小时前
                current_time - 1800,  # 30分钟前
                current_time           # 现在
            ]
            monitor._metrics['cpu_usage'] = [10, 20, 30, 40]

            # 获取最近1小时的数据
            data = monitor.get_historical_data('cpu_usage', hours=1)

            # 应该返回最近的数据点（1小时内的）
            assert len(data) >= 2  # 至少2个数据点
            # 验证数据按时间顺序排序
            for i in range(1, len(data)):
                assert data[i]['timestamp'] >= data[i-1]['timestamp']

    def test_check_thresholds(self):
        """测试阈值检查"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            # 设置高指标值来触发告警
            with patch.object(monitor, 'get_performance_metrics', return_value={
                'memory_usage': 600,  # 超过500MB
                'cpu_usage': 85,      # 超过80%
                'error_rate': 8,      # 超过5%
                'avg_response_time': 1200,  # 超过1000ms
                'throughput': 100,
                'total_logs': 1000,
                'error_logs': 80,
                'warning_logs': 20,
                'uptime': 3600
            }):
                alerts = monitor.check_thresholds()

                # 应该有4个告警
                assert len(alerts) == 4

                alert_types = [alert['type'] for alert in alerts]
                assert 'memory' in alert_types
                assert 'cpu' in alert_types
                assert 'error_rate' in alert_types
                assert 'response_time' in alert_types

                # 检查告警级别
                for alert in alerts:
                    if alert['type'] == 'error_rate':
                        assert alert['severity'] == 'ERROR'
                    else:
                        assert alert['severity'] == 'WARNING'

    def test_check_thresholds_normal(self):
        """测试正常情况下的阈值检查"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            # 设置正常指标值
            with patch.object(monitor, 'get_performance_metrics', return_value={
                'memory_usage': 200,  # 正常
                'cpu_usage': 50,      # 正常
                'error_rate': 2,      # 正常
                'avg_response_time': 500,  # 正常
                'throughput': 100,
                'total_logs': 1000,
                'error_logs': 20,
                'warning_logs': 20,
                'uptime': 3600
            }):
                alerts = monitor.check_thresholds()

                # 不应该有告警
                assert len(alerts) == 0

    def test_check_health_healthy(self):
        """测试健康状态检查 - 健康"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            with patch.object(monitor, 'get_performance_metrics', return_value={
                'memory_usage': 200,
                'cpu_usage': 50,
                'error_rate': 2,
                'avg_response_time': 500,
                'throughput': 100,
                'total_logs': 1000,
                'error_logs': 20,
                'warning_logs': 20,
                'uptime': 3600
            }):
                health = monitor._check_health()

                assert health['status'] == 'healthy'
                assert health['issues'] == []
                assert 'metrics' in health
                assert 'timestamp' in health

    def test_check_health_warning(self):
        """测试健康状态检查 - 警告"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            with patch.object(monitor, 'get_performance_metrics', return_value={
                'memory_usage': 1200,  # 高内存 (超过1GB)
                'cpu_usage': 50,
                'error_rate': 2,
                'avg_response_time': 500,
                'throughput': 100,
                'total_logs': 1000,
                'error_logs': 20,
                'warning_logs': 20,
                'uptime': 3600
            }):
                health = monitor._check_health()

                assert health['status'] == 'warning'
                assert len(health['issues']) == 1
                assert 'High memory usage' in health['issues']

    def test_check_health_critical(self):
        """测试健康状态检查 - 严重"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            with patch.object(monitor, 'get_performance_metrics', return_value={
                'memory_usage': 1200,  # 高内存 (>1000)
                'cpu_usage': 95,       # 高CPU (>90)
                'error_rate': 15,      # 高错误率 (>10)
                'avg_response_time': 500,
                'throughput': 100,
                'total_logs': 1000,
                'error_logs': 150,
                'warning_logs': 20,
                'uptime': 3600
            }):
                health = monitor._check_health()

                assert health['status'] == 'critical'
                assert len(health['issues']) == 3

    def test_check_health_error(self):
        """测试健康状态检查 - 错误"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            with patch.object(monitor, 'get_performance_metrics', side_effect=Exception("Test error")):
                health = monitor._check_health()

                assert health['status'] == 'error'
                assert 'error' in health
                assert health['error'] == 'Test error'

    def test_collect_metrics(self):
        """测试指标收集"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            with patch.object(monitor, 'get_performance_metrics', return_value={'test': 'data'}):
                metrics = monitor._collect_metrics()

                assert metrics == {'test': 'data'}

    def test_detect_anomalies(self):
        """测试异常检测"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            with patch.object(monitor, 'check_thresholds', return_value=[{'type': 'test'}]):
                anomalies = monitor._detect_anomalies()

                assert anomalies == [{'type': 'test'}]

    def test_stop(self):
        """测试停止监控"""
        with patch('threading.Thread'):
            monitor = PerformanceMonitor()

            # 模拟线程
            mock_thread = Mock()
            monitor._monitor_thread = mock_thread

            monitor.stop()

            assert monitor._running == False
            mock_thread.join.assert_called_once_with(timeout=2.0)
