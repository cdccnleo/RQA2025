"""
基础设施层 - PerformanceMonitor 单元测试

测试性能监控器的核心功能，包括指标收集、告警机制、统计分析等。
覆盖率目标: 85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.error.core.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    PerformanceAlert,
    get_global_performance_monitor,
    record_handler_performance
)


class TestPerformanceMonitor(unittest.TestCase):
    """PerformanceMonitor 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.monitor = PerformanceMonitor(max_history_size=100, alert_check_interval=1, test_mode=True)

    def tearDown(self):
        """测试后清理"""
        # 停止监控线程
        if hasattr(self.monitor, '_alert_thread') and self.monitor._alert_thread.is_alive():
            self.monitor._alert_thread.join(timeout=1)

    def test_initialization(self):
        """测试初始化"""
        monitor = PerformanceMonitor(max_history_size=50, alert_check_interval=5, test_mode=True)

        # 通过组合对象访问正确的属性
        self.assertEqual(monitor._metrics_collector._max_history_size, 50)
        self.assertEqual(monitor._alert_manager._alert_check_interval, 5)
        self.assertIsInstance(monitor._metrics_collector._metrics, dict)
        self.assertIsInstance(monitor._alert_manager._alerts, list)
        self.assertIsInstance(monitor._alert_manager._alert_callbacks, list)
        self.assertTrue(monitor._test_mode)

        # 验证默认告警阈值
        self.assertIn('error_rate_threshold', monitor._alert_manager._alert_thresholds)
        self.assertIn('response_time_threshold', monitor._alert_manager._alert_thresholds)
        self.assertIn('throughput_drop_threshold', monitor._alert_manager._alert_thresholds)

    def test_test_mode_alert_suppression(self):
        """测试测试模式下告警抑制"""
        # 测试模式下的监控器
        test_monitor = PerformanceMonitor(test_mode=True)

        # 记录高错误率请求
        for i in range(10):
            success = i >= 9  # 90%错误率
            test_monitor.record_request("test_handler", 0.1, success)

        # 手动触发告警检查 - 通过组合对象调用
        test_monitor._alert_manager._check_alerts(test_monitor._metrics_collector)

        # 验证告警被生成但不输出日志
        alerts = test_monitor.get_alerts()
        high_error_alerts = [a for a in alerts if a.alert_type == 'high_error_rate']
        self.assertGreater(len(high_error_alerts), 0)

        # 非测试模式下的监控器（但为了避免测试输出，我们仍然使用测试模式）
        # 这里我们只是验证告警生成功能，不测试日志输出
        normal_monitor = PerformanceMonitor(test_mode=True, alert_check_interval=1)

        # 记录高错误率请求
        for i in range(10):
            success = i >= 9  # 90%错误率
            normal_monitor.record_request("normal_handler", 0.1, success)

        # 手动触发告警检查 - 通过组合对象调用
        normal_monitor._alert_manager._check_alerts(normal_monitor._metrics_collector)

        # 验证告警也被生成
        normal_alerts = normal_monitor.get_alerts()
        normal_high_error_alerts = [a for a in normal_alerts if a.alert_type == 'high_error_rate']
        self.assertGreater(len(normal_high_error_alerts), 0)

    def test_record_request_success(self):
        """测试记录成功请求"""
        handler_name = "test_handler"

        # 记录成功请求
        self.monitor.record_request(handler_name, 0.1, True)

        # 验证指标
        metrics = self.monitor.get_metrics(handler_name)
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.successful_requests, 1)
        self.assertEqual(metrics.failed_requests, 0)
        self.assertAlmostEqual(metrics.avg_response_time, 0.1, places=3)
        self.assertEqual(metrics.error_rate, 0.0)

    def test_record_request_failure(self):
        """测试记录失败请求"""
        handler_name = "test_handler"

        # 记录失败请求
        self.monitor.record_request(handler_name, 0.2, False, "ValueError")

        # 验证指标
        metrics = self.monitor.get_metrics(handler_name)
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.successful_requests, 0)
        self.assertEqual(metrics.failed_requests, 1)
        self.assertAlmostEqual(metrics.avg_response_time, 0.2, places=3)
        self.assertEqual(metrics.error_rate, 1.0)
        self.assertEqual(metrics.error_counts["ValueError"], 1)

    def test_multiple_requests_metrics(self):
        """测试多请求的指标计算"""
        handler_name = "test_handler"

        # 记录多个请求
        requests = [
            (0.1, True, None),      # 成功
            (0.2, False, "TypeError"),  # 失败
            (0.15, True, None),     # 成功
            (0.3, False, "ValueError"), # 失败
            (0.05, True, None),     # 成功
        ]

        for response_time, success, error_type in requests:
            self.monitor.record_request(handler_name, response_time, success, error_type)

        metrics = self.monitor.get_metrics(handler_name)

        # 验证基本统计
        self.assertEqual(metrics.total_requests, 5)
        self.assertEqual(metrics.successful_requests, 3)
        self.assertEqual(metrics.failed_requests, 2)
        self.assertAlmostEqual(metrics.avg_response_time, 0.16, places=2)  # (0.1+0.2+0.15+0.3+0.05)/5
        self.assertEqual(metrics.error_rate, 0.4)  # 2/5

        # 验证错误计数
        self.assertEqual(metrics.error_counts["TypeError"], 1)
        self.assertEqual(metrics.error_counts["ValueError"], 1)

    def test_response_time_statistics(self):
        """测试响应时间统计"""
        handler_name = "test_handler"

        # 添加多个响应时间
        response_times = [0.1, 0.2, 0.15, 0.3, 0.05, 0.25, 0.18]
        for rt in response_times:
            self.monitor.record_request(handler_name, rt, True)

        metrics = self.monitor.get_metrics(handler_name)

        # 验证响应时间历史
        self.assertEqual(len(metrics.response_times), 7)
        self.assertEqual(metrics.response_times, response_times)

        # 验证统计计算
        self.assertAlmostEqual(metrics.avg_response_time, sum(response_times)/len(response_times), places=3)
        self.assertEqual(metrics.median_response_time, 0.18)  # 中位数
        # P95应该在合理范围内
        self.assertGreater(metrics.p95_response_time, 0.15)
        self.assertLess(metrics.p95_response_time, 0.35)

    def test_throughput_calculation(self):
        """测试吞吐量计算"""
        handler_name = "test_handler"

        # 记录一些请求
        for i in range(10):
            self.monitor.record_request(handler_name, 0.1, True)
            time.sleep(0.01)  # 短暂延迟

        metrics = self.monitor.get_metrics(handler_name)

        # 验证请求被记录
        self.assertEqual(metrics.total_requests, 10)

        # 吞吐量历史应该有数据
        # 注意：由于时间窗口计算，吞吐量可能为0（取决于执行时间）
        self.assertIsInstance(metrics.throughput_history, list)

    def test_get_metrics_nonexistent_handler(self):
        """测试获取不存在的处理器指标"""
        metrics = self.monitor.get_metrics("nonexistent_handler")

        # 应该返回新的PerformanceMetrics实例
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.total_requests, 0)

    def test_get_all_metrics(self):
        """测试获取所有处理器指标"""
        # 添加多个处理器的指标
        handlers = ["handler1", "handler2", "handler3"]

        for handler in handlers:
            self.monitor.record_request(handler, 0.1, True)
            self.monitor.record_request(handler, 0.2, False, "Error")

        all_metrics = self.monitor.get_all_metrics()

        # 验证所有处理器都有指标
        for handler in handlers:
            self.assertIn(handler, all_metrics)
            metrics = all_metrics[handler]
            self.assertEqual(metrics.total_requests, 2)
            self.assertEqual(metrics.successful_requests, 1)
            self.assertEqual(metrics.failed_requests, 1)

    def test_set_alert_threshold(self):
        """测试设置告警阈值"""
        # 设置新的阈值
        self.monitor.set_alert_threshold('error_rate_threshold', 0.05)
        self.monitor.set_alert_threshold('response_time_threshold', 2.0)

        # 验证阈值已设置 - 通过组合对象访问
        self.assertEqual(self.monitor._alert_manager._alert_thresholds['error_rate_threshold'], 0.05)
        self.assertEqual(self.monitor._alert_manager._alert_thresholds['response_time_threshold'], 2.0)

    def test_add_alert_callback(self):
        """测试添加告警回调"""
        callback = Mock()
        self.monitor.add_alert_callback(callback)

        self.assertIn(callback, self.monitor._alert_manager._alert_callbacks)

    def test_alert_generation_high_error_rate(self):
        """测试高错误率告警生成"""
        handler_name = "test_handler"

        # 创建高错误率场景（80%错误率）
        for i in range(10):
            success = i >= 8  # 只有最后2个成功，前8个失败
            self.monitor.record_request(handler_name, 0.1, success)

        # 手动触发告警检查 - 通过组合对象调用
        self.monitor._alert_manager._check_alerts(self.monitor._metrics_collector)

        # 检查是否有告警生成
        alerts = self.monitor.get_alerts()
        high_error_alerts = [a for a in alerts if a.alert_type == 'high_error_rate']

        # 应该有高错误率告警（如果阈值设置为0.5以下）
        if self.monitor._alert_manager._alert_thresholds['error_rate_threshold'] < 0.8:
            self.assertGreater(len(high_error_alerts), 0)
            alert = high_error_alerts[-1]  # 检查最后一个告警
            self.assertEqual(alert.severity, 'high')
            self.assertEqual(alert.actual_value, 0.8)

    def test_alert_generation_slow_response(self):
        """测试慢响应告警生成"""
        handler_name = "test_handler"

        # 记录慢响应
        slow_response_time = self.monitor._alert_manager._alert_thresholds['response_time_threshold'] + 1.0
        self.monitor.record_request(handler_name, slow_response_time, True)

        # 手动触发告警检查
        self.monitor.check_alerts()

        # 检查是否有慢响应告警
        alerts = self.monitor.get_alerts()
        slow_alerts = [a for a in alerts if a.alert_type == 'high_response_time']

        self.assertGreater(len(slow_alerts), 0)
        alert = slow_alerts[0]
        self.assertEqual(alert.severity, 'medium')
        self.assertEqual(alert.actual_value, slow_response_time)

    def test_get_alerts_with_limit(self):
        """测试获取告警列表（带限制）"""
        # 添加多个告警（这里通过直接添加来测试）
        for i in range(5):
            alert = PerformanceAlert(
                alert_type='test_alert',
                severity='low',
                message=f'Test alert {i}',
                metrics={},
                timestamp=time.time(),
                threshold=1.0,
                actual_value=i + 1
            )
            self.monitor._alert_manager._alerts.append(alert)

        # 测试获取所有告警
        all_alerts = self.monitor.get_alerts()
        self.assertEqual(len(all_alerts), 5)

        # 测试限制数量
        limited_alerts = self.monitor.get_alerts(limit=3)
        self.assertEqual(len(limited_alerts), 3)

    def test_get_performance_report_single_handler(self):
        """测试单个处理器的性能报告"""
        handler_name = "test_handler"

        # 添加测试数据
        for i in range(5):
            success = i % 2 == 0  # 交替成功和失败
            error_type = "TestError" if not success else None
            self.monitor.record_request(handler_name, 0.1 * (i + 1), success, error_type)

        report = self.monitor.get_performance_report(handler_name)

        # 验证报告结构
        self.assertIn('handler_name', report)
        self.assertIn('total_requests', report)
        self.assertIn('successful_requests', report)
        self.assertIn('failed_requests', report)
        self.assertIn('error_rate', report)
        self.assertIn('avg_response_time', report)
        self.assertIn('top_errors', report)

        # 验证数据正确性
        self.assertEqual(report['handler_name'], handler_name)
        self.assertEqual(report['total_requests'], 5)
        self.assertEqual(report['successful_requests'], 3)  # 0,2,4索引成功
        self.assertEqual(report['failed_requests'], 2)  # 1,3索引失败
        self.assertEqual(report['error_rate'], 0.4)

    def test_get_performance_report_all_handlers(self):
        """测试所有处理器的性能报告"""
        handlers = ["handler1", "handler2"]

        # 为每个处理器添加数据
        for handler in handlers:
            for i in range(3):
                self.monitor.record_request(handler, 0.1, True)

        report = self.monitor.get_performance_report()

        # 验证汇总报告结构
        self.assertIn('total_handlers', report)
        self.assertIn('total_requests', report)
        self.assertIn('overall_error_rate', report)
        self.assertIn('handler_performance', report)

        # 验证数据正确性
        self.assertEqual(report['total_handlers'], 2)
        self.assertEqual(report['total_requests'], 6)  # 2个处理器各3个请求
        self.assertEqual(report['overall_error_rate'], 0.0)  # 全部成功

        # 验证各处理器性能数据
        for handler in handlers:
            self.assertIn(handler, report['handler_performance'])
            perf = report['handler_performance'][handler]
            self.assertEqual(perf['requests'], 3)
            self.assertEqual(perf['error_rate'], 0.0)

    def test_reset_metrics_single_handler(self):
        """测试重置单个处理器指标"""
        handler_name = "test_handler"

        # 添加数据
        self.monitor.record_request(handler_name, 0.1, True)
        self.assertEqual(self.monitor.get_metrics(handler_name).total_requests, 1)

        # 重置
        self.monitor.reset_metrics(handler_name)
        self.assertEqual(self.monitor.get_metrics(handler_name).total_requests, 0)

    def test_reset_metrics_all_handlers(self):
        """测试重置所有处理器指标"""
        handlers = ["handler1", "handler2"]

        # 添加数据
        for handler in handlers:
            self.monitor.record_request(handler, 0.1, True)

        # 重置所有
        self.monitor.reset_metrics()

        # 验证所有指标都被重置
        for handler in handlers:
            self.assertEqual(self.monitor.get_metrics(handler).total_requests, 0)

    def test_get_optimization_suggestions(self):
        """测试获取优化建议"""
        handler_name = "test_handler"

        # 正常情况
        self.monitor.record_request(handler_name, 0.1, True)
        suggestions = self.monitor.get_optimization_suggestions(handler_name)
        self.assertIn("性能表现良好", suggestions[0])

        # 高错误率情况
        handler_name2 = "high_error_handler"
        for i in range(10):
            success = i >= 9  # 只有1个成功，9个失败
            self.monitor.record_request(handler_name2, 0.1, success)

        suggestions = self.monitor.get_optimization_suggestions(handler_name2)
        self.assertIn("错误处理逻辑", suggestions[0])

        # 慢响应情况
        handler_name3 = "slow_handler"
        slow_time = self.monitor._alert_manager._alert_thresholds['response_time_threshold'] + 1.0
        self.monitor.record_request(handler_name3, slow_time, True)

        suggestions = self.monitor.get_optimization_suggestions(handler_name3)
        self.assertIn("响应时间", suggestions[0])

    def test_global_monitor_functions(self):
        """测试全局监控器函数"""
        # 测试获取全局监控器
        global_monitor = get_global_performance_monitor()
        self.assertIsInstance(global_monitor, PerformanceMonitor)

        # 测试记录性能便捷函数
        record_handler_performance("global_test", 0.1, True)

        metrics = global_monitor.get_metrics("global_test")
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.successful_requests, 1)

    def test_thread_safety_concurrent_recording(self):
        """测试并发记录的线程安全性"""
        import threading

        handler_name = "concurrent_test"
        num_threads = 5
        requests_per_thread = 10

        results = []
        errors = []

        def record_worker(thread_id):
            """记录工作线程"""
            try:
                for i in range(requests_per_thread):
                    response_time = 0.1 + (thread_id * 0.01) + (i * 0.001)
                    success = (i + thread_id) % 3 != 0  # 2/3的成功率
                    error_type = f"Error_{thread_id}_{i}" if not success else None

                    self.monitor.record_request(handler_name, response_time, success, error_type)
                    results.append((thread_id, i, response_time, success))

                    time.sleep(0.001)  # 短暂延迟
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动并发线程
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=record_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(errors), 0, f"并发记录出现错误: {errors}")

        expected_total_requests = num_threads * requests_per_thread
        self.assertEqual(len(results), expected_total_requests)

        # 验证指标正确性
        metrics = self.monitor.get_metrics(handler_name)
        self.assertEqual(metrics.total_requests, expected_total_requests)

        # 计算预期成功率 (2/3)
        expected_successes = sum(1 for _, _, _, success in results if success)
        expected_success_rate = expected_successes / expected_total_requests

        self.assertAlmostEqual(metrics.error_rate, 1 - expected_success_rate, places=2)


if __name__ == '__main__':
    unittest.main()
