#!/usr/bin/env python3
"""
RQA2025 基础设施层性能监控器单元测试

测试性能监控器的功能和正确性。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
from unittest.mock import patch

from src.infrastructure.monitoring.components.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    ComponentPerformanceStats,
    PerformanceContext,
    monitor_performance,
    global_performance_monitor
)


class TestPerformanceMonitor(unittest.TestCase):
    """性能监控器测试类"""

    def setUp(self):
        """测试设置"""
        # 创建不启用自动监控的性能监控器，避免干扰测试
        self.monitor = PerformanceMonitor(enable_auto_monitoring=False)

    def tearDown(self):
        """测试清理"""
        self.monitor.stop_auto_monitoring()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.monitor.metrics_history, list)
        self.assertIsInstance(self.monitor.component_stats, dict)
        self.assertEqual(self.monitor.max_history_size, 10000)
        self.assertFalse(self.monitor.monitoring_active)

    def test_monitor_operation_context_manager(self):
        """测试操作监控上下文管理器"""
        with self.monitor.monitor_operation("TestComponent", "test_operation") as context:
            self.assertIsInstance(context, PerformanceContext)
            # 模拟一些操作 (优化：移除等待)
            pass  # time.sleep removed

        # 检查是否记录了性能指标
        self.assertEqual(len(self.monitor.metrics_history), 1)
        metrics = self.monitor.metrics_history[0]

        self.assertEqual(metrics.component_name, "TestComponent")
        self.assertEqual(metrics.operation_name, "test_operation")
        self.assertTrue(metrics.success)
        self.assertIsNotNone(metrics.duration_ms)
        self.assertGreater(metrics.duration_ms, 0)

    def test_monitor_operation_with_exception(self):
        """测试操作监控异常处理"""
        try:
            with self.monitor.monitor_operation("TestComponent", "failing_operation"):
                raise ValueError("Test exception")
        except ValueError:
            pass  # 预期的异常

        # 检查是否记录了失败的性能指标
        self.assertEqual(len(self.monitor.metrics_history), 1)
        metrics = self.monitor.metrics_history[0]

        self.assertEqual(metrics.component_name, "TestComponent")
        self.assertEqual(metrics.operation_name, "failing_operation")
        self.assertFalse(metrics.success)
        self.assertEqual(metrics.error_message, "Test exception")

    def test_record_metrics_manually(self):
        """测试手动记录性能指标"""
        metrics = PerformanceMetrics(
            component_name="ManualComponent",
            operation_name="manual_operation",
            start_time=time.time()
        )

        # 手动完成指标记录 (优化：移除等待)
        metrics.complete(success=True)

        self.monitor.record_metrics(metrics)

        # 检查统计信息
        stats = self.monitor.get_component_stats("ManualComponent")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.component_name, "ManualComponent")
        self.assertEqual(stats.total_operations, 1)
        self.assertEqual(stats.successful_operations, 1)
        self.assertEqual(stats.failed_operations, 0)

    def test_get_component_stats(self):
        """测试获取组件统计信息"""
        # 添加一些测试数据
        for i in range(3):
            metrics = PerformanceMetrics(
                component_name="StatsTestComponent",
                operation_name=f"operation_{i}",
                start_time=time.time()
            )
            metrics.complete(success=True if i < 2 else False)
            self.monitor.record_metrics(metrics)

        stats = self.monitor.get_component_stats("StatsTestComponent")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.total_operations, 3)
        self.assertEqual(stats.successful_operations, 2)
        self.assertEqual(stats.failed_operations, 1)
        self.assertAlmostEqual(stats.error_rate, 1/3, places=2)

    def test_get_all_component_stats(self):
        """测试获取所有组件统计信息"""
        # 添加不同组件的指标
        components = ["CompA", "CompB", "CompA"]  # CompA有两个操作

        for comp in components:
            metrics = PerformanceMetrics(
                component_name=comp,
                operation_name="test_op",
                start_time=time.time()
            )
            metrics.complete(success=True)
            self.monitor.record_metrics(metrics)

        all_stats = self.monitor.get_all_component_stats()
        self.assertEqual(len(all_stats), 2)  # 应该有两个组件
        self.assertIn("CompA", all_stats)
        self.assertIn("CompB", all_stats)
        self.assertEqual(all_stats["CompA"].total_operations, 2)
        self.assertEqual(all_stats["CompB"].total_operations, 1)

    def test_get_recent_metrics(self):
        """测试获取最近的性能指标"""
        # 添加多个指标
        for i in range(5):
            metrics = PerformanceMetrics(
                component_name="RecentTest",
                operation_name=f"op_{i}",
                start_time=time.time()
            )
            metrics.complete(success=True)
            self.monitor.record_metrics(metrics)

        # 获取最近3个指标
        recent = self.monitor.get_recent_metrics(limit=3)
        self.assertEqual(len(recent), 3)

        # 获取特定组件的指标
        component_recent = self.monitor.get_recent_metrics("RecentTest", limit=2)
        self.assertEqual(len(component_recent), 2)
        for metrics in component_recent:
            self.assertEqual(metrics.component_name, "RecentTest")

    def test_get_performance_summary(self):
        """测试获取性能汇总报告"""
        # 添加一些测试数据
        for comp in ["Comp1", "Comp2"]:
            for i in range(2):
                metrics = PerformanceMetrics(
                    component_name=comp,
                    operation_name=f"op_{i}",
                    start_time=time.time(),
                    duration_ms=100.0 if comp == "Comp1" else 200.0
                )
                metrics.complete(success=True)
                self.monitor.record_metrics(metrics)

        summary = self.monitor.get_performance_summary()

        self.assertIn('total_components', summary)
        self.assertIn('total_operations', summary)
        self.assertIn('components', summary)
        self.assertIn('system_health', summary)

        self.assertEqual(summary['total_components'], 2)
        self.assertEqual(summary['total_operations'], 4)

    def test_detect_performance_anomalies(self):
        """测试性能异常检测"""
        # 添加正常性能数据
        for i in range(5):
            metrics = PerformanceMetrics(
                component_name="NormalComp",
                operation_name=f"op_{i}",
                start_time=time.time(),
                duration_ms=50.0  # 正常响应时间
            )
            metrics.complete(success=True)
            self.monitor.record_metrics(metrics)

        # 添加高错误率组件
        for i in range(10):
            metrics = PerformanceMetrics(
                component_name="HighErrorComp",
                operation_name=f"op_{i}",
                start_time=time.time()
            )
            metrics.complete(success=i < 2)  # 80%错误率
            self.monitor.record_metrics(metrics)

        # 添加慢响应组件
        metrics = PerformanceMetrics(
            component_name="SlowComp",
            operation_name="slow_op",
            start_time=time.time(),
            duration_ms=2000.0  # 非常慢
        )
        metrics.complete(success=True)
        self.monitor.record_metrics(metrics)

        anomalies = self.monitor.detect_performance_anomalies()

        # 应该检测到多个异常
        self.assertGreater(len(anomalies), 0)

        # 检查异常类型
        anomaly_types = {a['type'] for a in anomalies}
        self.assertIn('high_error_rate', anomaly_types)
        self.assertIn('slow_response', anomaly_types)

    def test_generate_performance_recommendations(self):
        """测试生成性能优化建议"""
        # 添加需要优化的组件数据
        # 高错误率组件
        for i in range(20):
            metrics = PerformanceMetrics(
                component_name="ErrorProneComp",
                operation_name=f"op_{i}",
                start_time=time.time()
            )
            metrics.complete(success=i < 5)  # 75%错误率
            self.monitor.record_metrics(metrics)

        # 慢响应组件
        metrics = PerformanceMetrics(
            component_name="SlowResponseComp",
            operation_name="slow_op",
            start_time=time.time(),
            duration_ms=1500.0
        )
        metrics.complete(success=True)
        self.monitor.record_metrics(metrics)

        # 高内存使用组件（模拟）
        metrics = PerformanceMetrics(
            component_name="MemoryHogComp",
            operation_name="memory_op",
            start_time=time.time()
        )
        metrics.memory_usage_mb = 600.0  # 高内存使用
        metrics.complete(success=True)
        self.monitor.record_metrics(metrics)

        recommendations = self.monitor.generate_performance_recommendations()

        # 应该生成多个建议
        self.assertGreater(len(recommendations), 0)

        # 检查建议类型
        recommendation_types = {r['type'] for r in recommendations}
        self.assertIn('error_handling', recommendation_types)
        self.assertIn('performance_optimization', recommendation_types)
        self.assertIn('memory_optimization', recommendation_types)

    @patch('src.infrastructure.monitoring.components.performance_monitor.time.sleep')
    def test_auto_monitoring(self, mock_sleep):
        """测试自动性能监控"""
        monitor = PerformanceMonitor(enable_auto_monitoring=True)

        # 等待一小段时间让自动监控启动 (优化：移除等待)
        pass  # time.sleep removed

        # 应该有自动监控的指标
        self.assertTrue(monitor.monitoring_active)

        # 停止自动监控
        monitor.stop_auto_monitoring()
        self.assertFalse(monitor.monitoring_active)

    def test_monitor_performance_decorator(self):
        """测试性能监控装饰器"""
        @monitor_performance("DecoratorTest", "decorated_function")
        def test_function():
            time.sleep(0.01)
            return "success"

        # 调用被装饰的函数
        result = test_function()
        self.assertEqual(result, "success")

        # 检查是否记录了性能指标
        recent_metrics = global_performance_monitor.get_recent_metrics("DecoratorTest", limit=1)
        self.assertEqual(len(recent_metrics), 1)
        self.assertEqual(recent_metrics[0].operation_name, "decorated_function")
        self.assertTrue(recent_metrics[0].success)


class TestComponentPerformanceStats(unittest.TestCase):
    """组件性能统计测试类"""

    def test_stats_update(self):
        """测试统计信息更新"""
        stats = ComponentPerformanceStats(component_name="TestComp")

        # 添加成功的指标
        metrics1 = PerformanceMetrics(
            component_name="TestComp",
            operation_name="op1",
            start_time=time.time(),
            duration_ms=100.0
        )
        metrics1.complete(success=True)
        stats.update(metrics1)

        self.assertEqual(stats.total_operations, 1)
        self.assertEqual(stats.successful_operations, 1)
        self.assertEqual(stats.failed_operations, 0)
        self.assertEqual(stats.avg_response_time_ms, 100.0)

        # 添加失败的指标
        metrics2 = PerformanceMetrics(
            component_name="TestComp",
            operation_name="op2",
            start_time=time.time(),
            duration_ms=200.0
        )
        metrics2.complete(success=False, error_message="Test error")
        stats.update(metrics2)

        self.assertEqual(stats.total_operations, 2)
        self.assertEqual(stats.successful_operations, 1)
        self.assertEqual(stats.failed_operations, 1)
        self.assertAlmostEqual(stats.avg_response_time_ms, 150.0, places=1)
        self.assertAlmostEqual(stats.error_rate, 0.5, places=1)

    def test_stats_boundaries(self):
        """测试统计信息边界情况"""
        stats = ComponentPerformanceStats(component_name="BoundaryTest")

        # 测试最小/最大响应时间
        metrics_list = [
            PerformanceMetrics("BoundaryTest", "op1", time.time(), duration_ms=50.0),
            PerformanceMetrics("BoundaryTest", "op2", time.time(), duration_ms=200.0),
            PerformanceMetrics("BoundaryTest", "op3", time.time(), duration_ms=100.0),
        ]

        for metrics in metrics_list:
            metrics.complete(success=True)
            stats.update(metrics)

        self.assertEqual(stats.min_response_time_ms, 50.0)
        self.assertEqual(stats.max_response_time_ms, 200.0)
        self.assertAlmostEqual(stats.avg_response_time_ms, 116.67, places=1)


if __name__ == '__main__':
    unittest.main()
