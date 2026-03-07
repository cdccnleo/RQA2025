#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 PerformanceMonitor综合测试

测试性能监控器的所有功能，包括：
- 操作记录和指标收集
- 系统状态监控
- 性能报告生成
- 瓶颈分析
- 安全监控功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.infrastructure.security.monitoring.performance_monitor import PerformanceMonitor


class TestPerformanceMonitorComprehensive:
    """PerformanceMonitor综合测试"""

    @pytest.fixture
    def performance_monitor(self):
        """创建PerformanceMonitor实例"""
        return PerformanceMonitor(enabled=True, collection_interval=1)

    def test_initialization(self, performance_monitor):
        """测试初始化"""
        assert performance_monitor.enabled == True
        assert performance_monitor.collection_interval == 1
        assert isinstance(performance_monitor._metrics, dict)
        assert isinstance(performance_monitor.user_activity, dict)
        assert isinstance(performance_monitor.resource_access, dict)

    def test_record_operation_basic(self, performance_monitor):
        """测试基本操作记录"""
        performance_monitor.record_operation("test_operation", 0.1)

        metrics = performance_monitor.get_metrics("test_operation")
        assert "operation_name" in metrics
        assert "total_calls" in metrics
        assert metrics["total_calls"] == 1
        assert metrics["avg_time"] == 0.1

    def test_record_operation_with_user_and_resource(self, performance_monitor):
        """测试带用户和资源信息的操作记录"""
        performance_monitor.record_operation(
            "login",
            0.05,
            user_id="user123",
            resource="/api/auth"
        )

        # 检查用户活动记录
        assert "user123" in performance_monitor.user_activity
        assert "login" in performance_monitor.user_activity["user123"]

        # 检查资源访问记录
        assert "/api/auth" in performance_monitor.resource_access
        assert "login" in performance_monitor.resource_access["/api/auth"]

    def test_record_operation_with_error(self, performance_monitor):
        """测试记录错误操作"""
        performance_monitor.record_operation("failing_operation", 0.2, is_error=True)

        metrics = performance_monitor.get_metrics("failing_operation")
        assert metrics["error_count"] == 1
        assert metrics["error_rate"] > 0

    def test_get_metrics_nonexistent_operation(self, performance_monitor):
        """测试获取不存在操作的指标"""
        metrics = performance_monitor.get_metrics("nonexistent")
        assert metrics == {}

    def test_get_metrics_all_operations(self, performance_monitor):
        """测试获取所有操作的指标"""
        performance_monitor.record_operation("op1", 0.1)
        performance_monitor.record_operation("op2", 0.2)

        all_metrics = performance_monitor.get_metrics()
        assert isinstance(all_metrics, dict)
        assert "op1" in all_metrics
        assert "op2" in all_metrics

    def test_get_system_stats(self, performance_monitor):
        """测试获取系统统计信息"""
        stats = performance_monitor.get_system_stats()
        assert isinstance(stats, dict)

        # 检查基本字段
        expected_fields = ["cpu_percent", "memory_percent", "disk_usage", "timestamp"]
        for field in expected_fields:
            assert field in stats or any(field in key for key in stats.keys())

    def test_get_performance_report(self, performance_monitor):
        """测试获取性能报告"""
        # 记录一些操作
        for i in range(5):
            performance_monitor.record_operation("test_op", 0.1 * (i + 1))

        report = performance_monitor.get_performance_report()
        assert isinstance(report, dict)
        assert "operations" in report
        assert "system_stats" in report
        assert "summary" in report

    def test_reset_metrics_single_operation(self, performance_monitor):
        """测试重置单个操作的指标"""
        performance_monitor.record_operation("reset_test", 0.1)

        # 确认指标存在
        metrics = performance_monitor.get_metrics("reset_test")
        assert metrics["total_calls"] == 1

        # 重置指标
        performance_monitor.reset_metrics("reset_test")

        # 确认指标被重置
        metrics = performance_monitor.get_metrics("reset_test")
        assert metrics == {} or metrics["total_calls"] == 0

    def test_reset_metrics_all_operations(self, performance_monitor):
        """测试重置所有操作的指标"""
        performance_monitor.record_operation("op1", 0.1)
        performance_monitor.record_operation("op2", 0.2)

        # 重置所有指标
        performance_monitor.reset_metrics()

        # 确认所有指标都被重置
        all_metrics = performance_monitor.get_metrics()
        assert len(all_metrics) == 0 or all(
            metrics.get("total_calls", 0) == 0 for metrics in all_metrics.values()
        )

    def test_disabled_monitor(self):
        """测试禁用的监控器"""
        monitor = PerformanceMonitor(enabled=False)
        monitor.record_operation("disabled_test", 0.1)

        metrics = monitor.get_metrics("disabled_test")
        assert metrics == {}  # 不应该记录任何指标

    def test_multiple_calls_statistics(self, performance_monitor):
        """测试多次调用后的统计信息"""
        # 记录多次调用
        times = [0.1, 0.2, 0.15, 0.3, 0.05]
        for t in times:
            performance_monitor.record_operation("multi_call_test", t)

        metrics = performance_monitor.get_metrics("multi_call_test")

        assert metrics["total_calls"] == 5
        assert abs(metrics["avg_time"] - 0.16) < 0.01  # (0.1+0.2+0.15+0.3+0.05)/5
        assert metrics["min_time"] == 0.05
        assert metrics["max_time"] == 0.3

    def test_error_rate_calculation(self, performance_monitor):
        """测试错误率计算"""
        # 记录一些成功和失败的操作
        for i in range(10):
            is_error = i % 3 == 0  # 每3个操作中有1个失败
            performance_monitor.record_operation("error_test", 0.1, is_error=is_error)

        metrics = performance_monitor.get_metrics("error_test")
        expected_errors = 10 // 3 + (1 if 10 % 3 > 0 else 0)  # 4个错误

        assert metrics["error_count"] == expected_errors
        assert abs(metrics["error_rate"] - (expected_errors / 10 * 100)) < 1.0  # 允许1%的误差

    def test_p95_p99_percentiles(self, performance_monitor):
        """测试P95和P99百分位数计算"""
        # 记录足够的数据来计算百分位数
        import random
        random.seed(42)  # 确保可重复性

        for _ in range(100):
            duration = random.uniform(0.01, 1.0)
            performance_monitor.record_operation("percentile_test", duration)

        metrics = performance_monitor.get_metrics("percentile_test")

        # P95和P99应该在合理范围内
        assert 0.01 <= metrics["p95_time"] <= 1.0
        assert 0.01 <= metrics["p99_time"] <= 1.0
        assert metrics["p95_time"] <= metrics["p99_time"]  # P95应该小于等于P99

    def test_user_activity_tracking(self, performance_monitor):
        """测试用户活动跟踪"""
        # 记录不同用户的活动
        users = ["alice", "bob", "charlie"]
        operations = ["login", "logout", "data_access"]

        for user in users:
            for op in operations:
                performance_monitor.record_operation(op, 0.1, user_id=user)

        # 验证用户活动记录
        for user in users:
            assert user in performance_monitor.user_activity
            user_ops = performance_monitor.user_activity[user]
            for op in operations:
                assert op in user_ops
                assert len(user_ops[op]) > 0

    def test_resource_access_tracking(self, performance_monitor):
        """测试资源访问跟踪"""
        # 记录对不同资源的访问
        resources = ["/api/users", "/api/data", "/api/admin"]
        operations = ["GET", "POST", "PUT"]

        for resource in resources:
            for op in operations:
                performance_monitor.record_operation(op, 0.1, resource=resource)

        # 验证资源访问记录
        for resource in resources:
            assert resource in performance_monitor.resource_access
            resource_ops = performance_monitor.resource_access[resource]
            for op in operations:
                assert op in resource_ops
                assert len(resource_ops[op]) > 0

    def test_concurrent_access_simulation(self, performance_monitor):
        """测试并发访问模拟"""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                # 模拟各种操作记录
                for i in range(10):
                    op_name = f"worker_{worker_id}_op_{i}"
                    performance_monitor.record_operation(op_name, 0.01 * (i + 1))

                results.put(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.put(f"worker_{worker_id}: {e}")

        # 启动多个线程
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=5)

        # 验证结果
        assert errors.empty()

        result_count = 0
        while not results.empty():
            results.get()
            result_count += 1

        assert result_count == num_threads

        # 验证指标被正确记录
        all_metrics = performance_monitor.get_metrics()
        worker_ops = [key for key in all_metrics.keys() if key.startswith("worker_")]
        assert len(worker_ops) == num_threads * 10

    def test_large_scale_performance_monitoring(self, performance_monitor):
        """测试大规模性能监控"""
        # 记录大量操作
        num_operations = 1000
        operations = ["fast_op", "medium_op", "slow_op"]

        for i in range(num_operations):
            op = operations[i % len(operations)]
            base_time = {"fast_op": 0.01, "medium_op": 0.1, "slow_op": 1.0}[op]
            duration = base_time * (0.5 + i / num_operations)  # 一些变化
            performance_monitor.record_operation(op, duration)

        # 验证统计信息
        for i, op in enumerate(operations):
            metrics = performance_monitor.get_metrics(op)
            expected_calls = num_operations // len(operations) + (1 if i < num_operations % len(operations) else 0)
            assert metrics["total_calls"] == expected_calls
            assert metrics["avg_time"] > 0
            assert metrics["min_time"] > 0
            assert metrics["max_time"] > 0

    def test_shutdown_method(self, performance_monitor):
        """测试shutdown方法"""
        # 确认初始状态
        assert performance_monitor.enabled == True

        # 调用shutdown
        performance_monitor.shutdown()

        # 确认状态改变
        assert performance_monitor.enabled == False

        # 确认shutdown后不再记录操作
        performance_monitor.record_operation("after_shutdown", 0.1)
        metrics = performance_monitor.get_metrics("after_shutdown")
        assert metrics == {}

    def test_memory_efficiency(self, performance_monitor):
        """测试内存效率"""
        # 记录大量操作
        for i in range(1000):
            performance_monitor.record_operation(f"mem_test_op_{i % 10}", 0.01)

        # 验证内存使用合理（通过检查对象数量）
        all_metrics = performance_monitor.get_metrics()
        assert len(all_metrics) == 10  # 应该只有10个不同的操作

        # 验证每个操作的调用次数
        for op_name, metrics in all_metrics.items():
            assert metrics["total_calls"] == 100  # 每个操作被调用100次

    def test_performance_report_comprehensive(self, performance_monitor):
        """测试综合性能报告"""
        # 创建多样化的操作数据
        operations_data = {
            "fast_api": {"count": 100, "avg_time": 0.01},
            "slow_db": {"count": 10, "avg_time": 2.0},
            "medium_calc": {"count": 50, "avg_time": 0.5}
        }

        for op, data in operations_data.items():
            for _ in range(data["count"]):
                performance_monitor.record_operation(op, data["avg_time"])

        report = performance_monitor.get_performance_report()

        # 验证报告结构
        assert "operations" in report
        assert "system_stats" in report
        assert "summary" in report
        assert "timestamp" in report

        # 验证操作数据
        operations = report["operations"]
        for op in operations_data.keys():
            assert op in operations
            assert operations[op]["total_calls"] == operations_data[op]["count"]

    def test_security_operation_tracking(self, performance_monitor):
        """测试安全操作跟踪"""
        security_operations = [
            ("user_login", "user123", "/auth/login"),
            ("password_change", "user456", "/auth/password"),
            ("permission_check", "user789", "/api/admin"),
            ("audit_log_access", "admin", "/audit/logs")
        ]

        for op, user, resource in security_operations:
            performance_monitor.record_operation(op, 0.1, user_id=user, resource=resource)

        # 验证所有安全操作都被跟踪
        for op, user, resource in security_operations:
            assert user in performance_monitor.user_activity
            assert resource in performance_monitor.resource_access
            assert op in performance_monitor.user_activity[user]
            assert op in performance_monitor.resource_access[resource]

    def test_time_based_analysis(self, performance_monitor):
        """测试基于时间的分析"""
        import time

        # 记录不同时间段的操作
        start_time = time.time()

        # 第一批操作
        for i in range(10):
            performance_monitor.record_operation("time_test", 0.1)

        mid_time = time.time()

        # 第二批操作
        for i in range(10):
            performance_monitor.record_operation("time_test", 0.2)

        end_time = time.time()

        # 验证时间戳记录
        metrics = performance_monitor.get_metrics("time_test")
        assert metrics["last_call_time"] is not None
        assert isinstance(metrics["last_call_time"], str)  # to_dict方法返回ISO字符串

        # 验证时间戳可以解析为datetime并在合理范围内
        from datetime import datetime
        last_call_dt = datetime.fromisoformat(metrics["last_call_time"])
        assert start_time <= last_call_dt.timestamp() <= end_time

    def test_bottleneck_detection_simulation(self, performance_monitor):
        """测试瓶颈检测模拟"""
        # 创建一个明显慢的操作
        for i in range(20):
            duration = 0.01 if i < 15 else 2.0  # 大部分快，最后5个慢
            performance_monitor.record_operation("bottleneck_test", duration)

        metrics = performance_monitor.get_metrics("bottleneck_test")

        # 验证统计信息反映了瓶颈
        assert metrics["max_time"] >= 2.0  # 应该有慢操作
        assert metrics["avg_time"] < 1.0   # 平均应该受慢操作影响
        assert metrics["p95_time"] >= 1.0  # P95应该捕捉到慢操作

    def test_monitor_decorator_integration(self):
        """测试监控装饰器的集成"""
        from src.infrastructure.security.monitoring.performance_monitor import monitor_performance

        monitor = PerformanceMonitor()

        @monitor_performance("decorated_function", monitor)
        def test_function():
            time.sleep(0.01)  # 模拟一些工作
            return "result"

        # 调用被装饰的函数
        result = test_function()
        assert result == "result"

        # 验证性能被记录
        metrics = monitor.get_metrics("decorated_function")
        assert metrics["total_calls"] == 1
        assert metrics["avg_time"] >= 0.01

    def test_exception_handling_in_monitoring(self, performance_monitor):
        """测试监控中的异常处理"""
        # 测试无效输入
        try:
            performance_monitor.record_operation("", -0.1)  # 无效输入
        except Exception:
            # 应该优雅处理异常
            pass

        # 测试None输入
        try:
            performance_monitor.record_operation(None, 0.1)
        except Exception:
            pass

        # 验证监控器仍然正常工作
        performance_monitor.record_operation("after_exception", 0.1)
        metrics = performance_monitor.get_metrics("after_exception")
        assert metrics["total_calls"] == 1

    def test_performance_trends_analysis(self, performance_monitor):
        """测试性能趋势分析"""
        # 模拟性能变化趋势
        base_time = 0.1
        for i in range(50):
            # 逐渐增加响应时间（模拟性能下降）
            duration = base_time * (1 + i * 0.01)
            performance_monitor.record_operation("trend_test", duration)

        metrics = performance_monitor.get_metrics("trend_test")

        # 验证趋势反映在统计信息中
        assert metrics["total_calls"] == 50
        # 最后的操作应该比第一个慢
        assert metrics["max_time"] > metrics["min_time"]

        # 验证P95反映了趋势
        assert metrics["p95_time"] > base_time

    def test_analyze_bottlenecks(self, performance_monitor):
        """测试瓶颈分析功能"""
        # 创建一些测试指标数据
        metrics_data = {
            'operation1': {
                'avg_time': 2.0,  # 慢操作
                'error_rate': 0.1,  # 高错误率
                'total_calls': 100,
                'p95_time': 3.0
            },
            'operation2': {
                'avg_time': 0.05,  # 快操作
                'error_rate': 0.01,  # 低错误率
                'total_calls': 1000,
                'p95_time': 0.1
            },
            'operation3': {
                'avg_time': 0.5,  # 中等操作
                'error_rate': 0.05,  # 中等错误率
                'total_calls': 50,
                'p95_time': 1.0
            }
        }

        bottlenecks = performance_monitor._analyze_bottlenecks(metrics_data)

        # 验证返回的是列表
        assert isinstance(bottlenecks, list)

        # 应该识别出operation1作为瓶颈（慢且高错误率）
        bottleneck_names = [b.get('operation', b.get('name', '')) for b in bottlenecks]
        assert 'operation1' in bottleneck_names

    def test_generate_recommendations(self, performance_monitor):
        """测试生成性能建议"""
        # 创建需要优化的指标数据
        metrics_data = {
            'slow_operation': {
                'avg_time': 5.0,  # 很慢
                'error_rate': 0.15,  # 高错误率
                'total_calls': 100,
                'p95_time': 10.0
            },
            'high_error_operation': {
                'avg_time': 0.1,
                'error_rate': 0.25,  # 非常高的错误率
                'total_calls': 200,
                'p95_time': 0.5
            }
        }

        system_stats = {'cpu_percent': 90, 'memory_percent': 95}
        recommendations = performance_monitor._generate_recommendations(metrics_data, system_stats)

        # 验证返回的是列表
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # 验证建议是字符串列表
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_generate_summary(self, performance_monitor):
        """测试生成性能摘要"""
        # 创建测试指标数据
        metrics_data = {
            'api_call': {'avg_time': 0.2, 'total_calls': 1000, 'error_rate': 0.02, 'error_count': 20},
            'db_query': {'avg_time': 0.05, 'total_calls': 5000, 'error_rate': 0.01, 'error_count': 50},
            'file_upload': {'avg_time': 2.0, 'total_calls': 100, 'error_rate': 0.1, 'error_count': 10},
        }

        summary = performance_monitor._generate_summary(metrics_data)

        # 验证摘要结构
        assert isinstance(summary, dict)
        assert 'total_operations' in summary
        assert 'total_calls' in summary
        assert 'avg_response_time' in summary
        assert 'overall_error_rate' in summary
        assert 'performance_score' in summary

        # 验证数值计算
        assert summary['total_operations'] == 3
        assert summary['total_calls'] == 6100  # 1000 + 5000 + 100
        assert summary['avg_response_time'] > 0
        assert summary['overall_error_rate'] > 0
        assert 0 <= summary['performance_score'] <= 100

    def test_start_background_collection(self, performance_monitor):
        """测试后台统计收集启动"""
        # 这个方法主要启动后台线程，难以直接测试
        # 我们可以通过检查是否创建了相关属性来验证

        # 调用方法（它在__init__中已经被调用了）
        performance_monitor._start_background_collection()

        # 验证方法没有抛出异常（基本的健全性检查）
        assert performance_monitor.enabled == True

    @patch('src.infrastructure.security.monitoring.performance_monitor.time.sleep')
    def test_start_background_collection_with_exception(self, mock_sleep, performance_monitor):
        """测试后台统计收集中的异常处理"""
        # Mock get_system_stats使其抛出异常
        with patch.object(performance_monitor, 'get_system_stats', side_effect=Exception("Test exception")):
            # 调用后台收集方法
            performance_monitor._start_background_collection()

            # 短暂等待以允许后台线程执行
            time.sleep(0.1)

            # 验证time.sleep被调用了（表示异常处理生效）
            mock_sleep.assert_called()

    def test_disabled_monitor_does_not_start_collection(self):
        """测试禁用的监控器不启动后台收集"""
        monitor = PerformanceMonitor(enabled=False)

        # 验证没有启动后台线程
        assert monitor.enabled == False

        # 记录操作应该被忽略
        monitor.record_operation("test", 0.1)
        metrics = monitor.get_metrics("test")
        assert metrics == {}

    def test_system_stats_collection_integration(self, performance_monitor):
        """测试系统统计收集的集成"""
        import time

        # 等待一些统计数据被收集（后台线程运行）
        time.sleep(2)

        stats = performance_monitor.get_system_stats()

        # 验证系统统计包含预期字段
        expected_fields = ['cpu_percent', 'memory_usage', 'disk_usage']
        for field in expected_fields:
            assert any(field in key or field.replace('_', ' ') in key.lower()
                      for key in stats.keys()), f"Missing expected field similar to {field}"

    def test_performance_monitor_with_custom_collection_interval(self):
        """测试自定义收集间隔的性能监控器"""
        monitor = PerformanceMonitor(enabled=True, collection_interval=30)

        assert monitor.collection_interval == 30
        assert monitor.enabled == True

        # 验证可以正常工作
        monitor.record_operation("test", 0.1)
        metrics = monitor.get_metrics("test")
        assert metrics["total_calls"] == 1

    def test_concurrent_metrics_updates(self, performance_monitor):
        """测试并发指标更新"""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                # 每个线程执行多次操作记录
                for i in range(20):
                    operation_name = f"worker_{worker_id}_op_{i % 5}"
                    duration = 0.01 * (i + 1)
                    performance_monitor.record_operation(operation_name, duration)

                results.put(f"worker_{worker_id}_completed")
            except Exception as e:
                errors.put(f"worker_{worker_id}: {e}")

        # 启动多个线程
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=5)

        # 验证结果
        assert errors.empty()
        result_count = 0
        while not results.empty():
            results.get()
            result_count += 1

        assert result_count == num_threads

        # 验证指标被正确更新
        all_metrics = performance_monitor.get_metrics()
        # 每个线程有5个不同的操作
        assert len(all_metrics) >= 5

    def test_performance_monitor_memory_cleanup(self, performance_monitor):
        """测试性能监控器的内存清理"""
        # 记录大量操作
        for i in range(1000):
            operation_name = f"cleanup_test_op_{i % 20}"  # 重复使用20个操作名
            performance_monitor.record_operation(operation_name, 0.01)

        # 验证内存使用在合理范围内
        all_metrics = performance_monitor.get_metrics()
        assert len(all_metrics) == 20  # 应该只有20个不同的操作

        # 验证每个操作都有合理的调用次数
        for op_name, metrics in all_metrics.items():
            assert metrics['total_calls'] == 50  # 1000 / 20 = 50

    def test_performance_monitor_error_handling_in_background_collection(self):
        """测试后台收集中的错误处理"""
        monitor = PerformanceMonitor(enabled=True, collection_interval=1)

        # 模拟系统统计收集错误
        with patch('psutil.cpu_percent', side_effect=Exception("Mock CPU error")):
            with patch('psutil.virtual_memory', side_effect=Exception("Mock memory error")):
                # 等待后台收集尝试运行
                import time
                time.sleep(2)

                # 监控器应该仍然正常工作
                monitor.record_operation("test_after_error", 0.1)
                metrics = monitor.get_metrics("test_after_error")
                assert metrics["total_calls"] == 1

    def test_performance_report_comprehensive_analysis(self, performance_monitor):
        """测试性能报告的综合分析"""
        # 创建多样化的性能数据
        operations = [
            ("fast_api", 0.05, False),
            ("slow_db", 2.0, False),
            ("error_prone", 0.5, True),  # 有错误
            ("normal_op", 0.2, False),
        ]

        for op_name, duration, is_error in operations:
            performance_monitor.record_operation(op_name, duration, is_error)

        report = performance_monitor.get_performance_report()

        # 验证报告的全面性
        assert 'operations' in report
        assert 'system_stats' in report
        assert 'bottlenecks' in report
        assert 'recommendations' in report
        assert 'summary' in report

        operations_report = report['operations']
        bottlenecks = report['bottlenecks']
        recommendations = report['recommendations']

        # 验证操作数据
        assert len(operations_report) == 4

        # 验证瓶颈检测
        assert isinstance(bottlenecks, list)
        # slow_db应该被识别为瓶颈
        bottleneck_ops = [b.get('operation', '') for b in bottlenecks]
        assert any('slow_db' in op or 'slow' in op.lower() for op in bottleneck_ops)

        # 验证建议生成
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_user_activity_tracking_detailed(self, performance_monitor):
        """测试用户活动跟踪的详细信息"""
        # 记录多个用户的多种操作
        users = ['alice', 'bob', 'charlie']
        operations = ['login', 'logout', 'data_access', 'file_upload']

        for user in users:
            for op in operations:
                duration = 0.1 + hash(user + op) % 10 * 0.1  # 不同的持续时间
                performance_monitor.record_operation(op, duration, user_id=user)

        # 验证用户活动记录的详细程度
        for user in users:
            assert user in performance_monitor.user_activity
            user_ops = performance_monitor.user_activity[user]

            for op in operations:
                assert op in user_ops
                assert isinstance(user_ops[op], list)
                assert len(user_ops[op]) == 1  # 每个操作记录一次

                # 验证记录的内容
                record = user_ops[op][0]
                assert 'timestamp' in record
                assert 'duration' in record
                assert record['duration'] > 0

    def test_resource_access_tracking_detailed(self, performance_monitor):
        """测试资源访问跟踪的详细信息"""
        # 记录对多个资源的访问
        resources = ['/api/users', '/api/data', '/files/upload', '/admin/config']
        operations = ['GET', 'POST', 'PUT', 'DELETE']

        for resource in resources:
            for op in operations:
                duration = 0.05 + hash(resource + op) % 20 * 0.01
                performance_monitor.record_operation(op, duration, resource=resource)

        # 验证资源访问记录的详细程度
        for resource in resources:
            assert resource in performance_monitor.resource_access
            resource_ops = performance_monitor.resource_access[resource]

            for op in operations:
                assert op in resource_ops
                assert isinstance(resource_ops[op], list)
                assert len(resource_ops[op]) == 1

                # 验证记录的内容
                record = resource_ops[op][0]
                assert 'timestamp' in record
                assert 'duration' in record
                assert record['duration'] > 0

    def test_performance_monitor_config_validation(self):
        """测试性能监控器配置验证"""
        # 测试有效的配置
        monitor1 = PerformanceMonitor(enabled=True, collection_interval=60)
        assert monitor1.enabled == True
        assert monitor1.collection_interval == 60

        # 测试边界值
        monitor2 = PerformanceMonitor(enabled=False, collection_interval=1)
        assert monitor2.enabled == False
        assert monitor2.collection_interval == 1

        # 测试大值
        monitor3 = PerformanceMonitor(enabled=True, collection_interval=3600)
        assert monitor3.collection_interval == 3600

    def test_performance_monitor_statistics_accuracy(self, performance_monitor):
        """测试性能监控器统计准确性"""
        # 记录一系列已知的数据
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        operation_name = "accuracy_test"

        for duration in durations:
            performance_monitor.record_operation(operation_name, duration)

        metrics = performance_monitor.get_metrics(operation_name)

        # 验证统计准确性
        assert metrics['total_calls'] == 5
        assert abs(metrics['avg_time'] - 0.3) < 0.001  # (0.1+0.2+0.3+0.4+0.5)/5 = 0.3
        assert metrics['min_time'] == 0.1
        assert metrics['max_time'] == 0.5

        # 验证百分位数（对于少量数据，百分位数可能等于最大值）
        assert metrics['p95_time'] >= 0.0
        assert metrics['p99_time'] >= 0.0
        assert metrics['p95_time'] <= metrics['max_time']
        assert metrics['p99_time'] <= metrics['max_time']

    def test_performance_monitor_large_dataset_handling(self, performance_monitor):
        """测试性能监控器的大数据集处理"""
        # 模拟大规模监控场景
        num_operations = 100
        calls_per_operation = 50

        for i in range(num_operations):
            for j in range(calls_per_operation):
                op_name = f"large_test_op_{i}"
                duration = 0.01 + (j % 10) * 0.01  # 0.01 到 0.1 之间变化
                performance_monitor.record_operation(op_name, duration)

        # 验证可以处理大量数据
        all_metrics = performance_monitor.get_metrics()
        assert len(all_metrics) == num_operations

        # 验证每个操作的统计
        for op_name in [f"large_test_op_{i}" for i in range(min(10, num_operations))]:
            metrics = all_metrics[op_name]
            assert metrics['total_calls'] == calls_per_operation
            assert metrics['avg_time'] > 0

    def test_performance_monitor_shutdown_cleanup(self, performance_monitor):
        """测试性能监控器关闭时的清理"""
        # 记录一些数据
        performance_monitor.record_operation("cleanup_test", 0.1)

        # 验证数据存在
        metrics_before = performance_monitor.get_metrics("cleanup_test")
        assert metrics_before['total_calls'] == 1

        # 关闭监控器
        performance_monitor.shutdown()

        # 验证状态改变
        assert performance_monitor.enabled == False

        # 验证关闭后不再记录新数据
        performance_monitor.record_operation("after_shutdown", 0.1)
        metrics_after = performance_monitor.get_metrics("after_shutdown")
        assert metrics_after == {}  # 不应该有新记录
