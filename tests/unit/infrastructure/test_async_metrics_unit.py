"""
测试异步指标收集器
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.async_metrics import AsyncMetricsCollector


class TestAsyncMetricsCollector:
    """测试异步指标收集器"""

    def setup_method(self):
        """测试前准备"""
        self.collector = AsyncMetricsCollector()

    def test_async_metrics_collector_init(self):
        """测试异步指标收集器初始化"""
        assert self.collector is not None
        assert hasattr(self.collector, 'metrics')
        assert isinstance(self.collector.metrics, dict)
        assert len(self.collector.metrics) == 0

    @pytest.mark.asyncio
    async def test_collect_metric_existing(self):
        """测试收集存在的指标"""
        # 先记录指标
        await self.collector.record_metric("cpu_usage", 85.5)

        # 收集指标
        value = await self.collector.collect_metric("cpu_usage")
        assert value == 85.5

    @pytest.mark.asyncio
    async def test_collect_metric_nonexistent(self):
        """测试收集不存在的指标"""
        value = await self.collector.collect_metric("nonexistent_metric")
        assert value == 0  # 默认返回值

    @pytest.mark.asyncio
    async def test_record_metric(self):
        """测试记录指标"""
        name = "memory_usage"
        value = 1024.5

        await self.collector.record_metric(name, value)

        # 验证指标已记录
        assert name in self.collector.metrics
        assert self.collector.metrics[name] == value

    @pytest.mark.asyncio
    async def test_record_metric_override(self):
        """测试覆盖记录指标"""
        name = "disk_usage"

        # 第一次记录
        await self.collector.record_metric(name, 500.0)
        assert self.collector.metrics[name] == 500.0

        # 第二次记录，覆盖原有值
        await self.collector.record_metric(name, 750.0)
        assert self.collector.metrics[name] == 750.0

    @pytest.mark.asyncio
    async def test_multiple_metrics(self):
        """测试多个指标"""
        metrics_data = {
            "cpu_percent": 65.5,
            "memory_mb": 2048.0,
            "disk_usage": 85.2,
            "network_rx": 1024.5,
            "network_tx": 512.8
        }

        # 记录多个指标
        for name, value in metrics_data.items():
            await self.collector.record_metric(name, value)

        # 验证所有指标都已记录
        assert len(self.collector.metrics) == len(metrics_data)

        # 逐个验证指标值
        for name, expected_value in metrics_data.items():
            actual_value = await self.collector.collect_metric(name)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_metric_types(self):
        """测试不同类型的指标值"""
        test_metrics = {
            "int_metric": 42,
            "float_metric": 3.14159,
            "string_metric": "active",
            "bool_metric": True,
            "list_metric": [1, 2, 3],
            "dict_metric": {"status": "ok", "count": 5},
            "numpy_metric": np.array([1.0, 2.0, 3.0]),
            "none_metric": None
        }

        # 记录不同类型的指标
        for name, value in test_metrics.items():
            await self.collector.record_metric(name, value)

        # 验证所有类型的指标都能正确存储和检索
        for name, expected_value in test_metrics.items():
            actual_value = await self.collector.collect_metric(name)
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_concurrent_metric_operations(self):
        """测试并发指标操作"""
        async def record_and_collect(index):
            metric_name = f"metric_{index}"
            metric_value = index * 10.5

            # 记录指标
            await self.collector.record_metric(metric_name, metric_value)

            # 收集指标
            collected_value = await self.collector.collect_metric(metric_name)

            return collected_value == metric_value

        # 创建多个并发任务
        tasks = [record_and_collect(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # 验证所有并发操作都成功
        assert all(results)
        assert len(self.collector.metrics) == 20

    @pytest.mark.asyncio
    async def test_metric_updates_over_time(self):
        """测试指标随时间更新"""
        metric_name = "response_time"

        # 模拟一段时间内的指标更新
        response_times = [0.1, 0.15, 0.08, 0.12, 0.09, 0.11]

        collected_values = []

        for response_time in response_times:
            await self.collector.record_metric(metric_name, response_time)
            collected_value = await self.collector.collect_metric(metric_name)
            collected_values.append(collected_value)

        # 验证收集的值与记录的值一致
        assert collected_values == response_times

        # 最终值应该是最后一个记录的值
        final_value = await self.collector.collect_metric(metric_name)
        assert final_value == response_times[-1]

    @pytest.mark.asyncio
    async def test_empty_metric_name_handling(self):
        """测试空指标名称处理"""
        # 记录空名称的指标
        await self.collector.record_metric("", 100.0)

        value = await self.collector.collect_metric("")
        assert value == 100.0

    @pytest.mark.asyncio
    async def test_special_characters_in_metric_names(self):
        """测试指标名称中的特殊字符"""
        special_names = {
            "metric-with-dashes": 10.5,
            "metric_with_underscores": 20.3,
            "metric.with.dots": 15.7,
            "metric with spaces": 8.9,
            "metric@with@symbols": 12.4,
            "metric#hash": 25.0
        }

        # 记录包含特殊字符的指标名称
        for name, value in special_names.items():
            await self.collector.record_metric(name, value)

        # 验证都能正确存储和检索
        for name, expected_value in special_names.items():
            actual_value = await self.collector.collect_metric(name)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_large_number_of_metrics(self):
        """测试大量指标"""
        num_metrics = 1000

        # 创建大量指标
        for i in range(num_metrics):
            metric_name = f"metric_{i:04d}"
            metric_value = i * 1.5
            await self.collector.record_metric(metric_name, metric_value)

        # 验证所有指标都已记录
        assert len(self.collector.metrics) == num_metrics

        # 随机检查一些指标
        for i in [0, 100, 500, 999]:
            metric_name = f"metric_{i:04d}"
            expected_value = i * 1.5
            actual_value = await self.collector.collect_metric(metric_name)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_metric_value_ranges(self):
        """测试指标值范围"""
        # 测试各种数值范围的指标
        test_cases = {
            "zero_value": 0,
            "negative_value": -10.5,
            "positive_value": 100.0,
            "large_value": 1e6,
            "small_value": 1e-6,
            "very_large_value": float('inf'),
            "very_small_value": -float('inf')
        }

        for name, value in test_cases.items():
            await self.collector.record_metric(name, value)
            retrieved_value = await self.collector.collect_metric(name)
            assert retrieved_value == value

    @pytest.mark.asyncio
    async def test_metric_collection_performance(self):
        """测试指标收集性能"""
        # 预先记录一些指标
        for i in range(100):
            await self.collector.record_metric(f"perf_metric_{i}", i * 0.1)

        import time

        # 测试收集性能
        start_time = time.time()
        iterations = 1000

        for _ in range(iterations):
            # 随机收集指标
            metric_name = f"perf_metric_{np.random.randint(0, 100)}"
            await self.collector.collect_metric(metric_name)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能（每秒至少10000次收集）
        operations_per_second = iterations / total_time
        assert operations_per_second > 5000  # 合理的性能阈值


class TestAsyncMetricsCollectorIntegration:
    """测试异步指标收集器集成场景"""

    def setup_method(self):
        """测试前准备"""
        self.collector = AsyncMetricsCollector()

    @pytest.mark.asyncio
    async def test_system_metrics_collection_workflow(self):
        """测试系统指标收集工作流"""
        # 1. 初始化系统指标
        system_metrics = {
            "cpu_usage": 65.5,
            "memory_usage": 78.2,
            "disk_usage": 45.8,
            "network_in": 1024.5,
            "network_out": 512.8
        }

        for name, value in system_metrics.items():
            await self.collector.record_metric(f"system.{name}", value)

        # 2. 收集并验证系统指标
        collected_system_metrics = {}
        for name in system_metrics.keys():
            collected_system_metrics[name] = await self.collector.collect_metric(f"system.{name}")

        assert collected_system_metrics == system_metrics

        # 3. 业务指标收集
        business_metrics = {
            "active_users": 1250,
            "requests_per_second": 45.8,
            "error_rate": 0.02,
            "response_time_avg": 150.5,
            "throughput": 1024.0
        }

        for name, value in business_metrics.items():
            await self.collector.record_metric(f"business.{name}", value)

        # 4. 收集并验证业务指标
        collected_business_metrics = {}
        for name in business_metrics.keys():
            collected_business_metrics[name] = await self.collector.collect_metric(f"business.{name}")

        assert collected_business_metrics == business_metrics

        # 5. 验证总体指标数量
        assert len(self.collector.metrics) == len(system_metrics) + len(business_metrics)

    @pytest.mark.asyncio
    async def test_metrics_aggregation_scenario(self):
        """测试指标聚合场景"""
        # 模拟多个服务实例的指标收集
        services = ["web", "api", "database", "cache", "queue"]
        metrics_types = ["cpu", "memory", "requests", "errors"]

        # 为每个服务和指标类型收集数据
        for service in services:
            for metric_type in metrics_types:
                metric_name = f"{service}.{metric_type}"
                metric_value = np.random.rand() * 100  # 随机值
                await self.collector.record_metric(metric_name, metric_value)

        # 验证所有指标都已收集
        expected_count = len(services) * len(metrics_types)
        assert len(self.collector.metrics) == expected_count

        # 计算每个服务的所有指标平均值
        service_averages = {}
        for service in services:
            service_metrics = [await self.collector.collect_metric(f"{service}.{mt}")
                             for mt in metrics_types]
            service_averages[service] = np.mean(service_metrics)

        # 验证平均值计算正确
        assert len(service_averages) == len(services)
        for service in services:
            assert service in service_averages
            assert 0 <= service_averages[service] <= 100  # 应该在合理范围内

    @pytest.mark.asyncio
    async def test_metrics_monitoring_and_alerting(self):
        """测试指标监控和告警场景"""
        # 设置一些阈值
        thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 90.0,
            "error_rate": 0.05,
            "response_time": 1000.0  # 毫秒
        }

        # 记录正常指标
        normal_metrics = {
            "cpu_usage": 65.5,
            "memory_usage": 78.2,
            "error_rate": 0.02,
            "response_time": 150.5
        }

        for name, value in normal_metrics.items():
            await self.collector.record_metric(name, value)

        # 检查是否有指标超过阈值（应该都没有）
        alerts = []
        for name, threshold in thresholds.items():
            value = await self.collector.collect_metric(name)
            if value > threshold:
                alerts.append(f"{name}: {value} > {threshold}")

        assert len(alerts) == 0  # 不应该有告警

        # 模拟异常情况
        await self.collector.record_metric("cpu_usage", 95.0)  # 超过阈值
        await self.collector.record_metric("memory_usage", 95.0)  # 超过阈值

        # 检查告警
        alerts = []
        for name, threshold in thresholds.items():
            value = await self.collector.collect_metric(name)
            if value > threshold:
                alerts.append(f"{name}: {value} > {threshold}")

        assert len(alerts) == 2  # 应该有两个告警
        assert "cpu_usage" in alerts[0] or "cpu_usage" in alerts[1]
        assert "memory_usage" in alerts[0] or "memory_usage" in alerts[1]

    @pytest.mark.asyncio
    async def test_metrics_export_and_import(self):
        """测试指标导出和导入"""
        # 创建一些测试指标
        test_metrics = {
            "system.cpu": 75.5,
            "system.memory": 82.3,
            "app.requests": 1250,
            "app.errors": 5,
            "db.connections": 45
        }

        for name, value in test_metrics.items():
            await self.collector.record_metric(name, value)

        # 导出指标
        exported_data = dict(self.collector.metrics)

        # 创建新的收集器并导入数据
        new_collector = AsyncMetricsCollector()
        new_collector.metrics = exported_data

        # 验证导入的指标
        for name, expected_value in test_metrics.items():
            actual_value = await new_collector.collect_metric(name)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_metrics_cleanup_and_maintenance(self):
        """测试指标清理和维护"""
        # 添加大量指标
        for i in range(200):
            await self.collector.record_metric(f"temp_metric_{i}", i * 0.5)

        initial_count = len(self.collector.metrics)
        assert initial_count == 200

        # 模拟清理过期的指标（这里简单地删除一部分）
        metrics_to_remove = [f"temp_metric_{i}" for i in range(0, 200, 2)]  # 删除偶数索引的指标

        for metric_name in metrics_to_remove:
            if metric_name in self.collector.metrics:
                del self.collector.metrics[metric_name]

        # 验证清理后的指标数量
        after_cleanup_count = len(self.collector.metrics)
        assert after_cleanup_count == 100  # 应该剩下一半

        # 验证剩余的指标仍然可以访问
        for i in range(1, 200, 2):  # 检查奇数索引的指标
            metric_name = f"temp_metric_{i}"
            expected_value = i * 0.5
            actual_value = await self.collector.collect_metric(metric_name)
            assert actual_value == expected_value
