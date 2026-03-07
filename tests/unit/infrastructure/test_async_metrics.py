"""
测试异步指标收集器

覆盖 async_metrics.py 中的 AsyncMetricsCollector 类
"""

import pytest
from unittest.mock import AsyncMock
from src.infrastructure.async_metrics import AsyncMetricsCollector


class TestAsyncMetricsCollector:
    """AsyncMetricsCollector 类测试"""

    def test_initialization(self):
        """测试初始化"""
        collector = AsyncMetricsCollector()

        assert collector.metrics == {}
        assert isinstance(collector.metrics, dict)

    @pytest.mark.asyncio
    async def test_collect_metric_existing(self):
        """测试收集存在的指标"""
        collector = AsyncMetricsCollector()
        collector.metrics = {"response_time": 0.125}

        result = await collector.collect_metric("response_time")

        assert result == 0.125

    @pytest.mark.asyncio
    async def test_collect_metric_nonexistent(self):
        """测试收集不存在的指标"""
        collector = AsyncMetricsCollector()

        result = await collector.collect_metric("nonexistent_metric")

        assert result == 0  # 默认值

    @pytest.mark.asyncio
    async def test_record_metric(self):
        """测试记录指标"""
        collector = AsyncMetricsCollector()

        await collector.record_metric("cpu_usage", 85.5)

        assert collector.metrics["cpu_usage"] == 85.5

    @pytest.mark.asyncio
    async def test_record_metric_overwrite(self):
        """测试覆盖指标"""
        collector = AsyncMetricsCollector()

        # 记录初始值
        await collector.record_metric("memory_usage", 1024)
        assert collector.metrics["memory_usage"] == 1024

        # 覆盖值
        await collector.record_metric("memory_usage", 2048)
        assert collector.metrics["memory_usage"] == 2048

    @pytest.mark.asyncio
    async def test_multiple_metrics(self):
        """测试多个指标"""
        collector = AsyncMetricsCollector()

        # 记录多个指标
        await collector.record_metric("requests_per_second", 150.5)
        await collector.record_metric("error_rate", 0.02)
        await collector.record_metric("uptime", 99.9)

        # 验证所有指标
        assert await collector.collect_metric("requests_per_second") == 150.5
        assert await collector.collect_metric("error_rate") == 0.02
        assert await collector.collect_metric("uptime") == 99.9

    @pytest.mark.asyncio
    async def test_metric_types(self):
        """测试不同类型的指标值"""
        collector = AsyncMetricsCollector()

        test_cases = [
            ("counter", 42),
            ("gauge", 85.7),
            ("histogram", [0.1, 0.2, 0.3]),
            ("boolean", True),
            ("string", "healthy"),
        ]

        for name, value in test_cases:
            await collector.record_metric(name, value)
            result = await collector.collect_metric(name)
            assert result == value

    @pytest.mark.asyncio
    async def test_metric_isolation(self):
        """测试指标隔离"""
        collector1 = AsyncMetricsCollector()
        collector2 = AsyncMetricsCollector()

        # 在collector1中记录指标
        await collector1.record_metric("shared_metric", "value1")

        # 在collector2中记录不同的值
        await collector2.record_metric("shared_metric", "value2")

        # 验证隔离性
        assert await collector1.collect_metric("shared_metric") == "value1"
        assert await collector2.collect_metric("shared_metric") == "value2"

    @pytest.mark.asyncio
    async def test_concurrent_metric_operations(self):
        """测试并发指标操作"""
        import asyncio
        collector = AsyncMetricsCollector()

        async def record_and_collect(metric_id, value):
            """并发记录和收集指标"""
            await collector.record_metric(f"metric_{metric_id}", value)
            return await collector.collect_metric(f"metric_{metric_id}")

        # 创建多个并发任务
        tasks = []
        for i in range(20):
            task = asyncio.create_task(record_and_collect(i, f"value_{i}"))
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks)

        # 验证结果
        for i, result in enumerate(results):
            assert result == f"value_{i}"

    @pytest.mark.asyncio
    async def test_empty_metric_name(self):
        """测试空指标名称"""
        collector = AsyncMetricsCollector()

        # 记录空名称的指标
        await collector.record_metric("", "empty_name_value")
        result = await collector.collect_metric("")
        assert result == "empty_name_value"

    @pytest.mark.asyncio
    async def test_special_characters_in_metric_names(self):
        """测试指标名称中的特殊字符"""
        collector = AsyncMetricsCollector()

        special_names = [
            "metric-with-dashes",
            "metric_with_underscores",
            "metric.with.dots",
            "metric with spaces",
            "metric@domain.com"
        ]

        for name in special_names:
            await collector.record_metric(name, f"value_for_{name}")
            result = await collector.collect_metric(name)
            assert result == f"value_for_{name}"

    @pytest.mark.asyncio
    async def test_large_number_of_metrics(self):
        """测试大量指标"""
        collector = AsyncMetricsCollector()

        # 记录1000个指标
        for i in range(1000):
            await collector.record_metric(f"metric_{i}", i * 10)

        # 验证所有指标
        for i in range(1000):
            result = await collector.collect_metric(f"metric_{i}")
            assert result == i * 10

        # 验证内部状态
        assert len(collector.metrics) == 1000

    @pytest.mark.asyncio
    async def test_metric_persistence(self):
        """测试指标持久性"""
        collector = AsyncMetricsCollector()

        # 记录系统指标
        system_metrics = {
            "cpu_percent": 45.2,
            "memory_percent": 67.8,
            "disk_usage": 234567890,
            "network_bytes_sent": 1234567,
            "network_bytes_recv": 2345678,
            "active_connections": 42,
            "request_count": 1500,
            "error_count": 5,
            "response_time_avg": 0.125
        }

        for name, value in system_metrics.items():
            await collector.record_metric(name, value)

        # 验证所有指标都存在且正确
        for name, expected_value in system_metrics.items():
            actual_value = await collector.collect_metric(name)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_zero_and_negative_values(self):
        """测试零值和负值"""
        collector = AsyncMetricsCollector()

        test_values = [0, -1, -100, 0.0, -0.5]

        for i, value in enumerate(test_values):
            metric_name = f"test_metric_{i}"
            await collector.record_metric(metric_name, value)
            result = await collector.collect_metric(metric_name)
            assert result == value

    @pytest.mark.asyncio
    async def test_metric_updates_over_time(self):
        """测试指标随时间更新"""
        collector = AsyncMetricsCollector()

        # 模拟请求计数器
        await collector.record_metric("request_count", 0)

        # 模拟一系列请求
        for i in range(1, 11):
            current_count = await collector.collect_metric("request_count")
            await collector.record_metric("request_count", current_count + 1)

        final_count = await collector.collect_metric("request_count")
        assert final_count == 10

    @pytest.mark.asyncio
    async def test_metric_aggregation_simulation(self):
        """测试指标聚合模拟"""
        collector = AsyncMetricsCollector()

        # 模拟响应时间收集
        response_times = [0.1, 0.2, 0.15, 0.3, 0.05]

        # 记录所有响应时间
        for i, rt in enumerate(response_times):
            await collector.record_metric(f"response_time_{i}", rt)

        # 计算平均值
        total = 0
        for i in range(len(response_times)):
            total += await collector.collect_metric(f"response_time_{i}")

        average = total / len(response_times)
        expected_average = sum(response_times) / len(response_times)

        assert abs(average - expected_average) < 0.001

    @pytest.mark.asyncio
    async def test_async_operations_performance(self):
        """测试异步操作性能"""
        import asyncio
        import time
        collector = AsyncMetricsCollector()

        # 记录开始时间
        start_time = time.time()

        # 执行大量异步操作
        tasks = []
        for i in range(1000):
            task = asyncio.create_task(collector.record_metric(f"perf_metric_{i}", i))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # 验证所有指标都被记录
        for i in range(1000):
            result = await collector.collect_metric(f"perf_metric_{i}")
            assert result == i

        # 验证执行时间合理
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 2.0  # 应该在2秒内完成

    @pytest.mark.asyncio
    async def test_memory_usage_with_many_metrics(self):
        """测试大量指标的内存使用"""
        import sys
        collector = AsyncMetricsCollector()

        # 记录初始内存使用
        initial_memory = sys.getsizeof(collector.metrics)

        # 添加许多指标
        for i in range(1000):
            await collector.record_metric(f"memory_test_{i}", f"value_{i}")

        # 记录添加后的内存使用
        final_memory = sys.getsizeof(collector.metrics)

        # 内存使用应该合理增长
        memory_growth = final_memory - initial_memory
        assert memory_growth > 0  # 应该有内存增长
        assert memory_growth < 1000000  # 但不应该过大