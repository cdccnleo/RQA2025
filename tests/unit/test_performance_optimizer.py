#!/usr/bin/env python3
"""
性能优化器单元测试
测试并发控制、缓存策略和资源管理功能
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import asyncio
import psutil
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.performance_optimizer import (
    PerformanceOptimizer,
    PerformanceConfig,
    PerformanceMetrics,
    ResourceUsage,
    ConcurrentDownloader,
    RateLimiter,
    DataParser,
    SmartCache,
    MemoryManager,
    AdaptiveResourceManager,
    CPUMonitor
)


class TestPerformanceOptimizer:
    """性能优化器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.config = PerformanceConfig(
            max_concurrent_downloads=5,
            max_concurrent_parsing=3,
            batch_size=1000,
            cache_ttl_seconds=600,
            memory_limit_mb=512,
            cpu_limit_percent=70.0,
            network_timeout_seconds=15,
            retry_attempts=2,
            rate_limit_requests_per_minute=30
        )

        self.optimizer = PerformanceOptimizer({'performance': self.config.__dict__})

    def teardown_method(self):
        """测试后清理"""
        pass

    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """测试优化器初始化"""
        await self.optimizer.initialize()

        # 验证组件初始化
        assert self.optimizer.downloader is not None
        assert self.optimizer.parser is not None
        assert self.optimizer.cache is not None
        assert self.optimizer.resource_manager is not None

        # 验证自适应管理器启动
        assert self.optimizer.resource_manager.adjustment_task is not None

        await self.optimizer.cleanup()

    @pytest.mark.asyncio
    async def test_optimize_data_collection(self):
        """测试数据采集优化"""
        # Mock采集函数
        async def mock_collection_func(data_source, max_concurrency):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return [f"record_{i}" for i in range(100)]

        # 执行优化
        start_time = datetime.now()
        result = await self.optimizer.optimize_data_collection(
            mock_collection_func,
            data_source="test",
            max_concurrency=self.optimizer.resource_manager.get_optimal_concurrency()
        )
        end_time = datetime.now()

        # 验证结果
        assert len(result) == 100
        assert len(self.optimizer.metrics_history) == 1

        # 验证性能指标记录
        metrics = self.optimizer.metrics_history[0]
        assert metrics.duration_seconds > 0
        assert metrics.throughput_records_per_second > 0
        assert metrics.start_time == start_time.replace(microsecond=0)
        assert metrics.end_time == end_time.replace(microsecond=0)

    def test_performance_report_generation(self):
        """测试性能报告生成"""
        # 添加一些模拟指标
        mock_metrics = PerformanceMetrics(
            start_time=datetime.now() - timedelta(seconds=10),
            end_time=datetime.now(),
            duration_seconds=10.0,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_data_size_bytes=1024000,
            average_response_time=0.5,
            throughput_records_per_second=50.0
        )

        self.optimizer.metrics_history.append(mock_metrics)

        # 生成报告
        report = self.optimizer.get_performance_report()

        # 验证报告内容
        assert "latest_metrics" in report
        assert "average_metrics" in report
        assert "resource_usage" in report
        assert "cache_stats" in report
        assert report["latest_metrics"]["duration_seconds"] == 10.0
        assert report["latest_metrics"]["throughput_records_per_second"] == 50.0

    def test_concurrent_downloader_initialization(self):
        """测试并发下载器初始化"""
        downloader = ConcurrentDownloader(self.config)

        assert downloader.config == self.config
        assert downloader.semaphore._value == self.config.max_concurrent_downloads
        assert downloader.response_times == []

    @pytest.mark.asyncio
    async def test_concurrent_downloading(self):
        """测试并发下载"""
        downloader = ConcurrentDownloader(self.config)

        # Mock URLs
        urls = [f"http://example.com/data{i}.csv" for i in range(5)]

        # Mock aiohttp session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"test,data,content"
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.close = AsyncMock()

        with patch('aiohttp.ClientSession', return_value=mock_session):
            # 执行下载
            results = await downloader.download_batch(urls)

            # 验证结果
            assert len(results) == 5
            for url, content in results:
                assert url in urls
                assert content == b"test,data,content"

            # 验证性能指标
            assert downloader.metrics.total_requests == 5
            assert downloader.metrics.successful_requests == 5
            assert downloader.metrics.failed_requests == 0

        await downloader.close()

    @pytest.mark.asyncio
    async def test_download_error_handling(self):
        """测试下载错误处理"""
        downloader = ConcurrentDownloader(self.config)

        urls = ["http://example.com/data1.csv", "http://invalid.url/data2.csv"]

        # Mock session with mixed success/failure
        mock_session = AsyncMock()

        def mock_get(url):
            response_mock = AsyncMock()
            if "invalid" in url:
                response_mock.status = 404
                response_mock.read.return_value = b""
            else:
                response_mock.status = 200
                response_mock.read.return_value = b"success,data"
            return response_mock

        mock_session.get.return_value.__aenter__.side_effect = lambda: mock_get(mock_session.get.call_args[0][0])
        mock_session.close = AsyncMock()

        with patch('aiohttp.ClientSession', return_value=mock_session):
            results = await downloader.download_batch(urls)

            # 验证结果处理
            assert len(results) == 2
            success_count = sum(1 for _, content in results if content is not None)
            assert success_count == 1  # 一个成功，一个失败

            # 验证错误统计
            assert downloader.metrics.successful_requests == 1
            assert downloader.metrics.failed_requests == 1

        await downloader.close()

    def test_rate_limiter(self):
        """测试速率限制器"""
        limiter = RateLimiter(requests_per_minute=10)

        # 测试初始状态
        assert limiter.requests_per_minute == 10

        # 测试获取许可
        # 注意：这是一个同步测试，实际使用需要async
        # 这里主要测试初始化和基本逻辑

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """测试速率限制功能"""
        limiter = RateLimiter(requests_per_minute=2)  # 每分钟2个请求

        start_time = datetime.now()

        # 快速连续请求
        for i in range(3):
            await limiter.acquire()

        end_time = datetime.now()

        # 由于速率限制，执行时间应该大于一定值
        duration = (end_time - start_time).total_seconds()
        assert duration >= 1.0  # 至少1秒（因为第3个请求需要等待）

    def test_data_parser_initialization(self):
        """测试数据解析器初始化"""
        parser = DataParser(self.config)

        assert parser.config == self.config
        assert parser.executor is not None
        assert parser._structure_cache == {}

    @pytest.mark.asyncio
    async def test_concurrent_parsing(self):
        """测试并发解析"""
        parser = DataParser(self.config)

        # 模拟解析数据
        raw_data = [
            ("source1", b"symbol,date,close\n000001.SZ,2020-01-01,100.0"),
            ("source2", b"symbol,date,close\n000002.SZ,2020-01-01,200.0"),
            ("source3", b"symbol,date,close\n000003.SZ,2020-01-01,300.0")
        ]

        # Mock解析函数
        def mock_parser(data, **kwargs):
            # 简单CSV解析
            lines = data.decode('utf-8').strip().split('\n')
            headers = lines[0].split(',')
            values = lines[1].split(',')
            return [{
                headers[i]: values[i] for i in range(len(headers))
            }]

        # 执行并发解析
        results = await parser.parse_batch(raw_data, mock_parser)

        # 验证结果
        assert len(results) == 3
        for source, parsed_data in results:
            assert len(parsed_data) == 1
            assert "symbol" in parsed_data[0]
            assert "date" in parsed_data[0]
            assert "close" in parsed_data[0]

        parser.shutdown()

    def test_smart_cache_operations(self):
        """测试智能缓存操作"""
        cache = SmartCache(self.config)

        # 测试缓存设置和获取
        cache.set("test_key", {"data": "value"})
        result = cache.get("test_key")

        assert result == {"data": "value"}

        # 测试缓存过期
        cache._cache_timestamps["test_key"] = datetime.now() - timedelta(seconds=self.config.cache_ttl_seconds + 1)
        result = cache.get("test_key")
        assert result is None  # 应该已过期

    def test_cache_statistics(self):
        """测试缓存统计"""
        cache = SmartCache(self.config)

        # 执行一些缓存操作
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # 命中
        cache.get("key3")  # 缺失

        stats = cache.get_cache_stats()

        assert stats["total_requests"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["cache_size"] == 2

    def test_memory_manager(self):
        """测试内存管理器"""
        memory_mgr = MemoryManager(self.config)

        # 测试内存压力检测
        pressure = memory_mgr.check_memory_pressure()
        assert isinstance(pressure, bool)

        # 测试内存统计
        stats = memory_mgr.get_memory_stats()
        assert "total_mb" in stats
        assert "used_mb" in stats
        assert "free_mb" in stats
        assert "percent" in stats

    def test_cpu_monitor(self):
        """测试CPU监控器"""
        cpu_monitor = CPUMonitor(80.0)

        # 测试CPU使用率获取
        usage = cpu_monitor.get_cpu_usage()
        assert isinstance(usage, float)
        assert 0 <= usage <= 100

        # 测试过载检测
        overloaded = cpu_monitor.is_cpu_overloaded()
        assert isinstance(overloaded, bool)

    def test_adaptive_resource_manager(self):
        """测试自适应资源管理器"""
        resource_mgr = AdaptiveResourceManager(self.config)

        # 测试并发度获取
        concurrency = resource_mgr.get_optimal_concurrency()
        assert isinstance(concurrency, int)
        assert concurrency > 0

        # 测试资源使用情况获取
        usage = resource_mgr.get_resource_usage()
        assert isinstance(usage, ResourceUsage)
        assert hasattr(usage, 'memory_percent')
        assert hasattr(usage, 'cpu_percent')

    @pytest.mark.asyncio
    async def test_adaptive_adjustment(self):
        """测试自适应调整"""
        resource_mgr = AdaptiveResourceManager(self.config)

        await resource_mgr.start_adaptive_management()

        # 等待一段时间让调整逻辑运行
        await asyncio.sleep(0.1)

        # 验证调整任务正在运行
        assert resource_mgr.adjustment_task is not None
        assert not resource_mgr.adjustment_task.done()

        await resource_mgr.stop_adaptive_management()

        # 验证调整任务已停止
        assert resource_mgr.adjustment_task.done()

    @pytest.mark.asyncio
    async def test_resource_usage_monitoring(self):
        """测试资源使用监控"""
        resource_mgr = AdaptiveResourceManager(self.config)

        usage = resource_mgr.get_resource_usage()

        # 验证资源使用数据合理性
        assert 0 <= usage.memory_percent <= 100
        assert 0 <= usage.cpu_percent <= 100
        assert usage.active_threads >= 1  # 至少有主线程
        assert usage.active_coroutines >= 0

    def test_configuration_validation(self):
        """测试配置验证"""
        config = PerformanceConfig()

        # 验证默认配置
        assert config.max_concurrent_downloads > 0
        assert config.max_concurrent_parsing > 0
        assert config.cache_ttl_seconds > 0
        assert config.memory_limit_mb > 0
        assert config.cpu_limit_percent > 0

        # 验证边界条件
        assert config.batch_size > 0
        assert config.network_timeout_seconds > 0
        assert config.retry_attempts >= 0

    @pytest.mark.asyncio
    async def test_error_handling_in_optimization(self):
        """测试优化过程中的错误处理"""
        # 创建一个会抛出异常的采集函数
        async def failing_collection_func(*args, **kwargs):
            raise Exception("Simulated collection error")

        # 执行优化，应该不会崩溃
        with pytest.raises(Exception):
            await self.optimizer.optimize_data_collection(failing_collection_func)

    def test_performance_metrics_calculation(self):
        """测试性能指标计算"""
        metrics = PerformanceMetrics(
            start_time=datetime.now() - timedelta(seconds=10),
            end_time=datetime.now(),
            duration_seconds=10.0,
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
            total_data_size_bytes=1048576,  # 1MB
            average_response_time=0.5,
            throughput_records_per_second=50.0
        )

        # 验证指标计算
        assert metrics.duration_seconds == 10.0
        assert metrics.throughput_records_per_second == 50.0
        assert metrics.total_data_size_bytes == 1048576

        # 验证成功率计算（如果有的话）
        success_rate = metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0
        assert success_rate == 0.9

    def test_cache_cleanup_operations(self):
        """测试缓存清理操作"""
        cache = SmartCache(self.config)

        # 添加一些缓存项
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # 设置一个过期项
        expired_time = datetime.now() - timedelta(seconds=self.config.cache_ttl_seconds + 1)
        cache._cache_timestamps["key1"] = expired_time

        # 执行清理
        cache.clear_expired()

        # 验证过期项被清理
        assert cache.get("key1") is None
        assert cache.get("key2") is not None  # 未过期项保留

    @pytest.mark.asyncio
    async def test_optimizer_cleanup(self):
        """测试优化器清理"""
        await self.optimizer.initialize()

        # 验证组件已初始化
        assert self.optimizer.resource_manager.adjustment_task is not None

        # 执行清理
        await self.optimizer.cleanup()

        # 验证清理完成
        assert self.optimizer.resource_manager.adjustment_task is None or self.optimizer.resource_manager.adjustment_task.done()

    def test_resource_limits_enforcement(self):
        """测试资源限制执行"""
        # 测试内存限制
        memory_mgr = MemoryManager(PerformanceConfig(memory_limit_mb=100))  # 100MB限制

        # 验证配置生效
        assert memory_mgr.memory_threshold == 100

        # 测试CPU限制
        cpu_monitor = CPUMonitor(75.0)  # 75% CPU限制

        # 验证配置生效
        assert cpu_monitor.limit_percent == 75.0

    @pytest.mark.asyncio
    async def test_concurrent_operations_limit(self):
        """测试并发操作限制"""
        downloader = ConcurrentDownloader(PerformanceConfig(max_concurrent_downloads=2))

        # 验证信号量设置
        assert downloader.semaphore._value == 2

        # 创建超过限制的任务
        urls = [f"http://example.com/data{i}.csv" for i in range(5)]

        # Mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"test"
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.close = AsyncMock()

        with patch('aiohttp.ClientSession', return_value=mock_session):
            # 执行下载，应该能处理超过限制的任务（通过队列）
            results = await downloader.download_batch(urls)
            assert len(results) == 5  # 所有任务都应该完成

        await downloader.close()

    def test_performance_config_defaults(self):
        """测试性能配置默认值"""
        config = PerformanceConfig()

        # 验证所有默认值合理
        assert config.max_concurrent_downloads >= 1
        assert config.max_concurrent_parsing >= 1
        assert config.batch_size >= 100
        assert config.cache_ttl_seconds >= 60
        assert config.memory_limit_mb >= 100
        assert 0 < config.cpu_limit_percent <= 100
        assert config.network_timeout_seconds >= 5
        assert config.retry_attempts >= 0
        assert config.rate_limit_requests_per_minute >= 1

    @pytest.mark.asyncio
    async def test_metrics_history_management(self):
        """测试指标历史管理"""
        # 执行多次优化操作
        async def mock_operation(duration):
            await asyncio.sleep(duration)
            return ["result"]

        # 执行几次操作
        for i in range(3):
            await self.optimizer.optimize_data_collection(mock_operation, 0.1)

        # 验证历史记录
        assert len(self.optimizer.metrics_history) == 3

        # 验证指标累积
        total_duration = sum(m.duration_seconds for m in self.optimizer.metrics_history)
        assert total_duration > 0

        # 生成报告验证
        report = self.optimizer.get_performance_report()
        assert report["total_operations"] == 3


if __name__ == '__main__':
    pytest.main([__file__])