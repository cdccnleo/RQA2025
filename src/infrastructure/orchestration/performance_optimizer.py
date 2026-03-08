#!/usr/bin/env python3
"""
性能优化模块
提供并发采集机制、数据缓存策略和内存优化
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
import threading
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """性能配置"""
    max_concurrent_downloads: int = 10      # 最大并发下载数
    max_concurrent_parsing: int = 5         # 最大并发解析数
    batch_size: int = 1000                  # 批处理大小
    cache_ttl_seconds: int = 3600           # 缓存过期时间
    memory_limit_mb: int = 1024             # 内存限制
    cpu_limit_percent: float = 80.0         # CPU使用率限制
    network_timeout_seconds: int = 30       # 网络超时
    retry_attempts: int = 3                 # 重试次数
    rate_limit_requests_per_minute: int = 60  # 速率限制


@dataclass
class PerformanceMetrics:
    """性能指标"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_data_size_bytes: int = 0
    average_response_time: float = 0.0
    peak_memory_usage_mb: float = 0.0
    peak_cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_records_per_second: float = 0.0


@dataclass
class ResourceUsage:
    """资源使用情况"""
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_threads: int = 0
    active_coroutines: int = 0


class ConcurrentDownloader:
    """并发下载器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 并发控制
        self.semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        self.session = None

        # 性能监控
        self.metrics = PerformanceMetrics()
        self.response_times = []

        # 速率限制
        self.rate_limiter = RateLimiter(config.rate_limit_requests_per_minute)

    async def download_batch(self, urls: List[str], headers: Dict[str, str] = None) -> List[Tuple[str, Optional[bytes]]]:
        """
        并发下载URL列表

        Args:
            urls: URL列表
            headers: 请求头

        Returns:
            (url, content)元组列表
        """
        if not self.session:
            import aiohttp
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.network_timeout_seconds)
            )

        self.metrics.total_requests = len(urls)

        async def download_single(url: str) -> Tuple[str, Optional[bytes]]:
            async with self.semaphore:
                # 速率限制
                await self.rate_limiter.acquire()

                start_time = datetime.now()
                content = None

                for attempt in range(self.config.retry_attempts):
                    try:
                        async with self.session.get(url, headers=headers) as response:
                            if response.status == 200:
                                content = await response.read()
                                self.metrics.successful_requests += 1
                                self.metrics.total_data_size_bytes += len(content) if content else 0
                                break
                            else:
                                self.logger.warning(f"下载失败 {url}: HTTP {response.status}")

                    except Exception as e:
                        self.logger.warning(f"下载异常 {url} (尝试 {attempt + 1}): {e}")

                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)  # 指数退避

                if not content:
                    self.metrics.failed_requests += 1

                # 记录响应时间
                response_time = (datetime.now() - start_time).total_seconds()
                self.response_times.append(response_time)

                return url, content

        # 并发执行下载
        tasks = [download_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"下载任务异常: {result}")
                processed_results.append(("", None))
            else:
                processed_results.append(result)

        # 更新性能指标
        self._update_metrics()

        return processed_results

    def _update_metrics(self):
        """更新性能指标"""
        if self.response_times:
            self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)

        self.metrics.end_time = datetime.now()
        self.metrics.duration_seconds = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        if self.metrics.duration_seconds > 0:
            self.metrics.throughput_records_per_second = self.metrics.total_requests / self.metrics.duration_seconds

        # 记录资源使用
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)

        self.metrics.peak_memory_usage_mb = max(self.metrics.peak_memory_usage_mb, memory.used / 1024 / 1024)
        self.metrics.peak_cpu_usage_percent = max(self.metrics.peak_cpu_usage_percent, cpu)

    async def close(self):
        """关闭下载器"""
        if self.session:
            await self.session.close()


class RateLimiter:
    """速率限制器"""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取许可"""
        async with self.lock:
            now = datetime.now()

            # 清理过期请求
            cutoff = now - timedelta(minutes=1)
            self.requests = [req for req in self.requests if req > cutoff]

            # 检查是否超过限制
            if len(self.requests) >= self.requests_per_minute:
                # 计算需要等待的时间
                oldest_request = min(self.requests)
                wait_time = 60 - (now - oldest_request).total_seconds()

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # 记录当前请求
            self.requests.append(now)


class DataParser:
    """数据解析器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 线程池用于CPU密集型解析
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_parsing)

        # 缓存已解析的数据结构
        self._structure_cache = {}

    async def parse_batch(self, raw_data_batch: List[Tuple[str, bytes]], parser_func: callable,
                         **kwargs) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        并发解析数据批次

        Args:
            raw_data_batch: 原始数据批次 [(source, data), ...]
            parser_func: 解析函数
            **kwargs: 解析函数参数

        Returns:
            解析结果 [(source, parsed_data), ...]
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent_parsing)

        async def parse_single(source: str, data: bytes) -> Tuple[str, List[Dict[str, Any]]]:
            async with semaphore:
                try:
                    # 在线程池中执行解析
                    loop = asyncio.get_event_loop()
                    parsed_data = await loop.run_in_executor(
                        self.executor,
                        parser_func,
                        data,
                        **kwargs
                    )

                    return source, parsed_data or []

                except Exception as e:
                    self.logger.error(f"解析失败 {source}: {e}")
                    return source, []

        # 并发解析
        tasks = [parse_single(source, data) for source, data in raw_data_batch if data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        parsed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"解析任务异常: {result}")
                parsed_results.append(("", []))
            else:
                parsed_results.append(result)

        return parsed_results

    def shutdown(self):
        """关闭解析器"""
        self.executor.shutdown(wait=True)


class SmartCache:
    """智能缓存"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 内存缓存
        self.memory_cache = {}
        self.cache_timestamps = {}

        # 缓存统计
        self.hits = 0
        self.misses = 0

        # 预热数据
        self.warmup_data = {}

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key in self.memory_cache:
            timestamp = self.cache_timestamps.get(key)
            if timestamp and (datetime.now() - timestamp).total_seconds() < self.config.cache_ttl_seconds:
                self.hits += 1
                return self.memory_cache[key]
            else:
                # 缓存过期，删除
                del self.memory_cache[key]
                del self.cache_timestamps[key]

        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        """设置缓存数据"""
        # 检查内存使用
        if self._check_memory_usage():
            self.memory_cache[key] = value
            self.cache_timestamps[key] = datetime.now()
        else:
            self.logger.warning("内存使用过高，跳过缓存")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.memory_cache),
            "memory_usage_mb": self._get_memory_usage()
        }

    def _check_memory_usage(self) -> bool:
        """检查内存使用是否在限制内"""
        memory = psutil.virtual_memory()
        return (memory.used / 1024 / 1024) < self.config.memory_limit_mb

    def _get_memory_usage(self) -> float:
        """获取缓存内存使用"""
        # 估算内存使用（简化版）
        return len(self.memory_cache) * 1024  # 假设每个缓存项1KB

    def warmup_cache(self, warmup_data: Dict[str, Any]):
        """预热缓存"""
        self.logger.info(f"开始缓存预热，共 {len(warmup_data)} 项")

        for key, value in warmup_data.items():
            self.set(key, value)

        self.logger.info("缓存预热完成")

    def clear_expired(self):
        """清理过期缓存"""
        expired_keys = []
        now = datetime.now()

        for key, timestamp in self.cache_timestamps.items():
            if (now - timestamp).total_seconds() > self.config.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.memory_cache[key]
            del self.cache_timestamps[key]

        if expired_keys:
            self.logger.info(f"清理过期缓存: {len(expired_keys)} 项")


class MemoryManager:
    """内存管理器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 内存监控
        self.memory_threshold = config.memory_limit_mb
        self.gc_threshold = self.memory_threshold * 0.8  # 80%时触发GC

    def check_memory_pressure(self) -> bool:
        """检查内存压力"""
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / 1024 / 1024

        return memory_used_mb > self.gc_threshold

    def optimize_memory(self):
        """优化内存使用"""
        if self.check_memory_pressure():
            self.logger.info("检测到内存压力，开始优化")

            # 强制垃圾回收
            collected = gc.collect()
            self.logger.info(f"垃圾回收完成，释放 {collected} 个对象")

            # 清理未使用的缓存
            # 这里可以集成缓存清理逻辑

    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / 1024 / 1024,
            "used_mb": memory.used / 1024 / 1024,
            "free_mb": memory.free / 1024 / 1024,
            "percent": memory.percent
        }


class AdaptiveResourceManager:
    """自适应资源管理器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.memory_manager = MemoryManager(config)
        self.cpu_monitor = CPUMonitor(config.cpu_limit_percent)

        # 自适应参数
        self.concurrency_level = config.max_concurrent_downloads
        self.adjustment_interval = 60  # 每60秒调整一次

        # 启动自适应调整
        self.adjustment_task = None

    async def start_adaptive_management(self):
        """启动自适应管理"""
        self.adjustment_task = asyncio.create_task(self._adaptive_adjustment_loop())

    async def stop_adaptive_management(self):
        """停止自适应管理"""
        if self.adjustment_task:
            self.adjustment_task.cancel()
            try:
                await self.adjustment_task
            except asyncio.CancelledError:
                pass

    async def _adaptive_adjustment_loop(self):
        """自适应调整循环"""
        while True:
            try:
                await self._adjust_resources()
                await asyncio.sleep(self.adjustment_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自适应调整异常: {e}")
                await asyncio.sleep(self.adjustment_interval)

    async def _adjust_resources(self):
        """调整资源配置"""
        # 获取当前资源使用
        memory_stats = self.memory_manager.get_memory_stats()
        cpu_usage = self.cpu_monitor.get_cpu_usage()

        # 根据资源使用调整并发度
        memory_pressure = memory_stats["percent"] / 100.0
        cpu_pressure = cpu_usage / 100.0

        # 计算调整因子
        adjustment_factor = 1.0

        if memory_pressure > 0.8 or cpu_pressure > 0.8:
            # 高负载，降低并发度
            adjustment_factor = 0.7
        elif memory_pressure < 0.5 and cpu_pressure < 0.5:
            # 低负载，提高并发度
            adjustment_factor = 1.3

        # 限制调整范围
        new_concurrency = int(self.concurrency_level * adjustment_factor)
        new_concurrency = max(1, min(new_concurrency, self.config.max_concurrent_downloads))

        if new_concurrency != self.concurrency_level:
            self.logger.info(f"调整并发度: {self.concurrency_level} -> {new_concurrency} "
                           f"(内存: {memory_stats['percent']:.1f}%, CPU: {cpu_usage:.1f}%)")
            self.concurrency_level = new_concurrency

    def get_optimal_concurrency(self) -> int:
        """获取最优并发度"""
        return self.concurrency_level

    def get_resource_usage(self) -> ResourceUsage:
        """获取资源使用情况"""
        memory_stats = self.memory_manager.get_memory_stats()
        cpu_usage = self.cpu_monitor.get_cpu_usage()

        return ResourceUsage(
            memory_percent=memory_stats["percent"],
            cpu_percent=cpu_usage,
            active_threads=threading.active_count(),
            active_coroutines=len([t for t in asyncio.all_tasks() if not t.done()])
        )


class CPUMonitor:
    """CPU监控器"""

    def __init__(self, limit_percent: float):
        self.limit_percent = limit_percent

    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        return psutil.cpu_percent(interval=1)

    def is_cpu_overloaded(self) -> bool:
        """检查CPU是否过载"""
        return self.get_cpu_usage() > self.limit_percent


class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = PerformanceConfig(**config.get('performance', {}))
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.downloader = ConcurrentDownloader(self.config)
        self.parser = DataParser(self.config)
        self.cache = SmartCache(self.config)
        self.resource_manager = AdaptiveResourceManager(self.config)

        # 性能指标收集
        self.metrics_history = []

    async def initialize(self):
        """初始化优化器"""
        await self.resource_manager.start_adaptive_management()
        self.logger.info("性能优化器初始化完成")

    async def optimize_data_collection(self, collection_func: callable, *args, **kwargs) -> Any:
        """
        优化数据采集过程

        Args:
            collection_func: 采集函数
            *args, **kwargs: 函数参数

        Returns:
            采集结果
        """
        start_time = datetime.now()

        # 内存优化前检查
        self.resource_manager.memory_manager.optimize_memory()

        # 获取当前最优并发度
        optimal_concurrency = self.resource_manager.get_optimal_concurrency()

        # 执行采集（这里可以传递并发度参数给采集函数）
        kwargs['max_concurrency'] = optimal_concurrency

        try:
            result = await collection_func(*args, **kwargs)

            # 记录性能指标
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            metrics = PerformanceMetrics(
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                throughput_records_per_second=len(result) / duration if duration > 0 else 0
            )

            self.metrics_history.append(metrics)

            return result

        except Exception as e:
            self.logger.error(f"数据采集优化失败: {e}")
            raise

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics_history:
            return {"message": "暂无性能数据"}

        latest_metrics = self.metrics_history[-1]
        resource_usage = self.resource_manager.get_resource_usage()
        cache_stats = self.cache.get_cache_stats()

        # 计算平均性能
        avg_duration = sum(m.duration_seconds for m in self.metrics_history) / len(self.metrics_history)
        avg_throughput = sum(m.throughput_records_per_second for m in self.metrics_history) / len(self.metrics_history)

        return {
            "latest_metrics": {
                "duration_seconds": latest_metrics.duration_seconds,
                "throughput_records_per_second": latest_metrics.throughput_records_per_second,
                "peak_memory_mb": latest_metrics.peak_memory_usage_mb,
                "peak_cpu_percent": latest_metrics.peak_cpu_usage_percent
            },
            "average_metrics": {
                "duration_seconds": avg_duration,
                "throughput_records_per_second": avg_throughput
            },
            "resource_usage": {
                "memory_percent": resource_usage.memory_percent,
                "cpu_percent": resource_usage.cpu_percent,
                "active_threads": resource_usage.active_threads,
                "active_coroutines": resource_usage.active_coroutines
            },
            "cache_stats": cache_stats,
            "total_operations": len(self.metrics_history)
        }

    async def cleanup(self):
        """清理资源"""
        await self.resource_manager.stop_adaptive_management()
        await self.downloader.close()
        self.parser.shutdown()

        # 清理缓存
        self.cache.clear_expired()

        self.logger.info("性能优化器资源清理完成")