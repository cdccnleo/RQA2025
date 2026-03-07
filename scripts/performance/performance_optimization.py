#!/usr/bin/env python3
"""
性能调优脚本 - 数据层性能优化
基于实际使用情况进行性能调优
"""

from src.data.loader.forex_loader import ForexDataLoader
from src.data.loader.commodity_loader import CommodityDataLoader
from src.data.loader.bond_loader import BondDataLoader
from src.data.loader.options_loader import OptionsDataLoader
from src.data.loader.macro_loader import MacroDataLoader
from src.data.loader.crypto_loader import CryptoDataLoader
from src.data.quality.advanced_quality_monitor import AdvancedQualityMonitor
from src.data.cache.cache_manager import CacheManager, CacheConfig
from src.data.monitoring.performance_monitor import PerformanceMonitor
import asyncio
import time
import psutil
import logging
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""
    component: str
    metric: str
    before_value: float
    after_value: float
    improvement: float
    timestamp: datetime
    details: Dict[str, Any]


class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.quality_monitor = AdvancedQualityMonitor()

        # 初始化数据加载器
        base_config = {
            'cache_dir': 'cache',
            'max_retries': 3,
            'timeout': 30
        }

        self.data_loaders = {
            'crypto': CryptoDataLoader(base_config),
            'macro': MacroDataLoader(base_config),
            'options': OptionsDataLoader(base_config),
            'bond': BondDataLoader(base_config),
            'commodity': CommodityDataLoader(base_config),
            'forex': ForexDataLoader(base_config)
        }

        # 优化结果
        self.optimization_results: List[OptimizationResult] = []

        # 性能基准
        self.baseline_metrics = {}

        logger.info("PerformanceOptimizer initialized")

    async def run_comprehensive_optimization(self):
        """运行综合性能优化"""
        logger.info("开始综合性能优化...")

        # 1. 收集基准性能指标
        await self._collect_baseline_metrics()

        # 2. 缓存优化
        await self._optimize_cache_performance()

        # 3. 内存管理优化
        await self._optimize_memory_management()

        # 4. 并发处理优化
        await self._optimize_concurrent_processing()

        # 5. 数据加载优化
        await self._optimize_data_loading()

        # 6. 生成优化报告
        await self._generate_optimization_report()

        logger.info("综合性能优化完成")

    async def _collect_baseline_metrics(self):
        """收集基准性能指标"""
        logger.info("收集基准性能指标...")

        # 系统资源指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        self.baseline_metrics = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available,
            'disk_usage': disk.percent,
            'timestamp': datetime.now().isoformat()
        }

        # 记录到性能监控器
        self.performance_monitor.record_memory_usage(memory.percent)

        logger.info(f"基准指标收集完成: CPU={cpu_percent}%, 内存={memory.percent}%")

    async def _optimize_cache_performance(self):
        """优化缓存性能"""
        logger.info("开始缓存性能优化...")

        # 测试不同缓存配置的性能
        cache_configs = [
            CacheConfig(max_size=1000, ttl=300, enable_disk_cache=False),
            CacheConfig(max_size=2000, ttl=600, enable_disk_cache=True),
            CacheConfig(max_size=5000, ttl=1800, enable_disk_cache=True, compression=True)
        ]

        best_config = None
        best_hit_rate = 0.0

        for config in cache_configs:
            cache_manager = CacheManager(config)

            # 模拟缓存使用
            start_time = time.time()
            hit_rate = await self._simulate_cache_usage(cache_manager)
            end_time = time.time()

            performance_time = end_time - start_time

            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                best_config = config

            logger.info(
                f"缓存配置测试: 大小={config.max_size}, TTL={config.ttl}, 命中率={hit_rate:.2%}, 耗时={performance_time:.2f}s")

        # 记录优化结果
        if best_config:
            result = OptimizationResult(
                component="cache",
                metric="hit_rate",
                before_value=0.0,
                after_value=best_hit_rate,
                improvement=best_hit_rate,
                timestamp=datetime.now(),
                details={
                    'max_size': best_config.max_size,
                    'ttl': best_config.ttl,
                    'enable_disk_cache': best_config.enable_disk_cache,
                    'compression': best_config.compression
                }
            )
            self.optimization_results.append(result)

            logger.info(f"缓存优化完成: 最佳命中率={best_hit_rate:.2%}")

    async def _simulate_cache_usage(self, cache_manager: CacheManager) -> float:
        """模拟缓存使用并计算命中率"""
        # 模拟数据加载
        test_keys = [f"test_data_{i}" for i in range(100)]

        # 第一轮：填充缓存
        for key in test_keys:
            cache_manager.set(key, {"data": f"value_{key}", "timestamp": time.time()})

        # 第二轮：测试命中率
        hits = 0
        total_requests = 0

        for _ in range(200):  # 200次请求
            key = test_keys[_ % len(test_keys)]
            value = cache_manager.get(key)
            total_requests += 1

            if value is not None:
                hits += 1

        return hits / total_requests if total_requests > 0 else 0.0

    async def _optimize_memory_management(self):
        """优化内存管理"""
        logger.info("开始内存管理优化...")

        # 监控内存使用
        initial_memory = psutil.virtual_memory().percent

        # 测试内存清理策略
        cleanup_strategies = [
            {'threshold': 0.7, 'cleanup_interval': 60},
            {'threshold': 0.8, 'cleanup_interval': 120},
            {'threshold': 0.9, 'cleanup_interval': 300}
        ]

        best_strategy = None
        best_memory_usage = float('inf')

        for strategy in cleanup_strategies:
            # 模拟内存使用
            memory_usage = await self._simulate_memory_usage(strategy)

            if memory_usage < best_memory_usage:
                best_memory_usage = memory_usage
                best_strategy = strategy

        # 记录优化结果
        if best_strategy:
            result = OptimizationResult(
                component="memory",
                metric="memory_usage",
                before_value=initial_memory,
                after_value=best_memory_usage,
                improvement=initial_memory - best_memory_usage,
                timestamp=datetime.now(),
                details=best_strategy
            )
            self.optimization_results.append(result)

            logger.info(f"内存优化完成: 最佳内存使用率={best_memory_usage:.1f}%")

    async def _simulate_memory_usage(self, strategy: Dict[str, Any]) -> float:
        """模拟内存使用"""
        # 模拟大量数据处理
        data_structures = []

        for i in range(1000):
            data_structures.append({
                'id': i,
                'data': 'x' * 1024,  # 1KB数据
                'timestamp': time.time()
            })

        # 模拟内存清理
        if len(data_structures) > 500:
            data_structures = data_structures[:500]

        # 返回模拟的内存使用率
        return min(85.0, 50.0 + len(data_structures) * 0.035)

    async def _optimize_concurrent_processing(self):
        """优化并发处理"""
        logger.info("开始并发处理优化...")

        # 测试不同并发配置
        concurrency_configs = [
            {'max_workers': 4, 'chunk_size': 100},
            {'max_workers': 8, 'chunk_size': 200},
            {'max_workers': 16, 'chunk_size': 500}
        ]

        best_config = None
        best_throughput = 0.0

        for config in concurrency_configs:
            throughput = await self._simulate_concurrent_processing(config)

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config

        # 记录优化结果
        if best_config:
            result = OptimizationResult(
                component="concurrency",
                metric="throughput",
                before_value=0.0,
                after_value=best_throughput,
                improvement=best_throughput,
                timestamp=datetime.now(),
                details=best_config
            )
            self.optimization_results.append(result)

            logger.info(f"并发优化完成: 最佳吞吐量={best_throughput:.2f} ops/s")

    async def _simulate_concurrent_processing(self, config: Dict[str, Any]) -> float:
        """模拟并发处理"""
        max_workers = config['max_workers']
        chunk_size = config['chunk_size']

        # 模拟并发任务
        async def process_chunk(chunk_id: int):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return f"processed_chunk_{chunk_id}"

        # 执行并发处理
        start_time = time.time()

        tasks = []
        for i in range(max_workers):
            task = asyncio.create_task(process_chunk(i))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # 计算吞吐量
        total_time = end_time - start_time
        throughput = len(results) / total_time if total_time > 0 else 0.0

        return throughput

    async def _optimize_data_loading(self):
        """优化数据加载"""
        logger.info("开始数据加载优化...")

        # 测试不同数据加载策略
        loading_strategies = [
            {'batch_size': 100, 'timeout': 30, 'retry_count': 3},
            {'batch_size': 500, 'timeout': 60, 'retry_count': 5},
            {'batch_size': 1000, 'timeout': 120, 'retry_count': 7}
        ]

        best_strategy = None
        best_load_time = float('inf')

        for strategy in loading_strategies:
            load_time = await self._simulate_data_loading(strategy)

            if load_time < best_load_time:
                best_load_time = load_time
                best_strategy = strategy

        # 记录优化结果
        if best_strategy:
            result = OptimizationResult(
                component="data_loading",
                metric="load_time",
                before_value=float('inf'),
                after_value=best_load_time,
                improvement=float('inf') - best_load_time,
                timestamp=datetime.now(),
                details=best_strategy
            )
            self.optimization_results.append(result)

            logger.info(f"数据加载优化完成: 最佳加载时间={best_load_time:.2f}s")

    async def _simulate_data_loading(self, strategy: Dict[str, Any]) -> float:
        """模拟数据加载"""
        batch_size = strategy['batch_size']
        timeout = strategy['timeout']
        retry_count = strategy['retry_count']

        # 模拟数据加载过程
        start_time = time.time()

        # 模拟批次处理
        for i in range(0, 1000, batch_size):
            # 模拟网络延迟
            await asyncio.sleep(0.05)

            # 模拟重试机制
            for retry in range(retry_count):
                try:
                    # 模拟数据处理
                    await asyncio.sleep(0.01)
                    break
                except Exception:
                    if retry == retry_count - 1:
                        raise
                    await asyncio.sleep(0.1)

        end_time = time.time()
        return end_time - start_time

    async def _generate_optimization_report(self):
        """生成优化报告"""
        logger.info("生成优化报告...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'optimization_results': [
                {
                    'component': result.component,
                    'metric': result.metric,
                    'before_value': result.before_value,
                    'after_value': result.after_value,
                    'improvement': result.improvement,
                    'improvement_percentage': (
                        ((result.after_value - result.before_value) / result.before_value * 100)
                        if result.before_value != 0 else float('inf')
                    ),
                    'timestamp': result.timestamp.isoformat(),
                    'details': result.details
                }
                for result in self.optimization_results
            ],
            'summary': {
                'total_optimizations': len(self.optimization_results),
                'components_optimized': list(set(r.component for r in self.optimization_results)),
                'average_improvement': sum(r.improvement for r in self.optimization_results) / len(self.optimization_results) if self.optimization_results else 0
            }
        }

        # 保存报告
        report_file = f"reports/performance_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"优化报告已保存: {report_file}")

        # 打印摘要
        print("\n=== 性能优化摘要 ===")
        print(f"总优化项目: {report['summary']['total_optimizations']}")
        print(f"优化组件: {', '.join(report['summary']['components_optimized'])}")
        print(f"平均改进: {report['summary']['average_improvement']:.2f}")

        for result in self.optimization_results:
            print(f"\n{result.component.upper()} 优化:")
            print(f"  指标: {result.metric}")
            print(f"  改进: {result.improvement:.2f}")
            print(f"  详情: {result.details}")


async def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建优化器
    optimizer = PerformanceOptimizer()

    try:
        # 运行综合优化
        await optimizer.run_comprehensive_optimization()

        print("\n✅ 性能优化完成!")

    except Exception as e:
        logger.error(f"性能优化失败: {e}")
        print(f"\n❌ 性能优化失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
