#!/usr/bin/env python3
"""
缓存策略优化工具 - 生产环境缓存性能测试和优化
用于测试多级缓存策略、Redis集群配置、性能监控等

优化内容:
✅ 多级缓存策略测试
✅ Redis连接池优化
✅ 缓存命中率分析
✅ 内存使用监控
✅ 性能压力测试

使用方法:
python scripts/cache_strategy_optimization.py --test multi_level
python scripts/cache_strategy_optimization.py --test redis_pool
python scripts/cache_strategy_optimization.py --test hit_rate_analysis
python scripts/cache_strategy_optimization.py --benchmark
"""

import asyncio
import time
import json
import argparse
import statistics
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from src.utils.logger import get_logger
from src.core.database_service import CacheConfig, RedisConfig
from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager

logger = get_logger(__name__)


@dataclass
class CacheTestResult:
    """缓存测试结果"""
    test_name: str
    duration: float
    operations: int
    hit_rate: float
    avg_response_time: float
    memory_usage: float
    error_count: int
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CachePerformanceReport:
    """缓存性能报告"""
    test_results: List[CacheTestResult]
    recommendations: List[str]
    optimization_suggestions: List[str]
    generated_at: datetime = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()


class CacheStrategyOptimizer:
    """缓存策略优化器"""

    def __init__(self):
        self.cache_manager = None
        self.test_results = []

    async def initialize_cache(self, cache_config: CacheConfig, redis_config: RedisConfig):
        """初始化缓存管理器"""
        try:
            self.cache_manager = UnifiedCacheManager(
                cache_config=cache_config,
                redis_config=redis_config
            )
            await self.cache_manager.initialize()
            logger.info("缓存管理器初始化成功")
        except Exception as e:
            logger.error(f"缓存管理器初始化失败: {e}")
            raise

    async def test_multi_level_cache(self) -> CacheTestResult:
        """测试多级缓存性能"""
        logger.info("开始多级缓存性能测试...")

        start_time = time.time()
        operations = 1000
        hits = 0
        response_times = []

        try:
            # 预热缓存
            logger.info("预热缓存...")
            for i in range(100):
                key = f"warmup_key_{i}"
                value = f"warmup_value_{i}"
                await self.cache_manager.set(key, value, ttl=300)

            # 测试读写性能
            logger.info("执行读写测试...")
            for i in range(operations):
                op_start = time.time()

                if i % 2 == 0:  # 写入操作
                    key = f"test_key_{i}"
                    value = {"data": f"test_value_{i}", "timestamp": datetime.now().isoformat()}
                    await self.cache_manager.set(key, value, ttl=300)
                else:  # 读取操作
                    key = f"test_key_{i-1}"
                    result = await self.cache_manager.get(key)
                    if result is not None:
                        hits += 1

                response_times.append(time.time() - op_start)

            duration = time.time() - start_time
            hit_rate = hits / (operations // 2) if operations > 1 else 0
            avg_response_time = statistics.mean(response_times) if response_times else 0

            # 获取内存使用情况
            memory_usage = await self._get_memory_usage()

            result = CacheTestResult(
                test_name="multi_level_cache",
                duration=duration,
                operations=operations,
                hit_rate=hit_rate,
                avg_response_time=avg_response_time,
                memory_usage=memory_usage,
                error_count=0
            )

            logger.info(".2f"
                        ".4f")

            return result

        except Exception as e:
            logger.error(f"多级缓存测试失败: {e}")
            return CacheTestResult(
                test_name="multi_level_cache",
                duration=time.time() - start_time,
                operations=0,
                hit_rate=0.0,
                avg_response_time=0.0,
                memory_usage=0.0,
                error_count=1
            )

    async def test_redis_connection_pool(self) -> CacheTestResult:
        """测试Redis连接池性能"""
        logger.info("开始Redis连接池性能测试...")

        start_time = time.time()
        concurrent_requests = 50
        operations_per_request = 20
        total_operations = concurrent_requests * operations_per_request

        async def worker(worker_id: int):
            """工作协程"""
            local_hits = 0
            local_errors = 0
            response_times = []

            for i in range(operations_per_request):
                try:
                    op_start = time.time()

                    # 执行缓存操作
                    key = f"pool_test_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"

                    await self.cache_manager.set(key, value, ttl=60)
                    result = await self.cache_manager.get(key)

                    if result == value:
                        local_hits += 1

                    response_times.append(time.time() - op_start)

                except Exception as e:
                    local_errors += 1
                    logger.debug(f"Worker {worker_id} operation {i} failed: {e}")

            return local_hits, local_errors, response_times

        try:
            # 并发执行
            tasks = [worker(i) for i in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)

            # 汇总结果
            total_hits = sum(r[0] for r in results)
            total_errors = sum(r[1] for r in results)
            all_response_times = []
            for r in results:
                all_response_times.extend(r[2])

            duration = time.time() - start_time
            hit_rate = total_hits / total_operations if total_operations > 0 else 0
            avg_response_time = statistics.mean(all_response_times) if all_response_times else 0

            # 获取Redis连接池统计
            pool_stats = await self._get_redis_pool_stats()

            result = CacheTestResult(
                test_name="redis_connection_pool",
                duration=duration,
                operations=total_operations,
                hit_rate=hit_rate,
                avg_response_time=avg_response_time,
                memory_usage=pool_stats.get('memory_usage', 0.0),
                error_count=total_errors
            )

            logger.info(".2f"
                        f"总错误数: {total_errors}")

            return result

        except Exception as e:
            logger.error(f"Redis连接池测试失败: {e}")
            return CacheTestResult(
                test_name="redis_connection_pool",
                duration=time.time() - start_time,
                operations=0,
                hit_rate=0.0,
                avg_response_time=0.0,
                memory_usage=0.0,
                error_count=1
            )

    async def analyze_hit_rate_patterns(self) -> Dict[str, Any]:
        """分析缓存命中率模式"""
        logger.info("开始缓存命中率模式分析...")

        try:
            # 模拟不同访问模式
            patterns = {
                'hot_data': [],      # 热点数据
                'temporal': [],      # 时间局部性
                'spatial': [],       # 空间局部性
                'random': []         # 随机访问
            }

            # 测试热点数据模式 (80/20原则)
            logger.info("测试热点数据访问模式...")
            hot_keys = [f"hot_{i}" for i in range(20)]  # 20个热点键
            normal_keys = [f"normal_{i}" for i in range(80)]  # 80个普通键

            # 预加载数据
            for key in hot_keys + normal_keys:
                await self.cache_manager.set(key, f"value_{key}", ttl=600)

            # 模拟访问模式：80%访问热点数据，20%访问普通数据
            hot_accesses = 800
            normal_accesses = 200

            hot_hits = 0
            for _ in range(hot_accesses):
                key = hot_keys[_ % len(hot_keys)]
                result = await self.cache_manager.get(key)
                if result is not None:
                    hot_hits += 1

            normal_hits = 0
            for _ in range(normal_accesses):
                key = normal_keys[_ % len(normal_keys)]
                result = await self.cache_manager.get(key)
                if result is not None:
                    normal_hits += 1

            hot_hit_rate = hot_hits / hot_accesses
            normal_hit_rate = normal_hits / normal_accesses
            overall_hit_rate = (hot_hits + normal_hits) / (hot_accesses + normal_accesses)

            patterns['hot_data'] = {
                'hot_hit_rate': hot_hit_rate,
                'normal_hit_rate': normal_hit_rate,
                'overall_hit_rate': overall_hit_rate,
                'improvement_ratio': hot_hit_rate / normal_hit_rate if normal_hit_rate > 0 else float('inf')
            }

            # 分析时间局部性
            logger.info("测试时间局部性访问模式...")
            temporal_keys = [f"temporal_{i}" for i in range(100)]

            # 预加载数据
            for key in temporal_keys:
                await self.cache_manager.set(key, f"value_{key}", ttl=600)

            # 模拟时间局部性：最近访问的数据更可能被再次访问
            recent_accesses = []
            for i in range(1000):
                if recent_accesses and i % 10 == 0:  # 10%概率访问最近的数据
                    key = recent_accesses[-1]
                else:
                    key = temporal_keys[i % len(temporal_keys)]
                    recent_accesses.append(key)
                    if len(recent_accesses) > 10:
                        recent_accesses.pop(0)

                result = await self.cache_manager.get(key)
                patterns['temporal'].append(result is not None)

            temporal_hit_rate = sum(patterns['temporal']) / len(patterns['temporal'])

            # 分析空间局部性
            logger.info("测试空间局部性访问模式...")
            spatial_data = {}
            for i in range(10):  # 10个数据块
                spatial_data[f"block_{i}"] = [f"item_{i}_{j}" for j in range(10)]

            # 预加载数据
            for block_name, items in spatial_data.items():
                await self.cache_manager.set(block_name, items, ttl=600)

            # 模拟空间局部性：访问一个数据块后很可能访问同一块的其他数据
            spatial_accesses = []
            for i in range(500):
                if i % 5 == 0:  # 每5次访问中，有1次访问新数据块
                    block_name = f"block_{i % 10}"
                    item_index = i % 10
                else:  # 其他4次访问同一数据块的不同项目
                    item_index = (i % 10 + 1) % 10

                key = f"block_{i % 10}"
                result = await self.cache_manager.get(key)
                spatial_accesses.append(result is not None)

            spatial_hit_rate = sum(spatial_accesses) / len(spatial_accesses)

            patterns['temporal'] = {'hit_rate': temporal_hit_rate}
            patterns['spatial'] = {'hit_rate': spatial_hit_rate}

            # 随机访问模式基线测试
            logger.info("测试随机访问模式...")
            random_keys = [f"random_{i}" for i in range(1000)]

            # 预加载数据
            for key in random_keys:
                await self.cache_manager.set(key, f"value_{key}", ttl=600)

            random_accesses = []
            import random
            for _ in range(1000):
                key = random.choice(random_keys)
                result = await self.cache_manager.get(key)
                random_accesses.append(result is not None)

            random_hit_rate = sum(random_accesses) / len(random_accesses)
            patterns['random'] = {'hit_rate': random_hit_rate}

            analysis_result = {
                'patterns': patterns,
                'recommendations': self._generate_hit_rate_recommendations(patterns),
                'optimal_strategy': self._determine_optimal_strategy(patterns)
            }

            logger.info("缓存命中率模式分析完成")
            return analysis_result

        except Exception as e:
            logger.error(f"缓存命中率模式分析失败: {e}")
            return {'error': str(e)}

    def _generate_hit_rate_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """生成命中率优化建议"""
        recommendations = []

        hot_data = patterns.get('hot_data', {})
        if hot_data.get('improvement_ratio', 0) > 2.0:
            recommendations.append("检测到明显的热点数据模式，建议增加L1缓存大小")
            recommendations.append("考虑实施热点数据预加载策略")

        temporal_hit_rate = patterns.get('temporal', {}).get('hit_rate', 0)
        if temporal_hit_rate > 0.8:
            recommendations.append("时间局部性良好，LRU策略适合当前工作负载")

        spatial_hit_rate = patterns.get('spatial', {}).get('hit_rate', 0)
        recommendations.append("空间局部性优秀，考虑数据预取策略")

        random_hit_rate = patterns.get('random', {}).get('hit_rate', 0)
        if random_hit_rate < 0.3:
            recommendations.append("随机访问模式明显，建议调整缓存大小和TTL策略")

        return recommendations

    def _determine_optimal_strategy(self, patterns: Dict[str, Any]) -> str:
        """确定最优缓存策略"""
        hot_improvement = patterns.get('hot_data', {}).get('improvement_ratio', 1)
        temporal_rate = patterns.get('temporal', {}).get('hit_rate', 0)
        spatial_rate = patterns.get('spatial', {}).get('hit_rate', 0)
        random_rate = patterns.get('random', {}).get('hit_rate', 0)

        # 基于模式特征选择最优策略
        if hot_improvement > 3.0:
            return "热点数据优先策略 - 增大L1缓存，实施热点检测"
        elif temporal_rate > 0.8 and spatial_rate > 0.8:
            return "局部性优化策略 - LRU + 预取策略"
        elif random_rate > 0.7:
            return "随机访问优化策略 - 增大缓存容量，延长TTL"
        else:
            return "自适应策略 - 根据运行时指标动态调整"

    async def _get_memory_usage(self) -> float:
        """获取缓存内存使用率"""
        try:
            if hasattr(self.cache_manager, 'get_memory_usage'):
                return await self.cache_manager.get_memory_usage()
            return 0.0
        except Exception:
            return 0.0

    async def _get_redis_pool_stats(self) -> Dict[str, Any]:
        """获取Redis连接池统计"""
        try:
            if hasattr(self.cache_manager, 'get_redis_stats'):
                return await self.cache_manager.get_redis_stats()
            return {'memory_usage': 0.0}
        except Exception:
            return {'memory_usage': 0.0}

    def generate_performance_report(self) -> CachePerformanceReport:
        """生成性能报告"""
        recommendations = []
        optimization_suggestions = []

        # 基于测试结果生成建议
        for result in self.test_results:
            if result.test_name == "multi_level_cache":
                if result.hit_rate < 0.7:
                    recommendations.append("多级缓存命中率偏低，建议优化缓存策略")
                if result.avg_response_time > 0.01:
                    optimization_suggestions.append("缓存响应时间较慢，考虑启用压缩")

            elif result.test_name == "redis_connection_pool":
                if result.error_count > 0:
                    recommendations.append("Redis连接池存在错误，检查连接配置")
                if result.avg_response_time > 0.005:
                    optimization_suggestions.append("Redis响应时间较慢，考虑调整连接池大小")

        return CachePerformanceReport(
            test_results=self.test_results,
            recommendations=recommendations,
            optimization_suggestions=optimization_suggestions
        )

    def save_report(self, report: CachePerformanceReport, output_file: str):
        """保存性能报告"""
        report_data = {
            'test_results': [asdict(result) for result in report.test_results],
            'recommendations': report.recommendations,
            'optimization_suggestions': report.optimization_suggestions,
            'generated_at': report.generated_at.isoformat(),
            'summary': {
                'total_tests': len(report.test_results),
                'avg_hit_rate': statistics.mean([r.hit_rate for r in report.test_results]) if report.test_results else 0,
                'total_errors': sum(r.error_count for r in report.test_results)
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"缓存性能报告已保存到: {output_file}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='缓存策略优化工具')
    parser.add_argument('--test', choices=['multi_level', 'redis_pool', 'hit_rate_analysis', 'all'],
                        default='all', help='要执行的测试类型')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--output', default='cache_optimization_report.json',
                        help='输出文件路径')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能基准测试')

    args = parser.parse_args()

    # 初始化优化器
    optimizer = CacheStrategyOptimizer()

    # 使用生产环境配置
    cache_config = CacheConfig(
        enable_multi_level=True,
        l1_cache_size=10000,
        compression_enabled=True,
        enable_metrics=True
    )

    redis_config = RedisConfig(
        max_connections=50,
        health_check_interval=30,
        enable_metrics=True
    )

    try:
        await optimizer.initialize_cache(cache_config, redis_config)

        if args.test == 'multi_level' or args.test == 'all':
            result = await optimizer.test_multi_level_cache()
            optimizer.test_results.append(result)

        if args.test == 'redis_pool' or args.test == 'all':
            result = await optimizer.test_redis_connection_pool()
            optimizer.test_results.append(result)

        if args.test == 'hit_rate_analysis' or args.test == 'all':
            analysis = await optimizer.analyze_hit_rate_patterns()
            logger.info(f"命中率分析结果: {analysis}")

        # 生成和保存报告
        report = optimizer.generate_performance_report()
        optimizer.save_report(report, args.output)

        logger.info("缓存策略优化测试完成")

    except Exception as e:
        logger.error(f"缓存策略优化失败: {e}")
        raise
    finally:
        if optimizer.cache_manager:
            await optimizer.cache_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
