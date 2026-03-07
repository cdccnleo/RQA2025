#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis缓存集成示例

展示如何在数据层中使用Redis缓存进行性能优化。
"""

import asyncio
import time
import pandas as pd

from src.data import (
    DataOptimizer,
    OptimizationConfig,
    MultiLevelCache,
    CacheConfig,
    RedisCacheConfig,
    create_redis_cache
)
from src.engine.logging.unified_logger import get_unified_logger

logger = get_unified_logger('redis_cache_example')


class RedisCacheIntegrationExample:
    """Redis缓存集成示例类"""

    def __init__(self):
        """初始化示例"""
        self.setup_cache_config()
        self.setup_optimizer()

    def setup_cache_config(self):
        """设置缓存配置"""
        # Redis缓存配置
        redis_config = RedisCacheConfig(
            host='localhost',
            port=6379,
            db=0,
            default_ttl=3600,  # 1小时
            enable_compression=True,
            serialization_format='pickle'
        )

        # 多级缓存配置
        self.cache_config = CacheConfig(
            # 内存缓存
            memory_max_size=1000,
            memory_ttl=300,  # 5分钟

            # 磁盘缓存
            disk_enabled=True,
            disk_cache_dir="cache",
            disk_ttl=3600,  # 1小时
            disk_max_size_mb=1024,  # 1GB

            # Redis缓存
            redis_enabled=True,
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'default_ttl': 7200,  # 2小时
                'enable_compression': True,
                'serialization_format': 'pickle'
            },
            redis_ttl=7200  # 2小时
        )

    def setup_optimizer(self):
        """设置数据优化器"""
        self.optimizer_config = OptimizationConfig(
            enable_cache=True,
            cache_config=self.cache_config,
            enable_parallel_loading=True,
            max_workers=4,
            enable_quality_monitor=True,
            enable_performance_monitor=True
        )

        self.optimizer = DataOptimizer(self.optimizer_config)

    async def demonstrate_basic_redis_cache(self):
        """演示基础Redis缓存功能"""
        logger.info("=== 基础Redis缓存演示 ===")

        # 创建Redis缓存适配器
        redis_cache = create_redis_cache()

        # 测试数据
        test_data = {
            'stock_data': pd.DataFrame({
                'symbol': ['AAPL', 'GOOGL', 'MSFT'],
                'price': [150.0, 2800.0, 300.0],
                'volume': [1000000, 500000, 800000]
            }),
            'market_data': {
                'timestamp': time.time(),
                'indices': {'SPY': 450.0, 'QQQ': 380.0},
                'volatility': 0.15
            }
        }

        # 设置缓存
        for key, value in test_data.items():
            success = redis_cache.set(key, value, ttl=3600)
            logger.info(f"设置缓存 {key}: {'成功' if success else '失败'}")

        # 获取缓存
        for key in test_data.keys():
            cached_data = redis_cache.get(key)
            if cached_data is not None:
                logger.info(f"获取缓存 {key}: 成功")
                if isinstance(cached_data, pd.DataFrame):
                    logger.info(f"  DataFrame形状: {cached_data.shape}")
                else:
                    logger.info(f"  数据类型: {type(cached_data)}")
            else:
                logger.info(f"获取缓存 {key}: 失败")

        # 获取统计信息
        stats = redis_cache.get_stats()
        logger.info(f"Redis缓存统计: {stats}")

        # 清理
        for key in test_data.keys():
            redis_cache.delete(key)

        redis_cache.close()

    async def demonstrate_multi_level_cache(self):
        """演示多级缓存功能"""
        logger.info("=== 多级缓存演示 ===")

        # 创建多级缓存
        multi_cache = MultiLevelCache(self.cache_config)

        # 测试数据
        test_data = pd.DataFrame({
            'symbol': ['TSLA', 'AMZN', 'NVDA'],
            'price': [250.0, 3200.0, 450.0],
            'volume': [2000000, 1500000, 3000000]
        })

        # 设置缓存
        success = multi_cache.set('test_stock_data', test_data, ttl=1800)
        logger.info(f"设置多级缓存: {'成功' if success else '失败'}")

        # 获取缓存
        cached_data = multi_cache.get('test_stock_data')
        if cached_data is not None:
            logger.info(f"获取多级缓存: 成功")
            logger.info(f"  DataFrame形状: {cached_data.shape}")
        else:
            logger.info(f"获取多级缓存: 失败")

        # 获取统计信息
        stats = multi_cache.get_stats()
        logger.info(f"多级缓存统计: {stats}")

        # 清理
        multi_cache.delete('test_stock_data')

    async def demonstrate_optimized_data_loading(self):
        """演示优化的数据加载"""
        logger.info("=== 优化数据加载演示 ===")

        try:
            # 模拟数据加载
            result = await self.optimizer.optimize_data_loading(
                data_type='stock',
                start_date='2024-01-01',
                end_date='2024-01-31',
                frequency='1d',
                symbols=['AAPL', 'GOOGL', 'MSFT']
            )

            if result.success:
                logger.info("数据加载优化成功")
                logger.info(f"  加载时间: {result.load_time_ms:.2f}ms")
                logger.info(f"  缓存命中: {result.cache_hit}")
                if result.performance_metrics:
                    logger.info(f"  性能指标: {result.performance_metrics}")
            else:
                logger.error(f"数据加载优化失败: {result.error_message}")

        except Exception as e:
            logger.error(f"数据加载优化异常: {e}")

    async def demonstrate_batch_operations(self):
        """演示批量操作"""
        logger.info("=== 批量操作演示 ===")

        redis_cache = create_redis_cache()

        # 批量设置数据
        batch_data = {
            'batch_1': {'symbols': ['AAPL', 'GOOGL'], 'prices': [150, 2800]},
            'batch_2': {'symbols': ['MSFT', 'TSLA'], 'prices': [300, 250]},
            'batch_3': {'symbols': ['AMZN', 'NVDA'], 'prices': [3200, 450]}
        }

        # 批量设置
        success = redis_cache.mset(batch_data, ttl=1800)
        logger.info(f"批量设置: {'成功' if success else '失败'}")

        # 批量获取
        keys = list(batch_data.keys())
        cached_data = redis_cache.mget(keys)
        logger.info(f"批量获取: 成功获取 {len(cached_data)} 个键")

        # 模式清除
        deleted_count = redis_cache.clear_pattern('batch_*')
        logger.info(f"模式清除: 删除了 {deleted_count} 个键")

        redis_cache.close()

    async def demonstrate_performance_comparison(self):
        """演示性能对比"""
        logger.info("=== 性能对比演示 ===")

        # 创建不同配置的缓存
        memory_only_config = CacheConfig(
            redis_enabled=False,
            disk_enabled=False
        )

        memory_disk_config = CacheConfig(
            redis_enabled=False,
            disk_enabled=True
        )

        full_config = CacheConfig(
            redis_enabled=True,
            disk_enabled=True
        )

        configs = [
            ('仅内存', memory_only_config),
            ('内存+磁盘', memory_disk_config),
            ('内存+磁盘+Redis', full_config)
        ]

        test_data = pd.DataFrame({
            'symbol': [f'STOCK_{i}' for i in range(100)],
            'price': [100 + i for i in range(100)],
            'volume': [1000000 + i * 10000 for i in range(100)]
        })

        for name, config in configs:
            logger.info(f"\n--- {name} 缓存测试 ---")

            cache = MultiLevelCache(config)

            # 测试设置性能
            start_time = time.time()
            for i in range(10):
                cache.set(f'test_key_{i}', test_data, ttl=3600)
            set_time = time.time() - start_time

            # 测试获取性能
            start_time = time.time()
            for i in range(10):
                cache.get(f'test_key_{i}')
            get_time = time.time() - start_time

            # 获取统计信息
            stats = cache.get_stats()

            logger.info(f"  设置时间: {set_time:.4f}秒")
            logger.info(f"  获取时间: {get_time:.4f}秒")
            logger.info(f"  命中率: {stats['performance']['hit_rate']}")

            # 清理
            for i in range(10):
                cache.delete(f'test_key_{i}')

    async def run_all_demonstrations(self):
        """运行所有演示"""
        logger.info("开始Redis缓存集成演示")

        try:
            await self.demonstrate_basic_redis_cache()
            await asyncio.sleep(1)

            await self.demonstrate_multi_level_cache()
            await asyncio.sleep(1)

            await self.demonstrate_batch_operations()
            await asyncio.sleep(1)

            await self.demonstrate_performance_comparison()
            await asyncio.sleep(1)

            await self.demonstrate_optimized_data_loading()

        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")

        logger.info("Redis缓存集成演示完成")


async def main():
    """主函数"""
    example = RedisCacheIntegrationExample()
    await example.run_all_demonstrations()


if __name__ == "__main__":
    asyncio.run(main())
