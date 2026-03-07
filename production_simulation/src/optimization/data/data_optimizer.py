#!/usr/bin/env python3
"""
数据层优化器模块

整合并行加载、缓存优化、质量监控等功能，提供统一的数据优化接口。
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

from ..interfaces import IDataModel
from ..parallel.parallel_loader import ParallelLoadingManager
from ..cache.multi_level_cache import MultiLevelCache, CacheConfig
from ..quality.advanced_quality_monitor import AdvancedQualityMonitor
from ..data_manager import DataManagerSingleton

logger = get_infrastructure_logger('data_optimizer')


@dataclass
class OptimizationConfig:

    """数据优化配置"""
    # 并行加载配置
    max_workers: int = 4
    enable_parallel_loading: bool = True

    # 缓存配置
    enable_cache: bool = True
    cache_config: Optional[CacheConfig] = None

    # 质量监控配置
    enable_quality_monitor: bool = True
    quality_threshold: float = 0.8

    # 性能监控配置
    enable_performance_monitor: bool = True
    performance_threshold_ms: int = 5000

    # 预加载配置
    enable_preload: bool = False
    preload_symbols: List[str] = None
    preload_days: int = 30


@dataclass
class OptimizationResult:

    """优化结果"""
    success: bool
    data_model: Optional[IDataModel]
    performance_metrics: Dict[str, Any]
    quality_metrics: Optional[Dict[str, Any]]
    cache_hit: bool
    load_time_ms: float
    error_message: Optional[str] = None


class DataOptimizer:

    """
    数据层优化器

    整合并行加载、缓存优化、质量监控等功能，提供统一的数据优化接口。
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化数据优化器

        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()

        # 初始化组件
        self._init_components()

        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_load_time_ms': 0,
            'quality_scores': []
        }

        logger.info("DataOptimizer initialized")

    def _init_components(self):
        """初始化各个组件"""
        # 并行加载管理器
        if self.config.enable_parallel_loading:
            self.parallel_manager = ParallelLoadingManager(
                max_workers=self.config.max_workers
            )
        else:
            self.parallel_manager = None

        # 缓存管理器
        if self.config.enable_cache:
            cache_config = self.config.cache_config or CacheConfig()
            self.cache_manager = MultiLevelCache(cache_config)
        else:
            self.cache_manager = None

        # 质量监控器
        if self.config.enable_quality_monitor:
            self.quality_monitor = AdvancedQualityMonitor()
        else:
            self.quality_monitor = None

        # 性能监控器
        if self.config.enable_performance_monitor:
            from .performance_monitor import DataPerformanceMonitor
            self.performance_monitor = DataPerformanceMonitor()
        else:
            self.performance_monitor = None

        # 数据预加载器
        if self.config.enable_preload:
            from .data_preloader import DataPreloader, PreloadConfig
            preload_config = PreloadConfig(
                auto_preload_symbols=self.config.preload_symbols,
                auto_preload_days=self.config.preload_days
            )
            self.preloader = DataPreloader(preload_config)
        else:
            self.preloader = None

    async def optimize_data_loading(
        self,
        data_type: str,
        start_date: str,
        end_date: str,
        frequency: str = "1d",
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        优化数据加载

        Args:
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率
            symbols: 股票代码列表
            **kwargs: 其他参数

        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()
        cache_hit = False
        data_model = None
        error_message = None

        try:
            # 1. 检查缓存
            if self.cache_manager:
                cache_key = self._generate_cache_key(
                    data_type, start_date, end_date, frequency, symbols, **kwargs
                )
                cached_data = self.cache_manager.get(cache_key)

            if cached_data:
                cache_hit = True
                data_model = cached_data
                logger.info(f"Cache hit for key: {cache_key}")
            else:
                logger.info(f"Cache miss for key: {cache_key}")

            # 2. 如果缓存未命中，执行数据加载
            quality_metrics = None
            if not cache_hit:
                if self.parallel_manager and symbols and len(symbols) > 1:
                    # 使用并行加载
                    data_model = await self._parallel_load_data(
                        data_type, start_date, end_date, frequency, symbols, **kwargs
                    )
                else:
                    # 使用串行加载
                    data_model = await self._serial_load_data(
                        data_type, start_date, end_date, frequency, symbols, **kwargs
                    )

                # 3. 质量检查
            if self.quality_monitor and data_model is not None:
                quality_metrics = await self._check_data_quality(data_model.data)

                # 4. 缓存结果
            if self.cache_manager and data_model is not None:
                cache_key = self._generate_cache_key(
                    data_type, start_date, end_date, frequency, symbols, **kwargs
                )
                self.cache_manager.set(cache_key, data_model)
            else:
                # 缓存命中时，也需要进行质量检查
                if self.quality_monitor and data_model is not None:
                    quality_metrics = await self._check_data_quality(data_model.data)

            # 5. 更新性能统计
            load_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(load_time_ms, cache_hit)

            # 6. 记录性能指标
            if self.performance_monitor:
                self.performance_monitor.record_operation(
                    operation='optimize_data_loading',
                    duration_ms=load_time_ms,
                    success=True,
                    metadata={
                        'data_type': data_type,
                        'symbols_count': len(symbols) if symbols else 0,
                        'cache_hit': cache_hit
                    }
                )

            # 7. 生成性能指标
            performance_metrics = self._get_performance_metrics()

            return OptimizationResult(
                success=True,
                data_model=data_model,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                cache_hit=cache_hit,
                load_time_ms=load_time_ms
            )

        except Exception as e:
            load_time_ms = (time.time() - start_time) * 1000
            error_message = str(e)
            logger.error(f"Data loading failed: {error_message}")

            # 记录失败操作
        if self.performance_monitor:
            self.performance_monitor.record_operation(
                operation='optimize_data_loading',
                duration_ms=load_time_ms,
                success=False,
                error_message=error_message
            )

        return OptimizationResult(
            success=False,
            data_model=None,
            performance_metrics=self._get_performance_metrics(),
            quality_metrics=None,
            cache_hit=cache_hit,
            load_time_ms=load_time_ms,
            error_message=error_message
        )

    async def _parallel_load_data(
        self,
        data_type: str,
        start_date: str,
        end_date: str,
        frequency: str,
        symbols: List[str],
        **kwargs
    ) -> Optional[IDataModel]:
        """并行加载数据"""
        try:
            # 直接使用并行加载管理器加载所有数据
            results = self.parallel_manager.load_data_parallel(
                data_type, start_date, end_date, frequency, symbols, **kwargs
            )

            # 处理结果
            successful_results = []
            for symbol, result in results.items():
                if result is not None:
                    # 创建数据模型
                    pass
                    # 使用可序列化的数据模型

                    class SerializableDataModel:

                        def __init__(self, data, metadata):

                            self.data = data
                            self.metadata = metadata

                        def __reduce__(self):

                            return (SerializableDataModel, (self.data, self.metadata))

                    data_model = SerializableDataModel(
                        result,
                        {'symbol': symbol, 'source': 'parallel_loading'}
                    )
                    successful_results.append(data_model)

                if not successful_results:
                    raise Exception("All parallel loading tasks failed")

            # 合并结果
            return self._merge_data_models(successful_results)

        except Exception as e:
            logger.error(f"Parallel loading failed: {e}")
            raise

    async def _serial_load_data(
        self,
        data_type: str,
        start_date: str,
        end_date: str,
        frequency: str,
        symbols: Optional[List[str]],
        **kwargs
    ) -> Optional[IDataModel]:
        """串行加载数据"""
        try:
            # 使用数据管理器加载数据
            data_manager = DataManagerSingleton.get_instance()
            # 将symbols作为关键字参数传递
            kwargs_with_symbols = kwargs.copy()
            if symbols:
                kwargs_with_symbols['symbols'] = symbols

            # 使用数据管理器加载数据
            data_model = await data_manager.load_data(data_type, start_date, end_date, frequency, **kwargs_with_symbols)
            return data_model
        except Exception as e:
            logger.error(f"Serial loading failed: {e}")
            raise

    def _merge_data_models(self, data_models: List[IDataModel]) -> Optional[IDataModel]:
        """合并多个数据模型"""
        if not data_models:
            return None

        if len(data_models) == 1:
            return data_models[0]

        # 合并数据
        merged_data = pd.concat([model.data for model in data_models], ignore_index=True)

        # 创建合并后的数据模型

        class MergedDataModel:

            def __init__(self, data, metadata):

                self.data = data
                self.metadata = metadata

        merged_model = MergedDataModel(
            merged_data,
            {
                'source': 'merged',
                'original_models': len(data_models),
                'merged_at': datetime.now().isoformat()
            }
        )

        return merged_model

    async def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量"""
        if not self.quality_monitor:
            return {}

        try:
            # 检查是否是协程方法
            import inspect
            if inspect.iscoroutinefunction(self.quality_monitor.check_data_quality):
                quality_metrics = await self.quality_monitor.check_data_quality(data)
            else:
                quality_metrics = self.quality_monitor.check_data_quality(data)
            return quality_metrics
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {}

    def _generate_cache_key(


        self,
        data_type: str,
        start_date: str,
        end_date: str,
        frequency: str,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """生成缓存键"""
        key_parts = [
            data_type,
            start_date,
            end_date,
            frequency
        ]

        if symbols:
            key_parts.extend(sorted(symbols))

        # 添加其他参数，过滤掉不可哈希的类型
        for key, value in sorted(kwargs.items()):
            try:
                # 尝试将值转换为字符串
                if isinstance(value, (dict, list, set)):
                    # 对于复杂类型，使用类型名和长度
                    value_str = f"{type(value).__name__}_{len(value)}"
                else:
                    value_str = str(value)
                key_parts.append(f"{key}={value_str}")
            except (TypeError, ValueError):
                # 如果无法转换，跳过这个参数
                continue

        return "_".join(key_parts)

    def _update_performance_stats(self, load_time_ms: float, cache_hit: bool):
        """更新性能统计"""
        self.performance_stats['total_requests'] += 1

        if cache_hit:
            self.performance_stats['cache_hits'] += 1
        else:
            self.performance_stats['cache_misses'] += 1

        # 更新平均加载时间
        current_avg = self.performance_stats['avg_load_time_ms']
        total_requests = self.performance_stats['total_requests']

        self.performance_stats['avg_load_time_ms'] = (
            (current_avg * (total_requests - 1) + load_time_ms) / total_requests
        )

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        total_requests = self.performance_stats['total_requests']

        if total_requests == 0:
            return {
                'total_requests': 0,
                'cache_hit_rate': 0.0,
                'avg_load_time_ms': 0.0
            }

        cache_hit_rate = (
            self.performance_stats['cache_hits'] / total_requests
        )

        return {
            'total_requests': total_requests,
            'cache_hits': self.performance_stats['cache_hits'],
            'cache_misses': self.performance_stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'avg_load_time_ms': self.performance_stats['avg_load_time_ms']
        }

    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        performance_metrics = self._get_performance_metrics()

        report = {
            'optimization_config': {
                'max_workers': self.config.max_workers,
                'enable_parallel_loading': self.config.enable_parallel_loading,
                'enable_cache': self.config.enable_cache,
                'enable_quality_monitor': self.config.enable_quality_monitor,
                'enable_performance_monitor': self.config.enable_performance_monitor,
                'enable_preload': self.config.enable_preload
            },
            'performance_metrics': performance_metrics,
            'component_status': {
                'parallel_manager': self.parallel_manager is not None,
                'cache_manager': self.cache_manager is not None,
                'quality_monitor': self.quality_monitor is not None,
                'performance_monitor': self.performance_monitor is not None,
                'preloader': self.preloader is not None
            }
        }

        return report

    def cleanup(self):
        """清理资源"""
        if self.parallel_manager:
            self.parallel_manager.shutdown()

        if self.cache_manager:
            self.cache_manager.cleanup()

        if self.preloader:
            self.preloader.shutdown()

        logger.info("DataOptimizer cleanup completed")


async def optimize_data_loading(
    data_type: str,
    start_date: str,
    end_date: str,
    frequency: str = "1d",
    symbols: Optional[List[str]] = None,
    config: Optional[OptimizationConfig] = None,
    **kwargs
) -> OptimizationResult:
    """
    便捷函数：优化数据加载

    Args:
        data_type: 数据类型
        start_date: 开始日期
        end_date: 结束日期
        frequency: 频率
        symbols: 股票代码列表
        config: 优化配置
        **kwargs: 其他参数

    Returns:
        OptimizationResult: 优化结果
    """
    optimizer = DataOptimizer(config)

    try:
        result = await optimizer.optimize_data_loading(
            data_type, start_date, end_date, frequency, symbols, **kwargs
        )
        return result
    finally:
        optimizer.cleanup()


# 测试函数
async def test_data_optimizer():
    """测试数据优化器"""
    config = OptimizationConfig(
        max_workers=2,
        enable_parallel_loading=True,
        enable_cache=True,
        enable_quality_monitor=True,
        enable_performance_monitor=True
    )

    optimizer = DataOptimizer(config)

    try:
        result = await optimizer.optimize_data_loading(
            data_type='stock',
            start_date='2024 - 01 - 01',
            end_date='2024 - 01 - 01',
            frequency='1d',
            symbols=['600519.SH', '000858.SZ']
        )

        print(f"Optimization result: {result}")
        print(f"Optimization report: {optimizer.get_optimization_report()}")

    finally:
        optimizer.cleanup()


if __name__ == '__main__':
    asyncio.run(test_data_optimizer())
