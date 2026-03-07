#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层性能优化器

实现智能性能优化功能：
- 智能缓存策略优化
- 并行处理算法优化
- 数据压缩和序列化优化
- 内存使用优化
- 动态资源分配
- 性能监控和调优
"""

import time
import threading
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import pickle
import gzip
import zlib
import json
from datetime import datetime, timedelta
import logging

from src.data.cache.enhanced_cache_manager import EnhancedCacheManager
from src.data.parallel.enhanced_parallel_loader import EnhancedParallelLoadingManager

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):

    """优化策略枚举"""
    CACHE_FIRST = "cache_first"           # 缓存优先
    PARALLEL_FIRST = "parallel_first"     # 并行优先
    MEMORY_OPTIMIZED = "memory_optimized"  # 内存优化
    BALANCED = "balanced"                  # 平衡策略
    AGGRESSIVE = "aggressive"              # 激进策略


class CompressionAlgorithm(Enum):

    """压缩算法枚举"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"


@dataclass
class PerformanceMetrics:

    """性能指标数据类"""
    cache_hit_rate: float
    memory_usage: float
    cpu_usage: float
    load_time: float
    throughput: float
    compression_ratio: float
    parallel_efficiency: float
    timestamp: datetime


@dataclass
class OptimizationConfig:

    """优化配置数据类"""
    approach: OptimizationStrategy
    max_memory_usage: float = 0.8  # 最大内存使用率
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB
    compression_enabled: bool = True
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    parallel_workers: Optional[int] = None
    batch_size: int = 100
    enable_auto_tuning: bool = True


class DataPerformanceOptimizer:

    """
    数据性能优化器

    特性：
    - 智能缓存策略优化
    - 并行处理算法优化
    - 数据压缩和序列化优化
    - 内存使用优化
    - 动态资源分配
    - 性能监控和调优
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化性能优化器

        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig(approach=OptimizationStrategy.BALANCED)

        # 初始化组件
        self.cache_manager = EnhancedCacheManager()
        self.parallel_manager = EnhancedParallelLoadingManager()

        # 性能监控
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_stats = {
            'cache_optimizations': 0,
            'parallel_optimizations': 0,
            'compression_optimizations': 0,
            'memory_optimizations': 0,
            'total_optimizations': 0
        }

        # 自动调优
        self.auto_tuning_enabled = self.config.enable_auto_tuning
        self._tuning_thread = None
        self._stop_tuning = False

        if self.auto_tuning_enabled:
            self._start_auto_tuning()

        logger.info("DataPerformanceOptimizer initialized")

    def optimize_data_loading(self,


                              data_source: str,
                              start_date: str,
                              end_date: str,
                              **kwargs) -> pd.DataFrame:
        """
        优化数据加载过程

        Args:
            data_source: 数据源
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数

        Returns:
            pd.DataFrame: 优化后的数据
        """
        # 验证输入参数
        if not data_source or not data_source.strip():
            raise ValueError("Data source cannot be empty")
        if not start_date or not start_date.strip():
            raise ValueError("Start date cannot be empty")
        if not end_date or not end_date.strip():
            raise ValueError("End date cannot be empty")

        start_time = time.time()

        # 1. 缓存优化
        cached_data = self._try_cache_loading(data_source, start_date, end_date)
        if cached_data is not None:
            logger.info(f"Cache hit for {data_source}")
            return cached_data

        # 2. 并行加载优化
        data = self._parallel_data_loading(data_source, start_date, end_date, **kwargs)

        # 3. 数据压缩优化
        if self.config.compression_enabled:
            data = self._optimize_data_compression(data)

        # 4. 内存优化
        data = self._optimize_memory_usage(data)

        # 5. 缓存存储
        self._cache_data(data, data_source, start_date, end_date)

        # 6. 性能监控
        load_time = time.time() - start_time
        data_size = len(data) if data is not None else 0
        self._record_performance_metrics(load_time, data_size)

        logger.info(f"Data loading optimized for {data_source}, time: {load_time:.2f}s")
        return data

    def _try_cache_loading(self, data_source: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """尝试从缓存加载数据"""
        cache_key = f"{data_source}_{start_date}_{end_date}"
        try:
            cached_data = self.cache_manager.get(cache_key)

            if cached_data is not None:
                # 解压缩数据
                if isinstance(cached_data, bytes):
                    cached_data = self._decompress_data(cached_data)

                return cached_data
        except Exception as e:
            logger.error(f"Cache loading error: {e}")

        return None

    def _parallel_data_loading(self, data_source: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """并行数据加载优化"""
        # 根据数据源类型选择最优的并行策略
        if self.config.approach == OptimizationStrategy.PARALLEL_FIRST:
            return self._aggressive_parallel_loading(data_source, start_date, end_date, **kwargs)
        else:
            return self._balanced_parallel_loading(data_source, start_date, end_date, **kwargs)

    def _aggressive_parallel_loading(self, data_source: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """激进并行加载策略"""
        # 将时间范围分割成多个小批次
        date_ranges = self._split_date_range(start_date, end_date, batch_size=5)

        # 创建并行任务
        tasks = []
        for start, end in date_ranges:
            task = {
                'data_source': data_source,
                'start_date': start,
                'end_date': end,
                'kwargs': kwargs
            }
            tasks.append(task)

        # 并行执行
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers or 8) as processor:
            futures = [processor.submit(self._load_single_batch, task) for task in tasks]
            results = []

        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"Parallel loading error: {e}")

        # 合并结果
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    def _balanced_parallel_loading(self, data_source: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """平衡并行加载策略"""
        # 使用较少的并行度，避免资源竞争
        workers = min(4, self.config.parallel_workers or 4)

        with ThreadPoolExecutor(max_workers=workers) as processor:
            future = processor.submit(self._load_single_batch, {
                'data_source': data_source,
                'start_date': start_date,
                'end_date': end_date,
                'kwargs': kwargs
            })

        try:
            return future.result()
        except ValueError as e:
            # 对于日期解析错误，重新抛出ValueError
            logger.error(f"Balanced loading error: {e}")
            raise
        except Exception as e:
            logger.error(f"Balanced loading error: {e}")
            return pd.DataFrame()

    def _load_single_batch(self, task: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """加载单个批次数据"""
        try:
            # 这里应该调用实际的数据加载器
            # 暂时返回模拟数据
            return pd.DataFrame({
                'date': pd.date_range(task['start_date'], task['end_date']),
                'value': np.secrets.randn(len(pd.date_range(task['start_date'], task['end_date'])))
            })
        except ValueError as e:
            # 对于日期解析错误，重新抛出ValueError
            logger.error(f"Batch loading error: {e}")
            raise ValueError(f"Invalid date format: {e}")
        except Exception as e:
            logger.error(f"Batch loading error: {e}")
            return None

    def _optimize_data_compression(self, data: pd.DataFrame) -> pd.DataFrame:
        """优化数据压缩"""
        if not self.config.compression_enabled:
            return data

        # 检查数据是否为空
        if data is None or data.empty:
            return data

        # 根据数据类型选择最优压缩算法
        compression_algo = self._select_optimal_compression(data)

        # 压缩数据
        compressed_data = self._compress_data(data, compression_algo)

        # 记录压缩统计
        original_size = len(data.to_string())
        compressed_size = len(compressed_data)
        compression_ratio = 1 - (compressed_size / original_size)

        self.optimization_stats['compression_optimizations'] += 1
        logger.info(
            f"Data compressed with {compression_algo.value}, ratio: {compression_ratio:.2%}")

        return data  # 返回原始数据，压缩版本用于缓存

    def _select_optimal_compression(self, data: pd.DataFrame) -> CompressionAlgorithm:
        """选择最优压缩算法"""
        # 检查数据是否为空
        if data is None or data.empty:
            return CompressionAlgorithm.NONE

        # 根据数据特征选择压缩算法
        data_size = len(data)
        data_types = data.dtypes.value_counts()

        if data_size < 1000:
            return CompressionAlgorithm.NONE
        elif data_types.get('object', 0) > len(data_types) * 0.5:
            return CompressionAlgorithm.GZIP
        else:
            return CompressionAlgorithm.ZLIB

    def _compress_data(self, data: pd.DataFrame, algorithm: CompressionAlgorithm) -> bytes:
        """压缩数据"""
        data_bytes = pickle.dumps(data)

        if algorithm == CompressionAlgorithm.NONE:
            return data_bytes
        elif algorithm == CompressionAlgorithm.GZIP:
            try:
                return gzip.compress(data_bytes)
            except Exception as e:
                logger.error(f"GZIP compression error: {e}")
                return data_bytes
        elif algorithm == CompressionAlgorithm.ZLIB:
            try:
                return zlib.compress(data_bytes)
            except Exception as e:
                logger.error(f"ZLIB compression error: {e}")
                return data_bytes
        else:
            return data_bytes

    def _decompress_data(self, compressed_data: bytes) -> pd.DataFrame:
        """解压缩数据"""
        try:
            # 尝试不同的解压缩方法
            for algorithm in [CompressionAlgorithm.GZIP, CompressionAlgorithm.ZLIB, CompressionAlgorithm.NONE]:
                try:
                    if algorithm == CompressionAlgorithm.GZIP:
                        decompressed = gzip.decompress(compressed_data)
                    elif algorithm == CompressionAlgorithm.ZLIB:
                        decompressed = zlib.decompress(compressed_data)
                    else:
                        decompressed = compressed_data

                    return pickle.loads(decompressed)
                except Exception:
                    continue

            # 如果都失败了，返回空DataFrame
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            return pd.DataFrame()

    def _optimize_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """优化内存使用"""
        # 检查内存使用情况
        memory_usage = psutil.virtual_memory().percent

        if memory_usage > self.config.max_memory_usage * 100:
            # 内存使用过高，进行优化
            data = self._reduce_memory_usage(data)
            self.optimization_stats['memory_optimizations'] += 1

        return data

    def _reduce_memory_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """减少内存使用"""
        # 1. 优化数据类型
        for col in data.columns:
            if data[col].dtype == 'object':
                # 尝试转换为更小的数据类型
                try:
                    data[col] = pd.to_numeric(data[col])
                except (ValueError, TypeError):
                    pass

        # 2. 删除重复数据
        data = data.drop_duplicates()

        # 3. 强制垃圾回收
        gc.collect()

        return data

    def _cache_data(self, data: pd.DataFrame, data_source: str, start_date: str, end_date: str):
        """缓存数据"""
        cache_key = f"{data_source}_{start_date}_{end_date}"

        # 压缩数据用于缓存
        if self.config.compression_enabled:
            compressed_data = self._compress_data(data, self.config.compression_algorithm)
            self.cache_manager.set(cache_key, compressed_data, expire=3600)
        else:
            self.cache_manager.set(cache_key, data, expire=3600)

        self.optimization_stats['cache_optimizations'] += 1

    def _split_date_range(self, start_date: str, end_date: str, batch_size: int = 5) -> List[tuple]:
        """分割日期范围"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        date_ranges = []
        current = start

        while current < end:
            batch_end = min(current + timedelta(days=batch_size), end)
            date_ranges.append((current.strftime('%Y-%m-%d'), batch_end.strftime('%Y-%m-%d')))
            current = batch_end

        return date_ranges

    def _record_performance_metrics(self, load_time: float, data_size: int):
        """记录性能指标"""
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()

        # 计算缓存命中率
        cache_stats = self.cache_manager.get_stats()
        cache_hit_rate = cache_stats.get('hit_rate', 0.0)

        # 计算吞吐量
        throughput = data_size / load_time if load_time > 0 else 0

        metrics = PerformanceMetrics(
            cache_hit_rate=cache_hit_rate,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            load_time=load_time,
            throughput=throughput,
            compression_ratio=0.0,  # 需要从压缩统计中获取
            parallel_efficiency=0.0,  # 需要从并行统计中获取
            timestamp=datetime.now()
        )

        self.performance_history.append(metrics)

        # 保持历史记录在合理范围内
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

    def _start_auto_tuning(self):
        """启动自动调优"""

        def tuning_worker():

            while not self._stop_tuning:
                try:
                    self._auto_tune_performance()
                    time.sleep(60)  # 每分钟调优一次
                except Exception as e:
                    logger.error(f"Auto tuning error: {e}")
                    time.sleep(300)  # 出错后等待5分钟

        self._tuning_thread = threading.Thread(target=tuning_worker, daemon=True)
        self._tuning_thread.start()

    def _auto_tune_performance(self):
        """自动调优性能"""
        if len(self.performance_history) < 10:
            return

        # 分析最近的性能指标
        recent_metrics = self.performance_history[-10:]

        # 计算平均指标
        avg_load_time = np.mean([m.load_time for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
        avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])

        # 根据性能指标调整策略
        if avg_load_time > 5.0:  # 加载时间过长
            self._optimize_loading_strategy()
        elif avg_memory_usage > 80:  # 内存使用过高
            self._optimize_memory_strategy()
        elif avg_cache_hit_rate < 0.5:  # 缓存命中率过低
            self._optimize_cache_strategy()

    def optimize_loading_approach(self):
        """优化加载策略"""
        logger.info("Optimizing loading approach")
        # 增加并行度或调整批次大小
        if self.config.batch_size < 200:
            self.config.batch_size = min(200, self.config.batch_size * 2)

    def optimize_memory_approach(self):
        """优化内存策略"""
        logger.info("Optimizing memory approach")
        # 降低内存使用限制或启用更激进的压缩
        self.config.max_memory_usage = max(0.6, self.config.max_memory_usage - 0.1)

    def optimize_cache_approach(self):
        """优化缓存策略"""
        logger.info("Optimizing cache approach")
        # 增加缓存大小或调整缓存策略
        self.config.max_cache_size = min(2 * 1024 * 1024 * 1024, self.config.max_cache_size * 2)

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_history:
            return {}

        recent_metrics = self.performance_history[-50:]  # 最近50次

        return {
            'average_load_time': np.mean([m.load_time for m in recent_metrics]),
            'average_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'average_cache_hit_rate': np.mean([m.cache_hit_rate for m in recent_metrics]),
            'average_throughput': np.mean([m.throughput for m in recent_metrics]),
            'optimization_stats': self.optimization_stats,
            'total_operations': len(self.performance_history)
        }

    def get_performance_history(self) -> List[PerformanceMetrics]:
        """获取性能历史记录"""
        return self.performance_history.copy()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return self.optimization_stats.copy()

    def update_config(self, new_config: Union[OptimizationConfig, Dict[str, Any]]) -> None:
        """更新优化配置"""
        if isinstance(new_config, dict):
            # 验证配置值
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    # 验证特定配置项
                    if key == 'max_memory_usage' and (value <= 0 or value > 1.0):
                        raise ValueError(f"max_memory_usage must be between 0 and 1, got {value}")
                    elif key == 'max_cache_size' and value < 0:
                        raise ValueError(f"max_cache_size must be non - negative, got {value}")
                    elif key == 'parallel_workers' and value is not None and value < 1:
                        raise ValueError(f"parallel_workers must be at least 1, got {value}")
                    elif key == 'batch_size' and value < 1:
                        raise ValueError(f"batch_size must be at least 1, got {value}")

                    # 更新配置
                    setattr(self.config, key, value)
        else:
            # 如果是OptimizationConfig对象，直接替换
            self.config = new_config

        logger.info("Optimization config updated")

        # 如果自动调优状态改变，重新启动或停止
        if hasattr(new_config, 'enable_auto_tuning') and new_config.enable_auto_tuning != self.auto_tuning_enabled:
            if new_config.enable_auto_tuning:
                self._start_auto_tuning()
            else:
                self.stop_auto_tuning()
            self.auto_tuning_enabled = new_config.enable_auto_tuning

    def stop_auto_tuning(self) -> None:
        """停止自动调优"""
        self._stop_tuning = True
        if self._tuning_thread and self._tuning_thread.is_alive():
            self._tuning_thread.join(timeout=5)
        logger.info("Auto tuning stopped")

    def cleanup_resources(self) -> None:
        """清理资源"""
        self.shutdown()
        # 清理性能历史记录
        self.performance_history.clear()
        # 重置统计信息
        self.optimization_stats = {
            'cache_optimizations': 0,
            'parallel_optimizations': 0,
            'compression_optimizations': 0,
            'memory_optimizations': 0,
            'total_optimizations': 0
        }
        logger.info("Resources cleaned up")

    def export_performance_report(self, file_path: str = None) -> str:
        """导出性能报告到文件"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y % m % d_ % H % M % S")
            file_path = f"performance_report_{timestamp}.json"

        report_data = self.get_performance_report()

        # 添加配置信息
        report_data['config'] = {
            "approach": self.config.approach.value,
            'max_memory_usage': self.config.max_memory_usage,
            'max_cache_size': self.config.max_cache_size,
            'compression_enabled': self.config.compression_enabled,
            'compression_algorithm': self.config.compression_algorithm.value,
            'parallel_workers': self.config.parallel_workers,
            'batch_size': self.config.batch_size,
            'enable_auto_tuning': self.config.enable_auto_tuning
        }

        # 添加性能历史记录
        report_data['performance_history'] = [
            {
                'timestamp': m.timestamp.isoformat(),
                'cache_hit_rate': m.cache_hit_rate,
                'memory_usage': m.memory_usage,
                'cpu_usage': m.cpu_usage,
                'load_time': m.load_time,
                'throughput': m.throughput,
                'compression_ratio': m.compression_ratio,
                'parallel_efficiency': m.parallel_efficiency
            }
            for m in self.performance_history
        ]

        try:
            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Performance report exported to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            raise

    def shutdown(self):
        """关闭优化器"""
        self._stop_tuning = True
        if self._tuning_thread:
            self._tuning_thread.join(timeout=5)

        self.cache_manager.shutdown()
        self.parallel_manager.shutdown()

        logger.info("DataPerformanceOptimizer shutdown")


# 工厂函数

def create_performance_optimizer(config: Optional[OptimizationConfig] = None) -> DataPerformanceOptimizer:
    """创建性能优化器实例"""
    return DataPerformanceOptimizer(config)


# 便捷函数

def optimize_data_loading(data_source: str, start_date: str, end_date: str,


                          config: Optional[OptimizationConfig] = None, **kwargs) -> pd.DataFrame:
    """便捷的数据加载优化函数"""
    optimizer = create_performance_optimizer(config)
    try:
        return optimizer.optimize_data_loading(data_source, start_date, end_date, **kwargs)
    finally:
        optimizer.shutdown()
