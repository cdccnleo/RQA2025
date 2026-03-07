import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式特征处理器

from src.infrastructure.logging.core.unified_logger import get_unified_logger
支持多进程并行处理大规模特征数据。
"""

import pandas as pd
from typing import Dict, List, Optional, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path
import pickle

from .base_processor import BaseFeatureProcessor
from ..core.feature_config import FeatureConfig
from src.infrastructure.logging.core.unified_logger import get_unified_logger


logger = logging.getLogger(__name__)


class DistributedFeatureProcessor:

    """分布式特征处理器"""

    def __init__(self, max_workers: Optional[int] = None, chunk_size: int = 1000):
        """
        初始化分布式处理器

        Args:
            max_workers: 最大工作进程数，默认为CPU核心数
            chunk_size: 数据块大小
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.chunk_size = chunk_size

        # 初始化日志记录器
        self.logger = get_unified_logger('__name__')

        # 性能统计
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'chunks_processed': 0,
            'errors': 0
        }

    def process_in_parallel(self, data: pd.DataFrame, processor: BaseFeatureProcessor,


                            config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        并行处理特征

        Args:
            data: 输入数据
            processor: 特征处理器
            config: 配置

        Returns:
            处理后的特征数据
        """
        if data.empty:
            return pd.DataFrame()

        start_time = time.time()

        try:
            # 数据分块
            chunks = self._split_data(data)
            self.logger.info(f"数据分块完成，共 {len(chunks)} 个块")

            # 并行处理
            results = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交任务
                future_to_chunk = {
                    executor.submit(self._process_chunk, chunk, processor, config): i
                    for i, chunk in enumerate(chunks)
                }

                # 收集结果
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            results.append(result)
                        self.stats['chunks_processed'] += 1
                    except Exception as e:
                        self.stats['errors'] += 1
                        self.logger.error(f"处理块 {chunk_index} 失败: {e}")

            # 合并结果
            if results:
                final_result = pd.concat(results, ignore_index=True)
                self.stats['total_processed'] = len(final_result)
                self.stats['total_time'] = time.time() - start_time

                self.logger.info(
                    f"并行处理完成，处理了 {len(final_result)} 行数据，耗时 {self.stats['total_time']:.2f}秒")
                return final_result
            else:
                self.logger.warning("没有成功处理的数据块")
                return pd.DataFrame()

        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"并行处理失败: {e}")
            return pd.DataFrame()

    def _split_data(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        将数据分割成块

        Args:
            data: 输入数据

        Returns:
            数据块列表
        """
        chunks = []
        total_rows = len(data)

        for i in range(0, total_rows, self.chunk_size):
            end_idx = min(i + self.chunk_size, total_rows)
            chunk = data.iloc[i:end_idx].copy()
            chunks.append(chunk)

        return chunks

    def _process_chunk(self, chunk: pd.DataFrame, processor: BaseFeatureProcessor,


                       config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        处理单个数据块

        Args:
            chunk: 数据块
            processor: 处理器
            config: 配置

        Returns:
            处理后的数据块
        """
        try:
            # 这里需要序列化处理器，实际项目中可能需要更复杂的处理
            # 简化实现：直接处理
            return processor.process(chunk, config)
        except Exception as e:
            self.logger.error(f"处理数据块失败: {e}")
            return pd.DataFrame()

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.stats.copy()
        if stats['total_time'] > 0:
            stats['throughput'] = stats['total_processed'] / stats['total_time']
        else:
            stats['throughput'] = 0

        stats['success_rate'] = (stats['chunks_processed'] - stats['errors']
                                 ) / max(stats['chunks_processed'], 1)
        return stats

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'chunks_processed': 0,
            'errors': 0
        }


class MemoryOptimizedProcessor:

    """内存优化处理器"""

    def __init__(self, max_memory_mb: int = 1024):
        """
        初始化内存优化处理器

        Args:
            max_memory_mb: 最大内存使用量(MB)
        """
        self.max_memory_mb = max_memory_mb

        # 初始化日志记录器
        self.logger = get_unified_logger('__name__')

    def process_with_memory_optimization(self, data: pd.DataFrame, processor: BaseFeatureProcessor,


                                         config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        内存优化处理

        Args:
            data: 输入数据
            processor: 处理器
            config: 配置

        Returns:
            处理后的数据
        """
        if data.empty:
            return pd.DataFrame()

        try:
            # 估算内存使用
            estimated_memory = self._estimate_memory_usage(data)

            if estimated_memory > self.max_memory_mb:
                self.logger.info(
                    f"数据内存使用 {estimated_memory:.1f}MB，超过限制 {self.max_memory_mb}MB，启用内存优化")
                return self._process_in_batches(data, processor, config)
            else:
                self.logger.info(f"数据内存使用 {estimated_memory:.1f}MB，在限制范围内，直接处理")
                return processor.process(data, config)

        except Exception as e:
            self.logger.error(f"内存优化处理失败: {e}")
            return pd.DataFrame()

    def _estimate_memory_usage(self, data: pd.DataFrame) -> float:
        """估算内存使用量(MB)"""
        try:
            # 简单的内存估算
            memory_usage = data.memory_usage(deep=True).sum()
            return memory_usage / (1024 * 1024)  # 转换为MB
        except BaseException:
            # 如果无法估算，使用默认值
            return len(data) * len(data.columns) * 8 / (1024 * 1024)  # 假设每列8字节

    def _process_in_batches(self, data: pd.DataFrame, processor: BaseFeatureProcessor,


                            config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        分批处理数据

        Args:
            data: 输入数据
            processor: 处理器
            config: 配置

        Returns:
            处理后的数据
        """
        results = []
        total_rows = len(data)
        batch_size = max(1, total_rows // 10)  # 分成10批

        for i in range(0, total_rows, batch_size):
            end_idx = min(i + batch_size, total_rows)
            batch = data.iloc[i:end_idx].copy()

            try:
                batch_result = processor.process(batch, config)
                if not batch_result.empty:
                    results.append(batch_result)

                self.logger.info(
                    f"处理批次 {i // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size}")

            except Exception as e:
                self.logger.error(f"处理批次 {i // batch_size + 1} 失败: {e}")
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()


class CachingProcessor:

    """缓存处理器"""

    def __init__(self, cache_dir: str = "./feature_cache", max_cache_size_mb: int = 512):
        """
        初始化缓存处理器

        Args:
            cache_dir: 缓存目录
            max_cache_size_mb: 最大缓存大小(MB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb

        # 缓存统计
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'size_mb': 0.0
        }

        # 初始化日志记录器
        self.logger = get_unified_logger('__name__')

    def process_with_cache(self, data: pd.DataFrame, processor: BaseFeatureProcessor,


                           config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        带缓存的处理

        Args:
            data: 输入数据
            processor: 处理器
            config: 配置

        Returns:
            处理后的数据
        """
        if data.empty:
            return pd.DataFrame()

        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(data, processor, config)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            # 检查缓存
            if cache_file.exists():
                try:
                    cached_result = self._load_from_cache(cache_file)
                    if cached_result is not None:
                        self.cache_stats['hits'] += 1
                        self.logger.info(f"从缓存加载结果: {cache_key}")
                        return cached_result
                except Exception as e:
                    self.logger.warning(f"加载缓存失败: {e}")

            # 缓存未命中，执行处理
            self.cache_stats['misses'] += 1
            result = processor.process(data, config)

            # 保存到缓存
            if not result.empty:
                self._save_to_cache(cache_file, result)
                self.cache_stats['saves'] += 1
                self.logger.info(f"结果已缓存: {cache_key}")

            return result

        except Exception as e:
            self.logger.error(f"缓存处理失败: {e}")
            return pd.DataFrame()

    def _generate_cache_key(self, data: pd.DataFrame, processor: BaseFeatureProcessor,


                            config: Optional[FeatureConfig] = None) -> str:
        """生成缓存键"""
        import hashlib

        # 组合数据特征
        data_hash = hashlib.md5()
        data_hash.update(str(data.shape).encode())
        data_hash.update(str(data.dtypes).encode())
        data_hash.update(str(data.iloc[:10].values).encode())  # 前10行数据

        # 处理器信息
        processor_info = f"{processor.__class__.__name__}_{processor.get_processor_type()}"

        # 配置信息
        config_info = str(config.to_dict() if hasattr(config, 'to_dict') else config)

        # 组合所有信息
        combined = f"{data_hash.hexdigest()}_{processor_info}_{hashlib.md5(config_info.encode()).hexdigest()}"
        return combined[:32]  # 限制长度

    def _save_to_cache(self, cache_file: Path, result: pd.DataFrame) -> None:
        """保存到缓存"""
        try:
            # 检查缓存大小
            if self.cache_stats['size_mb'] > self.max_cache_size_mb:
                self._cleanup_cache()

            # 保存数据
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            # 更新缓存大小
            self.cache_stats['size_mb'] += cache_file.stat().st_size / (1024 * 1024)

        except Exception as e:
            self.logger.error(f"保存缓存失败: {e}")

    def _load_from_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """从缓存加载"""
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"加载缓存失败: {e}")
            return None

    def _cleanup_cache(self) -> None:
        """清理缓存"""
        try:
            # 简单的LRU清理：删除最旧的文件
            cache_files = list(self.cache_dir.glob("*.pkl"))
            if len(cache_files) > 10:  # 保留最新的10个文件
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in cache_files[:-10]:
                    old_file.unlink()
                    self.logger.info(f"清理缓存文件: {old_file.name}")
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = self.cache_stats.copy()
        if stats['hits'] + stats['misses'] > 0:
            stats['hit_rate'] = stats['hits'] / (stats['hits'] + stats['misses'])
        else:
            stats['hit_rate'] = 0.0
        return stats


# 导出主要类
__all__ = [
    'DistributedFeatureProcessor',
    'MemoryOptimizedProcessor',
    'CachingProcessor'
]
