#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据压缩引擎

功能：
- 支持多种压缩算法（LZ4、Snappy、Zstandard）
- 列式存储压缩（Parquet格式）
- 自适应压缩策略
- 压缩性能监控

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """压缩算法类型"""
    LZ4 = "lz4"           # 高速压缩
    SNAPPY = "snappy"    # Google Snappy
    ZSTD = "zstd"        # Zstandard（高压缩比）
    GZIP = "gzip"        # 标准gzip
    BROTLI = "brotli"    # Brotli（Web优化）
    NONE = "none"        # 无压缩


class StorageFormat(Enum):
    """存储格式"""
    PARQUET = "parquet"  # Apache Parquet（列式）
    FEATHER = "feather"  # Apache Feather
    HDF5 = "hdf5"        # HDF5格式
    CSV = "csv"          # CSV格式
    PICKLE = "pickle"    # Python Pickle


@dataclass
class CompressionResult:
    """压缩结果"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: float
    algorithm: CompressionAlgorithm
    format: StorageFormat


@dataclass
class CompressionStats:
    """压缩统计"""
    total_operations: int = 0
    total_bytes_saved: int = 0
    avg_compression_ratio: float = 0.0
    avg_compression_time_ms: float = 0.0


class AdvancedCompressionEngine:
    """
    高级数据压缩引擎
    
    提供高效的数据压缩和存储功能
    """
    
    def __init__(self):
        """初始化压缩引擎"""
        self._stats = CompressionStats()
        self._algorithm_performance: Dict[CompressionAlgorithm, List[float]] = {}
        
        # 检查可用的压缩库
        self._available_algorithms = self._check_available_algorithms()
        
        logger.info(f"高级压缩引擎初始化完成，可用算法: {list(self._available_algorithms.keys())}")
    
    def _check_available_algorithms(self) -> Dict[CompressionAlgorithm, bool]:
        """
        检查可用的压缩算法
        
        Returns:
            算法可用性字典
        """
        available = {}
        
        # 检查LZ4
        try:
            import lz4.frame
            available[CompressionAlgorithm.LZ4] = True
        except ImportError:
            available[CompressionAlgorithm.LZ4] = False
        
        # 检查Snappy
        try:
            import snappy
            available[CompressionAlgorithm.SNAPPY] = True
        except ImportError:
            available[CompressionAlgorithm.SNAPPY] = False
        
        # 检查Zstandard
        try:
            import zstandard
            available[CompressionAlgorithm.ZSTD] = True
        except ImportError:
            available[CompressionAlgorithm.ZSTD] = False
        
        # 检查Brotli
        try:
            import brotli
            available[CompressionAlgorithm.BROTLI] = True
        except ImportError:
            available[CompressionAlgorithm.BROTLI] = False
        
        # Gzip总是可用
        import gzip
        available[CompressionAlgorithm.GZIP] = True
        
        return available
    
    def compress_dataframe(
        self,
        df: pd.DataFrame,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD,
        format: StorageFormat = StorageFormat.PARQUET,
        file_path: Optional[str] = None
    ) -> CompressionResult:
        """
        压缩DataFrame
        
        Args:
            df: 待压缩的DataFrame
            algorithm: 压缩算法
            format: 存储格式
            file_path: 输出文件路径（可选）
            
        Returns:
            压缩结果
        """
        start_time = time.time()
        
        # 计算原始大小
        original_size = df.memory_usage(deep=True).sum()
        
        # 根据格式选择压缩方法
        if format == StorageFormat.PARQUET:
            compressed_size = self._compress_to_parquet(df, algorithm, file_path)
        elif format == StorageFormat.FEATHER:
            compressed_size = self._compress_to_feather(df, file_path)
        elif format == StorageFormat.HDF5:
            compressed_size = self._compress_to_hdf5(df, algorithm, file_path)
        elif format == StorageFormat.PICKLE:
            compressed_size = self._compress_to_pickle(df, algorithm, file_path)
        else:
            raise ValueError(f"不支持的存储格式: {format}")
        
        compression_time = (time.time() - start_time) * 1000
        
        # 测试解压时间
        decompression_time = self._benchmark_decompression(file_path, format)
        
        # 计算压缩比
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time,
            decompression_time_ms=decompression_time,
            algorithm=algorithm,
            format=format
        )
        
        # 更新统计
        self._update_stats(result)
        
        logger.info(f"DataFrame压缩完成: {original_size} -> {compressed_size} "
                   f"(压缩比: {compression_ratio:.2f}x)")
        
        return result
    
    def _compress_to_parquet(
        self,
        df: pd.DataFrame,
        algorithm: CompressionAlgorithm,
        file_path: Optional[str]
    ) -> int:
        """
        压缩为Parquet格式
        
        Args:
            df: DataFrame
            algorithm: 压缩算法
            file_path: 文件路径
            
        Returns:
            压缩后大小
        """
        if file_path is None:
            import tempfile
            file_path = tempfile.mktemp(suffix='.parquet')
        
        # 映射算法名称
        compression_map = {
            CompressionAlgorithm.LZ4: 'lz4',
            CompressionAlgorithm.SNAPPY: 'snappy',
            CompressionAlgorithm.ZSTD: 'zstd',
            CompressionAlgorithm.GZIP: 'gzip',
            CompressionAlgorithm.BROTLI: 'brotli',
            CompressionAlgorithm.NONE: 'none'
        }
        
        compression = compression_map.get(algorithm, 'zstd')
        
        # 写入Parquet
        df.to_parquet(
            file_path,
            compression=compression,
            engine='pyarrow'
        )
        
        return Path(file_path).stat().st_size
    
    def _compress_to_feather(
        self,
        df: pd.DataFrame,
        file_path: Optional[str]
    ) -> int:
        """
        压缩为Feather格式
        
        Args:
            df: DataFrame
            file_path: 文件路径
            
        Returns:
            压缩后大小
        """
        if file_path is None:
            import tempfile
            file_path = tempfile.mktemp(suffix='.feather')
        
        df.to_feather(file_path)
        
        return Path(file_path).stat().st_size
    
    def _compress_to_hdf5(
        self,
        df: pd.DataFrame,
        algorithm: CompressionAlgorithm,
        file_path: Optional[str]
    ) -> int:
        """
        压缩为HDF5格式
        
        Args:
            df: DataFrame
            algorithm: 压缩算法
            file_path: 文件路径
            
        Returns:
            压缩后大小
        """
        if file_path is None:
            import tempfile
            file_path = tempfile.mktemp(suffix='.h5')
        
        # 映射压缩算法
        compression_map = {
            CompressionAlgorithm.GZIP: 'gzip',
            CompressionAlgorithm.ZSTD: 'lzf',  # HDF5不支持zstd，使用lzf
            CompressionAlgorithm.NONE: None
        }
        
        compression = compression_map.get(algorithm)
        
        df.to_hdf(
            file_path,
            key='data',
            mode='w',
            complevel=9 if compression else None,
            complib=compression
        )
        
        return Path(file_path).stat().st_size
    
    def _compress_to_pickle(
        self,
        df: pd.DataFrame,
        algorithm: CompressionAlgorithm,
        file_path: Optional[str]
    ) -> int:
        """
        压缩为Pickle格式
        
        Args:
            df: DataFrame
            algorithm: 压缩算法
            file_path: 文件路径
            
        Returns:
            压缩后大小
        """
        import pickle
        
        if file_path is None:
            import tempfile
            file_path = tempfile.mktemp(suffix='.pkl')
        
        # 序列化数据
        data = pickle.dumps(df)
        
        # 压缩数据
        if algorithm == CompressionAlgorithm.GZIP:
            import gzip
            compressed = gzip.compress(data, compresslevel=9)
        elif algorithm == CompressionAlgorithm.LZ4 and self._available_algorithms[CompressionAlgorithm.LZ4]:
            import lz4.frame
            compressed = lz4.frame.compress(data)
        elif algorithm == CompressionAlgorithm.ZSTD and self._available_algorithms[CompressionAlgorithm.ZSTD]:
            import zstandard
            compressed = zstandard.compress(data)
        else:
            compressed = data
        
        # 写入文件
        with open(file_path, 'wb') as f:
            f.write(compressed)
        
        return len(compressed)
    
    def _benchmark_decompression(
        self,
        file_path: Optional[str],
        format: StorageFormat
    ) -> float:
        """
        基准测试解压时间
        
        Args:
            file_path: 文件路径
            format: 存储格式
            
        Returns:
            解压时间（毫秒）
        """
        if file_path is None or not Path(file_path).exists():
            return 0.0
        
        start_time = time.time()
        
        try:
            if format == StorageFormat.PARQUET:
                pd.read_parquet(file_path)
            elif format == StorageFormat.FEATHER:
                pd.read_feather(file_path)
            elif format == StorageFormat.HDF5:
                pd.read_hdf(file_path, key='data')
            elif format == StorageFormat.PICKLE:
                import pickle
                with open(file_path, 'rb') as f:
                    pickle.load(f)
        except Exception as e:
            logger.warning(f"解压基准测试失败: {e}")
            return 0.0
        
        return (time.time() - start_time) * 1000
    
    def select_optimal_algorithm(
        self,
        df: pd.DataFrame,
        priority: str = "balanced"
    ) -> Tuple[CompressionAlgorithm, StorageFormat]:
        """
        选择最优压缩算法
        
        Args:
            df: DataFrame样本
            priority: 优先级（"speed"/"ratio"/"balanced"）
            
        Returns:
            (最优算法, 最优格式)
        """
        import tempfile
        
        algorithms_to_test = [
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.SNAPPY,
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.GZIP
        ]
        
        formats_to_test = [
            StorageFormat.PARQUET,
            StorageFormat.FEATHER
        ]
        
        best_score = float('-inf')
        best_algorithm = CompressionAlgorithm.ZSTD
        best_format = StorageFormat.PARQUET
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for format in formats_to_test:
                for algorithm in algorithms_to_test:
                    if not self._available_algorithms.get(algorithm, False):
                        continue
                    
                    file_path = f"{tmpdir}/test.{format.value}"
                    
                    try:
                        result = self.compress_dataframe(
                            df, algorithm, format, file_path
                        )
                        
                        # 计算得分
                        if priority == "speed":
                            # 优先考虑速度
                            score = 1.0 / (result.compression_time_ms + result.decompression_time_ms + 1)
                        elif priority == "ratio":
                            # 优先考虑压缩比
                            score = result.compression_ratio
                        else:  # balanced
                            # 平衡考虑
                            speed_score = 1.0 / (result.compression_time_ms + result.decompression_time_ms + 1)
                            ratio_score = result.compression_ratio
                            score = speed_score * 0.3 + ratio_score * 0.7
                        
                        if score > best_score:
                            best_score = score
                            best_algorithm = algorithm
                            best_format = format
                            
                    except Exception as e:
                        logger.warning(f"测试 {algorithm.value}/{format.value} 失败: {e}")
                        continue
        
        logger.info(f"最优压缩方案: {best_algorithm.value}/{best_format.value}")
        
        return best_algorithm, best_format
    
    def _update_stats(self, result: CompressionResult):
        """
        更新统计信息
        
        Args:
            result: 压缩结果
        """
        self._stats.total_operations += 1
        self._stats.total_bytes_saved += (result.original_size - result.compressed_size)
        
        # 更新平均压缩比
        if self._stats.total_operations == 1:
            self._stats.avg_compression_ratio = result.compression_ratio
            self._stats.avg_compression_time_ms = result.compression_time_ms
        else:
            n = self._stats.total_operations
            self._stats.avg_compression_ratio = (
                (self._stats.avg_compression_ratio * (n - 1) + result.compression_ratio) / n
            )
            self._stats.avg_compression_time_ms = (
                (self._stats.avg_compression_time_ms * (n - 1) + result.compression_time_ms) / n
            )
        
        # 记录算法性能
        if result.algorithm not in self._algorithm_performance:
            self._algorithm_performance[result.algorithm] = []
        self._algorithm_performance[result.algorithm].append(result.compression_ratio)
    
    def get_stats(self) -> CompressionStats:
        """获取压缩统计"""
        return self._stats
    
    def get_algorithm_comparison(self) -> Dict[str, Any]:
        """
        获取算法性能对比
        
        Returns:
            算法性能对比数据
        """
        comparison = {}
        
        for algorithm, ratios in self._algorithm_performance.items():
            if ratios:
                comparison[algorithm.value] = {
                    'avg_ratio': np.mean(ratios),
                    'min_ratio': np.min(ratios),
                    'max_ratio': np.max(ratios),
                    'std_ratio': np.std(ratios),
                    'usage_count': len(ratios)
                }
        
        return comparison


# 全局压缩引擎实例
_compression_engine: Optional[AdvancedCompressionEngine] = None


def get_compression_engine() -> AdvancedCompressionEngine:
    """
    获取压缩引擎实例（单例模式）
    
    Returns:
        AdvancedCompressionEngine实例
    """
    global _compression_engine
    
    if _compression_engine is None:
        _compression_engine = AdvancedCompressionEngine()
    
    return _compression_engine
