"""
数据压缩服务 - 高性能数据压缩与解压

本模块提供多种数据压缩算法，支持：
1. LZ4高速压缩 - 适用于实时数据
2. Snappy压缩 - 平衡速度和质量
3. Zstandard压缩 - 高压缩率
4. Parquet列式存储 - 分析型数据优化
5. 自适应压缩策略 - 根据数据特征选择算法

作者: 数据团队
创建日期: 2026-02-21
版本: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import io

import pandas as pd
import numpy as np

# 尝试导入压缩库
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logging.warning("LZ4库未安装")

try:
    import snappy
    SNAPPY_AVAILABLE = True
except ImportError:
    SNAPPY_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# Parquet支持
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.exceptions import CompressionError


# 配置日志
logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """压缩算法类型"""
    LZ4 = "lz4"           # 极速压缩
    SNAPPY = "snappy"     # 快速压缩
    ZSTD = "zstd"         # 标准压缩
    GZIP = "gzip"         # 通用压缩
    NONE = "none"         # 不压缩


class DataType(Enum):
    """数据类型"""
    TIMESERIES = "timeseries"    # 时间序列数据
    TICK = "tick"                # Tick数据
    ORDERBOOK = "orderbook"      # 订单簿数据
    NEWS = "news"                # 新闻数据
    METRICS = "metrics"          # 指标数据


@dataclass
class CompressionResult:
    """压缩结果"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    algorithm: CompressionAlgorithm
    
    @property
    def space_saving(self) -> float:
        """空间节省比例"""
        if self.original_size == 0:
            return 0.0
        return (self.original_size - self.compressed_size) / self.original_size * 100


@dataclass
class CompressionStats:
    """压缩统计"""
    total_compressions: int = 0
    total_decompressions: int = 0
    total_bytes_original: int = 0
    total_bytes_compressed: int = 0
    avg_compression_time_ms: float = 0.0
    avg_compression_ratio: float = 0.0


class DataCompressionService:
    """
    数据压缩服务
    
    功能:
    1. 多算法压缩支持 (LZ4, Snappy, Zstd, Gzip)
    2. 自适应压缩策略选择
    3. DataFrame专用压缩 (Parquet)
    4. 压缩性能监控
    5. 批量压缩处理
    
    使用示例:
        service = DataCompressionService()
        
        # 压缩数据
        result = service.compress(
            data=original_data,
            algorithm=CompressionAlgorithm.LZ4
        )
        
        # 解压数据
        decompressed = service.decompress(
            compressed_data=result.compressed_data,
            algorithm=CompressionAlgorithm.LZ4
        )
    """
    
    def __init__(self):
        """初始化压缩服务"""
        self._stats = CompressionStats()
        self._algorithm_stats: Dict[CompressionAlgorithm, CompressionStats] = {
            algo: CompressionStats() for algo in CompressionAlgorithm
        }
        
        # 压缩算法选择阈值
        self._thresholds = {
            "min_size_for_compression": 1024,  # 1KB以下不压缩
            "lz4_max_ratio": 2.0,              # LZ4最大压缩比
            "zstd_min_ratio": 3.0              # Zstd最小压缩比
        }
        
        # 初始化Zstd压缩器
        self._zstd_compressor = None
        self._zstd_decompressor = None
        if ZSTD_AVAILABLE:
            self._zstd_compressor = zstd.ZstdCompressor()
            self._zstd_decompressor = zstd.ZstdDecompressor()
        
        logger.info("数据压缩服务已初始化")
    
    def compress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4,
        level: Optional[int] = None
    ) -> Tuple[bytes, CompressionResult]:
        """
        压缩数据
        
        参数:
            data: 原始数据
            algorithm: 压缩算法
            level: 压缩级别（可选）
            
        返回:
            Tuple[bytes, CompressionResult]: 压缩后的数据和结果信息
        """
        start_time = time.time()
        original_size = len(data)
        
        # 小数据不压缩
        if original_size < self._thresholds["min_size_for_compression"]:
            return data, CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time_ms=0.0,
                algorithm=CompressionAlgorithm.NONE
            )
        
        try:
            if algorithm == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
                compressed = lz4.frame.compress(data)
            elif algorithm == CompressionAlgorithm.SNAPPY and SNAPPY_AVAILABLE:
                compressed = snappy.compress(data)
            elif algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
                compressed = self._zstd_compressor.compress(data)
            elif algorithm == CompressionAlgorithm.GZIP:
                import gzip
                compressed = gzip.compress(data, compresslevel=level or 6)
            else:
                # 回退到不压缩
                compressed = data
                algorithm = CompressionAlgorithm.NONE
            
            compression_time = (time.time() - start_time) * 1000
            compressed_size = len(compressed)
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            result = CompressionResult(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=ratio,
                compression_time_ms=compression_time,
                algorithm=algorithm
            )
            
            # 更新统计
            self._update_stats(algorithm, result)
            
            return compressed, result
            
        except Exception as e:
            logger.error(f"压缩失败: {e}")
            raise CompressionError(f"压缩失败: {e}")
    
    def decompress(
        self,
        compressed_data: bytes,
        algorithm: CompressionAlgorithm
    ) -> bytes:
        """
        解压数据
        
        参数:
            compressed_data: 压缩数据
            algorithm: 压缩算法
            
        返回:
            bytes: 解压后的数据
        """
        if algorithm == CompressionAlgorithm.NONE:
            return compressed_data
        
        try:
            if algorithm == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
                return lz4.frame.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.SNAPPY and SNAPPY_AVAILABLE:
                return snappy.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
                return self._zstd_decompressor.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.GZIP:
                import gzip
                return gzip.decompress(compressed_data)
            else:
                return compressed_data
                
        except Exception as e:
            logger.error(f"解压失败: {e}")
            raise CompressionError(f"解压失败: {e}")
    
    def compress_dataframe(
        self,
        df: pd.DataFrame,
        compression: str = "zstd",
        use_dictionary: bool = False
    ) -> bytes:
        """
        压缩DataFrame为Parquet格式
        
        参数:
            df: DataFrame数据
            compression: 压缩算法 (zstd, lz4, snappy, gzip, none)
            use_dictionary: 是否使用字典编码
            
        返回:
            bytes: Parquet格式的压缩数据
        """
        try:
            # 转换为PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # 写入内存缓冲区
            buf = io.BytesIO()
            pq.write_table(
                table,
                buf,
                compression=compression,
                use_dictionary=use_dictionary,
                write_statistics=True
            )
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"DataFrame压缩失败: {e}")
            raise CompressionError(f"DataFrame压缩失败: {e}")
    
    def decompress_dataframe(self, compressed_data: bytes) -> pd.DataFrame:
        """
        从Parquet格式解压DataFrame
        
        参数:
            compressed_data: Parquet格式的压缩数据
            
        返回:
            pd.DataFrame: 解压后的DataFrame
        """
        try:
            buf = io.BytesIO(compressed_data)
            table = pq.read_table(buf)
            return table.to_pandas()
            
        except Exception as e:
            logger.error(f"DataFrame解压失败: {e}")
            raise CompressionError(f"DataFrame解压失败: {e}")
    
    def select_optimal_algorithm(
        self,
        data: bytes,
        data_type: DataType = DataType.TIMESERIES,
        speed_priority: bool = True
    ) -> CompressionAlgorithm:
        """
        选择最优压缩算法
        
        参数:
            data: 待压缩数据
            data_type: 数据类型
            speed_priority: 是否优先速度
            
        返回:
            CompressionAlgorithm: 最优算法
        """
        data_size = len(data)
        
        # 小数据使用快速算法
        if data_size < 10 * 1024:  # 10KB以下
            if LZ4_AVAILABLE:
                return CompressionAlgorithm.LZ4
            elif SNAPPY_AVAILABLE:
                return CompressionAlgorithm.SNAPPY
            else:
                return CompressionAlgorithm.GZIP
        
        # 根据数据类型选择
        if speed_priority:
            if data_type in [DataType.TICK, DataType.ORDERBOOK]:
                # 实时数据优先速度
                if LZ4_AVAILABLE:
                    return CompressionAlgorithm.LZ4
                elif SNAPPY_AVAILABLE:
                    return CompressionAlgorithm.SNAPPY
            else:
                # 历史数据可以使用更高压缩率
                if ZSTD_AVAILABLE:
                    return CompressionAlgorithm.ZSTD
                return CompressionAlgorithm.GZIP
        else:
            # 优先压缩率
            if ZSTD_AVAILABLE:
                return CompressionAlgorithm.ZSTD
            return CompressionAlgorithm.GZIP
        
        return CompressionAlgorithm.GZIP
    
    def benchmark_compression(
        self,
        data: bytes,
        algorithms: Optional[List[CompressionAlgorithm]] = None
    ) -> Dict[CompressionAlgorithm, CompressionResult]:
        """
        压缩算法性能基准测试
        
        参数:
            data: 测试数据
            algorithms: 要测试的算法列表
            
        返回:
            Dict[CompressionAlgorithm, CompressionResult]: 各算法的测试结果
        """
        if algorithms is None:
            algorithms = [
                CompressionAlgorithm.LZ4,
                CompressionAlgorithm.SNAPPY,
                CompressionAlgorithm.ZSTD,
                CompressionAlgorithm.GZIP
            ]
        
        results = {}
        
        for algo in algorithms:
            try:
                compressed, result = self.compress(data, algo)
                
                # 验证解压
                decompressed = self.decompress(compressed, algo)
                if decompressed != data:
                    logger.warning(f"{algo.value} 压缩/解压数据不一致")
                    continue
                
                results[algo] = result
                
            except Exception as e:
                logger.warning(f"{algo.value} 测试失败: {e}")
        
        return results
    
    def compress_batch(
        self,
        data_items: List[bytes],
        algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4,
        parallel: bool = False
    ) -> List[Tuple[bytes, CompressionResult]]:
        """
        批量压缩
        
        参数:
            data_items: 数据列表
            algorithm: 压缩算法
            parallel: 是否并行处理
            
        返回:
            List[Tuple[bytes, CompressionResult]]: 压缩结果列表
        """
        results = []
        
        if parallel:
            # 使用多线程并行压缩
            from concurrent.futures import ThreadPoolExecutor
            
            def compress_item(item):
                return self.compress(item, algorithm)
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(compress_item, data_items))
        else:
            # 串行压缩
            for item in data_items:
                result = self.compress(item, algorithm)
                results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取压缩统计信息
        
        返回:
            Dict: 统计信息
        """
        total_saving = 0
        if self._stats.total_bytes_original > 0:
            total_saving = (
                (self._stats.total_bytes_original - self._stats.total_bytes_compressed)
                / self._stats.total_bytes_original * 100
            )
        
        return {
            "total_compressions": self._stats.total_compressions,
            "total_decompressions": self._stats.total_decompressions,
            "total_bytes_original": self._stats.total_bytes_original,
            "total_bytes_compressed": self._stats.total_bytes_compressed,
            "total_space_saving": f"{total_saving:.1f}%",
            "avg_compression_time_ms": f"{self._stats.avg_compression_time_ms:.2f}",
            "avg_compression_ratio": f"{self._stats.avg_compression_ratio:.2f}x",
            "algorithm_stats": {
                algo.value: {
                    "compressions": stats.total_compressions,
                    "avg_ratio": f"{stats.avg_compression_ratio:.2f}x" if stats.total_compressions > 0 else "N/A"
                }
                for algo, stats in self._algorithm_stats.items()
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self._stats = CompressionStats()
        self._algorithm_stats = {
            algo: CompressionStats() for algo in CompressionAlgorithm
        }
        logger.info("压缩统计已重置")
    
    def _update_stats(self, algorithm: CompressionAlgorithm, result: CompressionResult):
        """更新统计信息"""
        # 更新总体统计
        self._stats.total_compressions += 1
        self._stats.total_bytes_original += result.original_size
        self._stats.total_bytes_compressed += result.compressed_size
        
        # 更新平均时间
        n = self._stats.total_compressions
        self._stats.avg_compression_time_ms = (
            (self._stats.avg_compression_time_ms * (n - 1) + result.compression_time_ms) / n
        )
        
        # 更新平均压缩比
        self._stats.avg_compression_ratio = (
            (self._stats.avg_compression_ratio * (n - 1) + result.compression_ratio) / n
        )
        
        # 更新算法统计
        algo_stats = self._algorithm_stats[algorithm]
        algo_stats.total_compressions += 1
        algo_stats.total_bytes_original += result.original_size
        algo_stats.total_bytes_compressed += result.compressed_size
        
        n_algo = algo_stats.total_compressions
        algo_stats.avg_compression_ratio = (
            (algo_stats.avg_compression_ratio * (n_algo - 1) + result.compression_ratio) / n_algo
        )


# 便捷函数
def compress_data(
    data: bytes,
    algorithm: str = "lz4"
) -> Tuple[bytes, Dict[str, Any]]:
    """
    便捷函数 - 压缩数据
    
    参数:
        data: 原始数据
        algorithm: 算法名称 (lz4, snappy, zstd, gzip)
        
    返回:
        Tuple[bytes, Dict]: 压缩数据和结果信息
    """
    service = DataCompressionService()
    
    algo_map = {
        "lz4": CompressionAlgorithm.LZ4,
        "snappy": CompressionAlgorithm.SNAPPY,
        "zstd": CompressionAlgorithm.ZSTD,
        "gzip": CompressionAlgorithm.GZIP
    }
    
    algo = algo_map.get(algorithm, CompressionAlgorithm.LZ4)
    compressed, result = service.compress(data, algo)
    
    return compressed, {
        "original_size": result.original_size,
        "compressed_size": result.compressed_size,
        "ratio": result.compression_ratio,
        "time_ms": result.compression_time_ms
    }


def compress_dataframe_fast(df: pd.DataFrame) -> bytes:
    """
    便捷函数 - 快速压缩DataFrame
    
    参数:
        df: DataFrame数据
        
    返回:
        bytes: 压缩后的数据
    """
    service = DataCompressionService()
    return service.compress_dataframe(df, compression="lz4")
