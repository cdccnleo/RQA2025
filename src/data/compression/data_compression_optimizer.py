#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据压缩优化器

实现智能数据压缩和传输优化
- 自适应压缩算法选择
- 压缩效果监控
- 传输性能优化
- 压缩策略管理
"""

import gzip
import bz2
import lzma
import zlib
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging


@dataclass
class CompressionMetrics:

    """压缩指标"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    algorithm: str
    data_type: str
    timestamp: datetime

    @property
    def compression_efficiency(self) -> float:
        """压缩效率 (0 - 1, 1为最佳)"""
        if self.compression_ratio <= 0:
            return 0.0
        return min(1.0, self.compression_ratio / 10.0)  # 10:1压缩比为满分

    @property
    def performance_score(self) -> float:
        """性能评分 (考虑压缩时间和压缩比)"""
        time_penalty = min(1.0, self.compression_time / 10.0)  # 10秒为最大惩罚
        return self.compression_efficiency * (1 - time_penalty)


@dataclass
class CompressionStrategy:

    """压缩策略"""
    name: str
    algorithm: str
    compression_level: int = 6
    min_size_threshold: int = 1024  # 最小压缩大小阈值
    max_size_threshold: int = 104857600  # 最大压缩大小阈值 (100MB)
    enabled: bool = True
    priority: int = 5

    def is_applicable(self, data_size: int, data_type: str) -> bool:
        """检查是否适用于给定的数据"""
        if not self.enabled:
            return False

        if data_size < self.min_size_threshold:
            return False

        if data_size > self.max_size_threshold:
            return False

        return True


class DataCompressionOptimizer:

    """
    数据压缩优化器

    提供智能的数据压缩和传输优化：
    - 自适应压缩算法选择
    - 压缩效果实时监控
    - 传输性能优化
    - 压缩策略动态调整
    """

    def __init__(self):
        """初始化数据压缩优化器"""
        self.algorithms = {
            'gzip': self._compress_gzip,
            'bz2': self._compress_bz2,
            'lzma': self._compress_lzma,
            'zlib': self._compress_zlib,
            'none': self._compress_none
        }

        self.decompress_algorithms = {
            'gzip': self._decompress_gzip,
            'bz2': self._decompress_bz2,
            'lzma': self._decompress_lzma,
            'zlib': self._decompress_zlib,
            'none': self._decompress_none
        }

        # 压缩策略
        self.strategies = self._initialize_strategies()

        # 性能历史
        self.metrics_history: List[CompressionMetrics] = []
        self.algorithm_performance: Dict[str, List[float]] = defaultdict(list)

        # 自适应参数
        self._adaptive_params = {
            'learning_rate': 0.1,
            'performance_window': 100,  # 性能评估窗口
            'strategy_update_interval': 300,  # 策略更新间隔(秒)
            'last_strategy_update': datetime.now()
        }

        logging.info("数据压缩优化器初始化完成")

    def _initialize_strategies(self) -> List[CompressionStrategy]:
        """初始化压缩策略"""
        return [
            CompressionStrategy(
                name="text_gzip_fast",
                algorithm="gzip",
                compression_level=1,
                min_size_threshold=1024,
                max_size_threshold=10485760,  # 10MB
                priority=8
            ),
            CompressionStrategy(
                name="text_gzip_balanced",
                algorithm="gzip",
                compression_level=6,
                min_size_threshold=1024,
                max_size_threshold=104857600,  # 100MB
                priority=6
            ),
            CompressionStrategy(
                name="binary_lzma",
                algorithm="lzma",
                compression_level=6,
                min_size_threshold=10240,  # 10KB
                max_size_threshold=1073741824,  # 1GB
                priority=4
            ),
            CompressionStrategy(
                name="large_data_bz2",
                algorithm="bz2",
                compression_level=9,
                min_size_threshold=1048576,  # 1MB
                max_size_threshold=1073741824,  # 1GB
                priority=3
            ),
            CompressionStrategy(
                name="no_compression",
                algorithm="none",
                min_size_threshold=0,
                max_size_threshold=1024,  # 1KB以下不压缩
                priority=10
            )
        ]

    # =========================================================================
    # 核心压缩方法
    # =========================================================================

    def compress_data(self, data: Union[str, bytes], data_type: str = "general",


                      strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        压缩数据

        Args:
            data: 要压缩的数据
            data_type: 数据类型
            strategy_name: 指定的策略名称

        Returns:
            压缩结果字典
        """
        # 转换为字节串
        if isinstance(data, str):
            data_bytes = data.encode('utf - 8')
        else:
            data_bytes = data

        original_size = len(data_bytes)

        # 选择压缩策略
        strategy = self._select_compression_strategy(data_bytes, data_type, strategy_name)

        if not strategy:
            # 不压缩
            return {
                'compressed_data': data_bytes,
                'original_size': original_size,
                'compressed_size': original_size,
                'compression_ratio': 1.0,
                'algorithm': 'none',
                'strategy': 'no_compression',
                'compression_time': 0.0,
                'decompression_time': 0.0
            }

        # 执行压缩
        start_time = time.time()
        compressed_data = self.algorithms[strategy.algorithm](
            data_bytes, strategy.compression_level)
        compression_time = time.time() - start_time

        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

        # 测试解压时间
        start_time = time.time()
        self.decompress_algorithms[strategy.algorithm](compressed_data)
        decompression_time = time.time() - start_time

        # 记录指标
        metrics = CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            decompression_time=decompression_time,
            algorithm=strategy.algorithm,
            data_type=data_type,
            timestamp=datetime.now()
        )

        self._record_metrics(metrics)

        result = {
            'compressed_data': compressed_data,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'algorithm': strategy.algorithm,
            'strategy': strategy.name,
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'metrics': metrics
        }

        logging.info(".1f"
                     f"压缩时间: {compression_time:.3f}s")

        return result

    def decompress_data(self, compressed_data: bytes, algorithm: str) -> bytes:
        """
        解压数据

        Args:
            compressed_data: 压缩数据
            algorithm: 压缩算法

        Returns:
            解压后的数据
        """
        if algorithm not in self.decompress_algorithms:
            raise ValueError(f"不支持的解压算法: {algorithm}")

        start_time = time.time()
        decompressed_data = self.decompress_algorithms[algorithm](compressed_data)
        decompression_time = time.time() - start_time

        logging.info(f"数据解压完成，大小: {len(decompressed_data)} bytes, "
                     f"解压时间: {decompression_time:.3f}s")

        return decompressed_data

    # =========================================================================
    # 压缩算法实现
    # =========================================================================

    def _compress_gzip(self, data: bytes, level: int) -> bytes:
        """GZIP压缩"""
        return gzip.compress(data, compresslevel=level)

    def _compress_bz2(self, data: bytes, level: int) -> bytes:
        """BZ2压缩"""
        return bz2.compress(data, compresslevel=level)

    def _compress_lzma(self, data: bytes, level: int) -> bytes:
        """LZMA压缩"""
        return lzma.compress(data, preset=level)

    def _compress_zlib(self, data: bytes, level: int) -> bytes:
        """ZLIB压缩"""
        return zlib.compress(data, level=level)

    def _compress_none(self, data: bytes, level: int) -> bytes:
        """不压缩"""
        return data

    def _decompress_gzip(self, data: bytes) -> bytes:
        """GZIP解压"""
        return gzip.decompress(data)

    def _decompress_bz2(self, data: bytes) -> bytes:
        """BZ2解压"""
        return bz2.decompress(data)

    def _decompress_lzma(self, data: bytes) -> bytes:
        """LZMA解压"""
        return lzma.decompress(data)

    def _decompress_zlib(self, data: bytes) -> bytes:
        """ZLIB解压"""
        return zlib.decompress(data)

    def _decompress_none(self, data: bytes) -> bytes:
        """不解压"""
        return data

    # =========================================================================
    # 策略选择和优化
    # =========================================================================

    def _select_compression_strategy(self, data: bytes, data_type: str,


                                     strategy_name: Optional[str] = None) -> Optional[CompressionStrategy]:
        """选择压缩策略"""
        data_size = len(data)

        # 如果指定了策略名称，直接使用
        if strategy_name:
            for strategy in self.strategies:
                if strategy.name == strategy_name and strategy.is_applicable(data_size, data_type):
                    return strategy

        # 基于数据特征智能选择策略
        applicable_strategies = [
            strategy for strategy in self.strategies
            if strategy.is_applicable(data_size, data_type)
        ]

        if not applicable_strategies:
            return None

        # 基于历史性能选择最佳策略
        best_strategy = self._select_best_strategy(applicable_strategies, data_type)

        return best_strategy

    def _select_best_strategy(self, strategies: List[CompressionStrategy],


                              data_type: str) -> CompressionStrategy:
        """选择最佳策略"""
        # 基于历史性能评分选择
        strategy_scores = {}

        for strategy in strategies:
            performance_scores = self.algorithm_performance.get(strategy.algorithm, [])
            if performance_scores:
                avg_score = sum(performance_scores) / len(performance_scores)
            else:
                avg_score = 0.5  # 默认评分

            # 考虑优先级
            final_score = avg_score * (1 + strategy.priority / 10.0)
            strategy_scores[strategy.name] = (final_score, strategy)

        # 选择得分最高的策略
        best_strategy_name = max(strategy_scores.keys(), key=lambda k: strategy_scores[k][0])
        best_strategy = strategy_scores[best_strategy_name][1]

        return best_strategy

    def _record_metrics(self, metrics: CompressionMetrics):
        """记录指标"""
        self.metrics_history.append(metrics)

        # 限制历史记录数量
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]

        # 更新算法性能历史
        self.algorithm_performance[metrics.algorithm].append(metrics.performance_score)

        # 限制性能历史长度
        if len(self.algorithm_performance[metrics.algorithm]) > self._adaptive_params['performance_window']:
            self.algorithm_performance[metrics.algorithm] = \
                self.algorithm_performance[metrics.algorithm][-self._adaptive_params['performance_window']:]

        # 检查是否需要更新策略
        self._check_strategy_update()

    def _check_strategy_update(self):
        """检查是否需要更新策略"""
        now = datetime.now()
        time_since_update = (now - self._adaptive_params['last_strategy_update']).total_seconds()

        if time_since_update >= self._adaptive_params['strategy_update_interval']:
            self._update_strategies()
            self._adaptive_params['last_strategy_update'] = now

    def _update_strategies(self):
        """更新策略"""
        # 基于性能数据调整策略参数
        for algorithm, scores in self.algorithm_performance.items():
            if len(scores) >= 10:  # 至少10个数据点
                avg_score = sum(scores) / len(scores)
                recent_scores = scores[-10:]
                recent_avg = sum(recent_scores) / len(recent_scores)

                # 如果最近性能下降，调整策略
                if recent_avg < avg_score * 0.9:
                    self._adjust_strategy_for_algorithm(algorithm, 'performance_decline')
                elif recent_avg > avg_score * 1.1:
                    self._adjust_strategy_for_algorithm(algorithm, 'performance_improvement')

    def _adjust_strategy_for_algorithm(self, algorithm: str, reason: str):
        """调整算法的策略"""
        for strategy in self.strategies:
            if strategy.algorithm == algorithm:
                if reason == 'performance_decline':
                    # 降低压缩级别以提高速度
                    if strategy.compression_level > 1:
                        strategy.compression_level = max(1, strategy.compression_level - 1)
                        logging.info(f"降低策略 {strategy.name} 的压缩级别至 {strategy.compression_level}")
                elif reason == 'performance_improvement':
                    # 提高压缩级别以提高压缩比
                    if strategy.compression_level < 9:
                        strategy.compression_level = min(9, strategy.compression_level + 1)
                        logging.info(f"提高策略 {strategy.name} 的压缩级别至 {strategy.compression_level}")

    # =========================================================================
    # 性能监控和报告
    # =========================================================================

    def get_compression_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        获取压缩报告

        Args:
            time_range_hours: 时间范围（小时）

        Returns:
            压缩报告
        """
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

        # 过滤时间范围内的指标
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        if not recent_metrics:
            return {
                'summary': '无数据',
                'time_range_hours': time_range_hours,
                'timestamp': datetime.now().isoformat()
            }

        # 计算统计信息
        total_original = sum(m.original_size for m in recent_metrics)
        total_compressed = sum(m.compressed_size for m in recent_metrics)
        avg_compression_ratio = sum(
            m.compression_ratio for m in recent_metrics) / len(recent_metrics)
        avg_compression_time = sum(m.compression_time for m in recent_metrics) / len(recent_metrics)
        avg_decompression_time = sum(
            m.decompression_time for m in recent_metrics) / len(recent_metrics)

        # 按算法分组统计
        algorithm_stats = defaultdict(list)
        for metrics in recent_metrics:
            algorithm_stats[metrics.algorithm].append(metrics)

        algorithm_summary = {}
        for algorithm, metrics_list in algorithm_stats.items():
            ratios = [m.compression_ratio for m in metrics_list]
            algorithm_summary[algorithm] = {
                'count': len(metrics_list),
                'avg_ratio': sum(ratios) / len(ratios),
                'best_ratio': max(ratios),
                'worst_ratio': min(ratios)
            }

        return {
            'summary': {
                'total_operations': len(recent_metrics),
                'total_original_size': total_original,
                'total_compressed_size': total_compressed,
                'overall_compression_ratio': total_original / total_compressed if total_compressed > 0 else 0,
                'avg_compression_ratio': avg_compression_ratio,
                'avg_compression_time': avg_compression_time,
                'avg_decompression_time': avg_decompression_time
            },
            'algorithm_stats': algorithm_summary,
            'strategies': [
                {
                    'name': s.name,
                    'algorithm': s.algorithm,
                    'compression_level': s.compression_level,
                    'enabled': s.enabled,
                    'priority': s.priority
                }
                for s in self.strategies
            ],
            'time_range_hours': time_range_hours,
            'timestamp': datetime.now().isoformat()
        }

    def get_algorithm_performance(self, algorithm: str) -> Dict[str, Any]:
        """
        获取算法性能

        Args:
            algorithm: 算法名称

        Returns:
            算法性能信息
        """
        scores = self.algorithm_performance.get(algorithm, [])

        if not scores:
            return {
                'algorithm': algorithm,
                'status': 'no_data',
                'timestamp': datetime.now().isoformat()
            }

        return {
            'algorithm': algorithm,
            'performance_score': sum(scores) / len(scores),
            'measurements': len(scores),
            'best_score': max(scores),
            'worst_score': min(scores),
            'recent_trend': self._calculate_trend(scores[-10:] if len(scores) >= 10 else scores),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_trend(self, scores: List[float]) -> str:
        """计算趋势"""
        if len(scores) < 2:
            return 'insufficient_data'

        # 简单的线性趋势
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n

        numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 'stable'

        slope = numerator / denominator

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'

    # =========================================================================
    # 策略管理
    # =========================================================================

    def add_compression_strategy(self, strategy: CompressionStrategy):
        """
        添加压缩策略

        Args:
            strategy: 压缩策略
        """
        # 检查是否已存在同名策略
        for existing in self.strategies:
            if existing.name == strategy.name:
                raise ValueError(f"策略名称已存在: {strategy.name}")

        self.strategies.append(strategy)
        logging.info(f"压缩策略已添加: {strategy.name}")

    def remove_compression_strategy(self, strategy_name: str):
        """
        移除压缩策略

        Args:
            strategy_name: 策略名称
        """
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        logging.info(f"压缩策略已移除: {strategy_name}")

    def update_strategy_priority(self, strategy_name: str, priority: int):
        """
        更新策略优先级

        Args:
            strategy_name: 策略名称
            priority: 新优先级
        """
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.priority = priority
                logging.info(f"策略 {strategy_name} 优先级已更新为 {priority}")
                return

        raise ValueError(f"策略不存在: {strategy_name}")

    def enable_strategy(self, strategy_name: str):
        """
        启用策略

        Args:
            strategy_name: 策略名称
        """
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.enabled = True
                logging.info(f"策略 {strategy_name} 已启用")
                return

        raise ValueError(f"策略不存在: {strategy_name}")

    def disable_strategy(self, strategy_name: str):
        """
        禁用策略

        Args:
            strategy_name: 策略名称
        """
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.enabled = False
                logging.info(f"策略 {strategy_name} 已禁用")
                return

        raise ValueError(f"策略不存在: {strategy_name}")

    # =========================================================================
    # 批量压缩优化
    # =========================================================================

    def compress_batch(self, data_list: List[Dict[str, Any]],


                       parallel: bool = True) -> List[Dict[str, Any]]:
        """
        批量压缩数据

        Args:
            data_list: 数据列表 [{'data': bytes, 'type': str}]
            parallel: 是否并行处理

        Returns:
            压缩结果列表
        """
        results = []

        if parallel:
            # 并行处理（简化实现）
            for item in data_list:
                result = self.compress_data(
                    item['data'],
                    item.get('type', 'general')
                )
                results.append(result)
        else:
            # 串行处理
            for item in data_list:
                result = self.compress_data(
                    item['data'],
                    item.get('type', 'general')
                )
                results.append(result)

        logging.info(f"批量压缩完成，共处理 {len(data_list)} 个数据项")
        return results

    # =========================================================================
    # 配置和清理
    # =========================================================================

    def clear_metrics_history(self):
        """清除指标历史"""
        self.metrics_history.clear()
        self.algorithm_performance.clear()
        logging.info("指标历史已清除")

    def get_optimizer_status(self) -> Dict[str, Any]:
        """
        获取优化器状态

        Returns:
            优化器状态
        """
        return {
            'strategies_count': len(self.strategies),
            'enabled_strategies': len([s for s in self.strategies if s.enabled]),
            'metrics_history_count': len(self.metrics_history),
            'algorithms_tracked': len(self.algorithm_performance),
            'adaptive_params': self._adaptive_params.copy(),
            'timestamp': datetime.now().isoformat()
        }
