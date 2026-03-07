#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据层分块处理优化脚本

功能：
1. 动态分块大小调整
2. 内存使用模式优化
3. 处理速度提升
4. 自适应性能调优
"""

import time
import gc
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import json
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.utils.logger import get_logger
except ImportError:
    def get_logger(name):
        class MockLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARNING] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
            def debug(self, msg): print(f"[DEBUG] {msg}")
        return MockLogger()


@dataclass
class ChunkPerformanceMetrics:
    """分块性能指标"""
    chunk_size: int
    processing_time: float
    memory_usage_mb: float
    throughput_records_per_sec: float
    memory_efficiency: float
    cpu_utilization: float


@dataclass
class AdaptiveChunkConfig:
    """自适应分块配置"""
    min_chunk_size: int = 1000
    max_chunk_size: int = 100000
    target_processing_time: float = 5.0
    target_memory_usage_mb: float = 500.0
    performance_window: int = 10
    adjustment_factor: float = 1.2


class DynamicChunkProcessor:
    """动态分块处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("dynamic_chunk_processor")

        # 自适应配置
        self.adaptive_config = AdaptiveChunkConfig(
            min_chunk_size=self.config.get('min_chunk_size', 1000),
            max_chunk_size=self.config.get('max_chunk_size', 100000),
            target_processing_time=self.config.get('target_processing_time', 5.0),
            target_memory_usage_mb=self.config.get('target_memory_usage_mb', 500.0),
            performance_window=self.config.get('performance_window', 10),
            adjustment_factor=self.config.get('adjustment_factor', 1.2)
        )

        # 性能历史
        self.performance_history = deque(maxlen=self.adaptive_config.performance_window)
        self.current_chunk_size = self.adaptive_config.min_chunk_size

        # 处理统计
        self.total_records_processed = 0
        self.total_processing_time = 0.0
        self.optimization_count = 0

        # 内存监控
        self.memory_monitor = MemoryMonitor()

        # 并行处理配置
        self.max_workers = self.config.get('max_workers', 4)
        self.use_process_pool = self.config.get('use_process_pool', False)

        self.logger.info(f"动态分块处理器初始化完成，初始分块大小: {self.current_chunk_size}")

    def optimize_chunked_processing(self, data_source: str, total_records: int = 1000000) -> Dict[str, Any]:
        """优化分块处理性能"""
        self.logger.info(f"开始优化分块处理: {data_source}, 总记录数: {total_records}")

        results = {
            'data_source': data_source,
            'total_records': total_records,
            'initial_chunk_size': self.adaptive_config.min_chunk_size,
            'final_chunk_size': 0,
            'optimization_iterations': 0,
            'performance_improvements': [],
            'memory_optimizations': [],
            'processing_metrics': [],
            'adaptive_adjustments': []
        }

        start_time = time.time()

        try:
            # 生成测试数据
            test_data = self._generate_test_data(total_records)

            # 自适应分块处理
            processed_data = self._adaptive_chunk_processing(test_data, results)

            # 记录最终结果
            results['final_chunk_size'] = self.current_chunk_size
            results['total_processing_time'] = time.time() - start_time
            results['total_records_processed'] = self.total_records_processed
            results['optimization_count'] = self.optimization_count

            # 计算性能提升
            results['performance_improvement'] = self._calculate_performance_improvement(results)

            self.logger.info(f"分块处理优化完成: {results}")

        except Exception as e:
            self.logger.error(f"分块处理优化失败: {e}")
            results['error'] = str(e)

        return results

    def _adaptive_chunk_processing(self, data: pd.DataFrame, results: Dict[str, Any]) -> pd.DataFrame:
        """自适应分块处理"""
        processed_chunks = []
        iteration = 0

        while len(data) > 0:
            iteration += 1
            self.logger.info(f"处理迭代 {iteration}, 当前分块大小: {self.current_chunk_size}")

            # 获取当前分块
            chunk_size = min(self.current_chunk_size, len(data))
            chunk = data.iloc[:chunk_size]
            data = data.iloc[chunk_size:]

            # 处理分块
            chunk_start_time = time.time()
            processed_chunk = self._process_chunk_optimized(chunk)
            chunk_processing_time = time.time() - chunk_start_time

            # 记录性能指标
            memory_usage = self.memory_monitor.get_memory_usage()
            metrics = ChunkPerformanceMetrics(
                chunk_size=len(chunk),
                processing_time=chunk_processing_time,
                memory_usage_mb=memory_usage['rss_mb'],
                throughput_records_per_sec=len(
                    chunk) / chunk_processing_time if chunk_processing_time > 0 else 0,
                memory_efficiency=len(
                    chunk) / memory_usage['rss_mb'] if memory_usage['rss_mb'] > 0 else 0,
                cpu_utilization=psutil.cpu_percent()
            )

            # 更新性能历史
            self.performance_history.append(metrics)
            results['processing_metrics'].append({
                'iteration': iteration,
                'chunk_size': metrics.chunk_size,
                'processing_time': metrics.processing_time,
                'memory_usage_mb': metrics.memory_usage_mb,
                'throughput_records_per_sec': metrics.throughput_records_per_sec,
                'memory_efficiency': metrics.memory_efficiency,
                'cpu_utilization': metrics.cpu_utilization
            })

            # 自适应调整分块大小
            if len(self.performance_history) >= 3:
                self._adaptive_chunk_size_adjustment(results)

            processed_chunks.append(processed_chunk)
            self.total_records_processed += len(chunk)
            self.total_processing_time += chunk_processing_time

            # 定期垃圾回收
            if iteration % 5 == 0:
                gc.collect()

        # 合并处理结果
        return pd.concat(processed_chunks, ignore_index=True)

    def _process_chunk_optimized(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """优化的分块处理"""
        # 使用并行处理
        if self.use_process_pool and len(chunk) > 10000:
            return self._process_chunk_parallel(chunk)
        else:
            return self._process_chunk_sequential(chunk)

    def _process_chunk_sequential(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """顺序处理分块"""
        processed_chunk = chunk.copy()

        # 计算技术指标（优化版本）
        if len(chunk) >= 20:
            processed_chunk['sma_5'] = processed_chunk['close'].rolling(
                window=5, min_periods=1).mean()
            processed_chunk['sma_20'] = processed_chunk['close'].rolling(
                window=20, min_periods=1).mean()
            processed_chunk['volatility'] = processed_chunk['close'].pct_change().rolling(
                window=20, min_periods=1).std()
        else:
            processed_chunk['sma_5'] = processed_chunk['close']
            processed_chunk['sma_20'] = processed_chunk['close']
            processed_chunk['volatility'] = 0.0

        # 计算收益率
        processed_chunk['returns'] = processed_chunk['close'].pct_change()

        # 添加更多技术指标
        processed_chunk['rsi'] = self._calculate_rsi(processed_chunk['close'])
        processed_chunk['macd'] = self._calculate_macd(processed_chunk['close'])

        return processed_chunk

    def _process_chunk_parallel(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """并行处理分块"""
        # 将分块分割成更小的子分块
        sub_chunk_size = max(1000, len(chunk) // self.max_workers)
        sub_chunks = [chunk.iloc[i:i+sub_chunk_size] for i in range(0, len(chunk), sub_chunk_size)]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_chunk_sequential, sub_chunk)
                       for sub_chunk in sub_chunks]
            processed_sub_chunks = [future.result() for future in futures]

        return pd.concat(processed_sub_chunks, ignore_index=True)

    def _adaptive_chunk_size_adjustment(self, results: Dict[str, Any]) -> None:
        """自适应分块大小调整"""
        if len(self.performance_history) < 3:
            return

        # 计算最近性能指标的平均值
        recent_metrics = list(self.performance_history)[-3:]
        avg_processing_time = np.mean([m.processing_time for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_records_per_sec for m in recent_metrics])

        # 性能评估
        processing_time_ratio = avg_processing_time / self.adaptive_config.target_processing_time
        memory_usage_ratio = avg_memory_usage / self.adaptive_config.target_memory_usage_mb

        # 调整策略
        adjustment_needed = False
        adjustment_reason = ""

        if processing_time_ratio > 1.5:  # 处理时间过长
            new_chunk_size = int(self.current_chunk_size / self.adaptive_config.adjustment_factor)
            adjustment_needed = True
            adjustment_reason = "processing_time_too_high"
        elif processing_time_ratio < 0.5 and memory_usage_ratio < 0.8:  # 处理时间短且内存充足
            new_chunk_size = int(self.current_chunk_size * self.adaptive_config.adjustment_factor)
            adjustment_needed = True
            adjustment_reason = "performance_optimization"
        elif memory_usage_ratio > 1.2:  # 内存使用过高
            new_chunk_size = int(self.current_chunk_size / self.adaptive_config.adjustment_factor)
            adjustment_needed = True
            adjustment_reason = "memory_usage_too_high"

        if adjustment_needed:
            # 确保分块大小在合理范围内
            new_chunk_size = max(self.adaptive_config.min_chunk_size,
                                 min(self.adaptive_config.max_chunk_size, new_chunk_size))

            if new_chunk_size != self.current_chunk_size:
                old_chunk_size = self.current_chunk_size
                self.current_chunk_size = new_chunk_size
                self.optimization_count += 1

                # 记录调整
                results['adaptive_adjustments'].append({
                    'iteration': len(results['processing_metrics']),
                    'old_chunk_size': old_chunk_size,
                    'new_chunk_size': new_chunk_size,
                    'reason': adjustment_reason,
                    'avg_processing_time': avg_processing_time,
                    'avg_memory_usage_mb': avg_memory_usage,
                    'avg_throughput': avg_throughput
                })

                self.logger.info(
                    f"分块大小调整: {old_chunk_size} -> {new_chunk_size}, 原因: {adjustment_reason}")

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line

    def _generate_test_data(self, size: int) -> pd.DataFrame:
        """生成测试数据"""
        self.logger.info(f"生成 {size} 行测试数据...")

        # 生成时间序列
        dates = pd.date_range(start='2020-01-01', periods=size, freq='1min')

        # 生成股票数据
        data = {
            'timestamp': dates,
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], size),
            'open': np.random.uniform(100, 500, size),
            'high': np.random.uniform(100, 500, size),
            'low': np.random.uniform(100, 500, size),
            'close': np.random.uniform(100, 500, size),
            'volume': np.random.randint(1000, 100000, size)
        }

        return pd.DataFrame(data)

    def _calculate_performance_improvement(self, results: Dict[str, Any]) -> Dict[str, float]:
        """计算性能提升"""
        if len(results['processing_metrics']) < 2:
            return {'throughput_improvement': 0.0, 'memory_efficiency_improvement': 0.0}

        # 计算吞吐量提升
        initial_throughput = results['processing_metrics'][0]['throughput_records_per_sec']
        final_throughput = results['processing_metrics'][-1]['throughput_records_per_sec']
        throughput_improvement = ((final_throughput - initial_throughput) /
                                  initial_throughput * 100) if initial_throughput > 0 else 0

        # 计算内存效率提升
        initial_memory_efficiency = results['processing_metrics'][0]['memory_efficiency']
        final_memory_efficiency = results['processing_metrics'][-1]['memory_efficiency']
        memory_efficiency_improvement = ((final_memory_efficiency - initial_memory_efficiency) /
                                         initial_memory_efficiency * 100) if initial_memory_efficiency > 0 else 0

        return {
            'throughput_improvement': throughput_improvement,
            'memory_efficiency_improvement': memory_efficiency_improvement
        }


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }


def main():
    """主函数"""
    print("=" * 60)
    print("数据层分块处理优化")
    print("=" * 60)

    # 配置
    config = {
        'min_chunk_size': 1000,
        'max_chunk_size': 50000,
        'target_processing_time': 3.0,
        'target_memory_usage_mb': 400.0,
        'performance_window': 5,
        'adjustment_factor': 1.3,
        'max_workers': 4,
        'use_process_pool': False
    }

    # 创建处理器
    processor = DynamicChunkProcessor(config)

    # 执行优化
    results = processor.optimize_chunked_processing('test_source', total_records=500000)

    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'optimization_results': results,
        'summary': {
            'total_records_processed': results['total_records_processed'],
            'total_processing_time': results['total_processing_time'],
            'optimization_iterations': results['optimization_count'],
            'initial_chunk_size': results['initial_chunk_size'],
            'final_chunk_size': results['final_chunk_size'],
            'performance_improvement': results.get('performance_improvement', {}),
            'adaptive_adjustments_count': len(results['adaptive_adjustments'])
        }
    }

    # 保存报告
    report_file = f"reports/chunked_processing_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n优化完成！")
    print(f"总记录数: {results['total_records_processed']}")
    print(f"总处理时间: {results['total_processing_time']:.2f}秒")
    print(f"优化迭代次数: {results['optimization_count']}")
    print(f"初始分块大小: {results['initial_chunk_size']}")
    print(f"最终分块大小: {results['final_chunk_size']}")
    print(f"自适应调整次数: {len(results['adaptive_adjustments'])}")

    if 'performance_improvement' in results:
        improvement = results['performance_improvement']
        print(f"吞吐量提升: {improvement.get('throughput_improvement', 0):.2f}%")
        print(f"内存效率提升: {improvement.get('memory_efficiency_improvement', 0):.2f}%")

    print(f"\n详细报告已保存到: {report_file}")

    return report


if __name__ == "__main__":
    main()
