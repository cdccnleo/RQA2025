#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多GPU处理演示脚本

展示多GPU设备检测、数据分发、并行处理和性能对比
"""

from src.utils.logger import get_logger
from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor
from src.features.processors.gpu.multi_gpu_processor import MultiGPUProcessor
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


logger = get_logger(__name__)


def create_large_dataset(size: int = 100000) -> pd.DataFrame:
    """创建大规模测试数据集"""
    logger.info(f"创建 {size:,} 条记录的测试数据集...")

    # 使用简单的整数索引避免日期范围问题
    index = pd.RangeIndex(size)

    # 生成价格数据
    np.random.seed(42)
    base_price = 1000
    price_changes = np.random.randn(size) * 10
    prices = base_price + np.cumsum(price_changes)

    # 生成OHLCV数据
    data = pd.DataFrame({
        'close': prices,
        'open': prices + np.random.randn(size) * 5,
        'high': prices + np.abs(np.random.randn(size) * 10),
        'low': prices - np.abs(np.random.randn(size) * 10),
        'volume': np.random.randint(1000000, 10000000, size)
    }, index=index)

    # 确保high >= max(open, close) 和 low <= min(open, close)
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    logger.info(f"数据集创建完成，形状: {data.shape}")
    return data


def benchmark_single_gpu(data: pd.DataFrame, indicators: List[str]) -> Dict[str, float]:
    """单GPU性能基准测试"""
    logger.info("开始单GPU性能测试...")

    try:
        # 创建单GPU处理器
        single_gpu_config = {
            'use_gpu': True,
            'batch_size': 1000,
            'memory_limit': 0.8,
            'fallback_to_cpu': True,
            'gpu_threshold': 1000,
            'optimization_level': 'balanced'
        }

        single_processor = GPUTechnicalProcessor(config=single_gpu_config)

        # 获取GPU信息
        gpu_info = single_processor.get_gpu_info()
        logger.info(f"单GPU信息: {gpu_info}")

        # 性能测试
        start_time = time.time()
        result = single_processor.calculate_multiple_indicators_gpu(data, indicators)
        end_time = time.time()

        processing_time = end_time - start_time
        data_size = len(data)
        throughput = data_size / processing_time

        logger.info(f"单GPU处理完成:")
        logger.info(f"  处理时间: {processing_time:.2f} 秒")
        logger.info(f"  数据量: {data_size:,} 条记录")
        logger.info(f"  吞吐量: {throughput:,.0f} 记录/秒")
        logger.info(f"  结果形状: {result.shape}")

        return {
            'processing_time': processing_time,
            'throughput': throughput,
            'data_size': data_size,
            'result_shape': result.shape,
            'gpu_info': gpu_info
        }

    except Exception as e:
        logger.error(f"单GPU测试失败: {e}")
        return {}


def benchmark_multi_gpu(data: pd.DataFrame, indicators: List[str]) -> Dict[str, float]:
    """多GPU性能基准测试"""
    logger.info("开始多GPU性能测试...")

    try:
        # 创建多GPU处理器
        multi_gpu_config = {
            'use_multi_gpu': True,
            'max_gpus': 4,
            'load_balancing': 'round_robin',
            'chunk_size': 50000,
            'memory_threshold': 0.8,
            'fallback_to_single_gpu': True,
            'fallback_to_cpu': True,
            'sync_mode': True,
            'warmup_iterations': 2
        }

        multi_processor = MultiGPUProcessor(config=multi_gpu_config)

        # 获取多GPU信息
        multi_gpu_info = multi_processor.get_multi_gpu_info()
        logger.info(f"多GPU信息: {multi_gpu_info}")

        # 检查多GPU可用性
        if not multi_processor.is_multi_gpu_available():
            logger.warning("多GPU不可用，回退到单GPU模式")
            return benchmark_single_gpu(data, indicators)

        # 性能测试
        start_time = time.time()
        result = multi_processor.calculate_multiple_indicators_multi_gpu(data, indicators)
        end_time = time.time()

        processing_time = end_time - start_time
        data_size = len(data)
        throughput = data_size / processing_time

        logger.info(f"多GPU处理完成:")
        logger.info(f"  处理时间: {processing_time:.2f} 秒")
        logger.info(f"  数据量: {data_size:,} 条记录")
        logger.info(f"  吞吐量: {throughput:,.0f} 记录/秒")
        logger.info(f"  结果形状: {result.shape}")
        logger.info(f"  使用GPU数量: {len(multi_processor.available_gpus)}")

        return {
            'processing_time': processing_time,
            'throughput': throughput,
            'data_size': data_size,
            'result_shape': result.shape,
            'gpu_count': len(multi_processor.available_gpus),
            'multi_gpu_info': multi_gpu_info
        }

    except Exception as e:
        logger.error(f"多GPU测试失败: {e}")
        return {}


def compare_performance(single_gpu_results: Dict[str, Any],
                        multi_gpu_results: Dict[str, Any]) -> Dict[str, Any]:
    """性能对比分析"""
    logger.info("开始性能对比分析...")

    if not single_gpu_results or not multi_gpu_results:
        logger.warning("性能对比数据不完整")
        return {}

    # 计算性能提升
    speedup = single_gpu_results['processing_time'] / multi_gpu_results['processing_time']
    throughput_improvement = multi_gpu_results['throughput'] / single_gpu_results['throughput']

    comparison = {
        'single_gpu_time': single_gpu_results['processing_time'],
        'multi_gpu_time': multi_gpu_results['processing_time'],
        'speedup': speedup,
        'single_gpu_throughput': single_gpu_results['throughput'],
        'multi_gpu_throughput': multi_gpu_results['throughput'],
        'throughput_improvement': throughput_improvement,
        'gpu_count': multi_gpu_results.get('gpu_count', 1)
    }

    logger.info("性能对比结果:")
    logger.info(f"  单GPU处理时间: {comparison['single_gpu_time']:.2f} 秒")
    logger.info(f"  多GPU处理时间: {comparison['multi_gpu_time']:.2f} 秒")
    logger.info(f"  加速比: {comparison['speedup']:.2f}x")
    logger.info(f"  单GPU吞吐量: {comparison['single_gpu_throughput']:,.0f} 记录/秒")
    logger.info(f"  多GPU吞吐量: {comparison['multi_gpu_throughput']:,.0f} 记录/秒")
    logger.info(f"  吞吐量提升: {comparison['throughput_improvement']:.2f}x")
    logger.info(f"  使用GPU数量: {comparison['gpu_count']}")

    return comparison


def test_different_data_sizes():
    """测试不同数据规模的性能"""
    logger.info("开始不同数据规模的性能测试...")

    data_sizes = [10000, 50000, 100000, 200000]
    indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands']

    results = {}

    for size in data_sizes:
        logger.info(f"\n测试数据规模: {size:,} 条记录")

        # 创建测试数据
        data = create_large_dataset(size)

        # 单GPU测试
        single_results = benchmark_single_gpu(data, indicators)

        # 多GPU测试
        multi_results = benchmark_multi_gpu(data, indicators)

        # 性能对比
        comparison = compare_performance(single_results, multi_results)

        results[size] = {
            'single_gpu': single_results,
            'multi_gpu': multi_results,
            'comparison': comparison
        }

    return results


def test_load_balancing_strategies():
    """测试不同负载均衡策略"""
    logger.info("开始负载均衡策略测试...")

    data = create_large_dataset(100000)
    indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands']

    strategies = ['round_robin', 'memory_based', 'performance_based']
    results = {}

    for strategy in strategies:
        logger.info(f"\n测试负载均衡策略: {strategy}")

        config = {
            'use_multi_gpu': True,
            'max_gpus': 4,
            'load_balancing': strategy,
            'chunk_size': 50000,
            'memory_threshold': 0.8,
            'fallback_to_single_gpu': True,
            'fallback_to_cpu': True,
            'sync_mode': True,
            'warmup_iterations': 1
        }

        try:
            processor = MultiGPUProcessor(config=config)

            if processor.is_multi_gpu_available():
                start_time = time.time()
                result = processor.calculate_multiple_indicators_multi_gpu(data, indicators)
                end_time = time.time()

                processing_time = end_time - start_time
                throughput = len(data) / processing_time

                results[strategy] = {
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'result_shape': result.shape,
                    'gpu_count': len(processor.available_gpus)
                }

                logger.info(f"  {strategy} 策略结果:")
                logger.info(f"    处理时间: {processing_time:.2f} 秒")
                logger.info(f"    吞吐量: {throughput:,.0f} 记录/秒")
                logger.info(f"    使用GPU数量: {len(processor.available_gpus)}")
            else:
                logger.warning(f"  {strategy} 策略: 多GPU不可用")
                results[strategy] = {'error': 'Multi-GPU not available'}

        except Exception as e:
            logger.error(f"  {strategy} 策略测试失败: {e}")
            results[strategy] = {'error': str(e)}

    return results


def test_memory_optimization():
    """测试内存优化效果"""
    logger.info("开始内存优化测试...")

    data = create_large_dataset(150000)
    indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands']

    # 测试不同内存配置
    memory_configs = [
        {'memory_threshold': 0.5, 'chunk_size': 25000},
        {'memory_threshold': 0.7, 'chunk_size': 35000},
        {'memory_threshold': 0.8, 'chunk_size': 50000},
        {'memory_threshold': 0.9, 'chunk_size': 75000}
    ]

    results = {}

    for i, config in enumerate(memory_configs):
        logger.info(f"\n测试内存配置 {i+1}: {config}")

        multi_gpu_config = {
            'use_multi_gpu': True,
            'max_gpus': 4,
            'load_balancing': 'memory_based',
            'memory_threshold': config['memory_threshold'],
            'chunk_size': config['chunk_size'],
            'fallback_to_single_gpu': True,
            'fallback_to_cpu': True,
            'sync_mode': True,
            'warmup_iterations': 1
        }

        try:
            processor = MultiGPUProcessor(config=multi_gpu_config)

            if processor.is_multi_gpu_available():
                # 获取初始内存信息
                initial_info = processor.get_multi_gpu_info()

                start_time = time.time()
                result = processor.calculate_multiple_indicators_multi_gpu(data, indicators)
                end_time = time.time()

                # 获取处理后的内存信息
                final_info = processor.get_multi_gpu_info()

                processing_time = end_time - start_time
                throughput = len(data) / processing_time

                results[f"config_{i+1}"] = {
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'result_shape': result.shape,
                    'gpu_count': len(processor.available_gpus),
                    'initial_memory': initial_info,
                    'final_memory': final_info,
                    'config': config
                }

                logger.info(f"  配置 {i+1} 结果:")
                logger.info(f"    处理时间: {processing_time:.2f} 秒")
                logger.info(f"    吞吐量: {throughput:,.0f} 记录/秒")
                logger.info(f"    使用GPU数量: {len(processor.available_gpus)}")

                # 清理内存
                processor.clear_multi_gpu_memory()
            else:
                logger.warning(f"  配置 {i+1}: 多GPU不可用")
                results[f"config_{i+1}"] = {'error': 'Multi-GPU not available'}

        except Exception as e:
            logger.error(f"  配置 {i+1} 测试失败: {e}")
            results[f"config_{i+1}"] = {'error': str(e)}

    return results


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("多GPU处理演示脚本")
    logger.info("=" * 60)

    try:
        # 1. 基础性能对比测试
        logger.info("\n1. 基础性能对比测试")
        data = create_large_dataset(100000)
        indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands']

        single_results = benchmark_single_gpu(data, indicators)
        multi_results = benchmark_multi_gpu(data, indicators)
        comparison = compare_performance(single_results, multi_results)

        # 2. 不同数据规模测试
        logger.info("\n2. 不同数据规模测试")
        size_results = test_different_data_sizes()

        # 3. 负载均衡策略测试
        logger.info("\n3. 负载均衡策略测试")
        balancing_results = test_load_balancing_strategies()

        # 4. 内存优化测试
        logger.info("\n4. 内存优化测试")
        memory_results = test_memory_optimization()

        # 总结
        logger.info("\n" + "=" * 60)
        logger.info("测试总结")
        logger.info("=" * 60)

        if comparison:
            logger.info(f"最佳加速比: {comparison['speedup']:.2f}x")
            logger.info(f"最佳吞吐量提升: {comparison['throughput_improvement']:.2f}x")

        logger.info("多GPU处理演示完成!")

    except Exception as e:
        logger.error(f"演示脚本执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
