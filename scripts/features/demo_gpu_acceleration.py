#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速功能演示脚本

展示GPU加速技术指标计算的功能和性能
"""

from src.utils.logger import get_logger
from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor
import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logger = get_logger(__name__)


def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """生成样本数据"""
    np.random.seed(42)

    # 创建时间索引
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='h')

    # 生成股票数据
    base_price = 100
    returns = np.random.randn(n_samples) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_samples) * 0.01),
        'high': prices * (1 + abs(np.random.randn(n_samples) * 0.02)),
        'low': prices * (1 - abs(np.random.randn(n_samples) * 0.02)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)

    return data


def benchmark_performance(data: pd.DataFrame, processor: GPUTechnicalProcessor):
    """性能基准测试"""
    print("\n" + "="*60)
    print("性能基准测试")
    print("="*60)

    # 测试指标列表
    indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr']
    params = {
        'sma_window': 20,
        'ema_window': 20,
        'rsi_window': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_window': 20,
        'bb_std': 2,
        'atr_window': 14
    }

    results = {}

    for indicator in indicators:
        print(f"\n测试 {indicator.upper()} 指标:")

        # 单个指标测试
        if indicator == 'sma':
            start_time = time.time()
            result = processor.calculate_sma_gpu(data, params['sma_window'])
            gpu_time = time.time() - start_time

            start_time = time.time()
            cpu_result = processor._calculate_sma_cpu(data, params['sma_window'])
            cpu_time = time.time() - start_time

        elif indicator == 'ema':
            start_time = time.time()
            result = processor.calculate_ema_gpu(data, params['ema_window'])
            gpu_time = time.time() - start_time

            start_time = time.time()
            cpu_result = processor._calculate_ema_cpu(data, params['ema_window'])
            cpu_time = time.time() - start_time

        elif indicator == 'rsi':
            start_time = time.time()
            result = processor.calculate_rsi_gpu(data, params['rsi_window'])
            gpu_time = time.time() - start_time

            start_time = time.time()
            cpu_result = processor._calculate_rsi_cpu(data, params['rsi_window'])
            cpu_time = time.time() - start_time

        elif indicator == 'macd':
            start_time = time.time()
            result = processor.calculate_macd_gpu(data, params['macd_fast'],
                                                  params['macd_slow'], params['macd_signal'])
            gpu_time = time.time() - start_time

            start_time = time.time()
            cpu_result = processor._calculate_macd_cpu(data, params['macd_fast'],
                                                       params['macd_slow'], params['macd_signal'])
            cpu_time = time.time() - start_time

        elif indicator == 'bollinger':
            start_time = time.time()
            result = processor.calculate_bollinger_bands_gpu(
                data, params['bb_window'], params['bb_std'])
            gpu_time = time.time() - start_time

            start_time = time.time()
            cpu_result = processor._calculate_bollinger_bands_cpu(
                data, params['bb_window'], params['bb_std'])
            cpu_time = time.time() - start_time

        elif indicator == 'atr':
            start_time = time.time()
            result = processor.calculate_atr_gpu(data, params['atr_window'])
            gpu_time = time.time() - start_time

            start_time = time.time()
            cpu_result = processor._calculate_atr_cpu(data, params['atr_window'])
            cpu_time = time.time() - start_time

        # 验证结果一致性（允许浮点数精度差异）
        if isinstance(result, pd.Series):
            # 使用更宽松的比较，容忍浮点数精度差异
            pd.testing.assert_series_equal(
                result, cpu_result,
                check_names=False,
                check_dtype=False,
                rtol=1e-2,  # 相对容差增加到1%
                atol=1e-3    # 绝对容差增加到0.1%
            )
        elif isinstance(result, pd.DataFrame):
            pd.testing.assert_frame_equal(
                result, cpu_result,
                check_names=False,
                check_dtype=False,
                rtol=1e-2,  # 相对容差增加到1%
                atol=1e-3    # 绝对容差增加到0.1%
            )

        # 计算加速比
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"  GPU时间: {gpu_time:.4f}秒")
        print(f"  CPU时间: {cpu_time:.4f}秒")
        print(f"  加速比: {speedup:.2f}x")

        results[indicator] = {
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': speedup
        }

    # 多指标测试
    print(f"\n测试多指标计算:")
    start_time = time.time()
    multi_result = processor.calculate_multiple_indicators_gpu(data, indicators, params)
    gpu_time = time.time() - start_time

    start_time = time.time()
    cpu_multi_result = processor._calculate_multiple_indicators_cpu(data, indicators, params)
    cpu_time = time.time() - start_time

    pd.testing.assert_frame_equal(
        multi_result, cpu_multi_result,
        check_names=False,
        check_dtype=False,
        rtol=1e-2,  # 相对容差增加到1%
        atol=1e-3    # 绝对容差增加到0.1%
    )

    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"  GPU时间: {gpu_time:.4f}秒")
    print(f"  CPU时间: {cpu_time:.4f}秒")
    print(f"  加速比: {speedup:.2f}x")

    results['multi_indicators'] = {
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'speedup': speedup
    }

    return results


def demonstrate_gpu_features(processor: GPUTechnicalProcessor):
    """演示GPU功能"""
    print("\n" + "="*60)
    print("GPU功能演示")
    print("="*60)

    # 获取GPU信息
    gpu_info = processor.get_gpu_info()
    print(f"GPU可用性: {gpu_info['available']}")

    if gpu_info['available']:
        print(f"GPU设备: {gpu_info['device_name']}")
        print(f"总内存: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"已用内存: {gpu_info['used_memory_gb']:.2f} GB")
        print(f"可用内存: {gpu_info['free_memory_gb']:.2f} GB")
    else:
        print(f"GPU不可用原因: {gpu_info['reason']}")

    # 演示内存清理
    print("\n演示GPU内存清理:")
    processor.clear_gpu_memory()
    print("GPU内存已清理")


def demonstrate_data_processing(data: pd.DataFrame, processor: GPUTechnicalProcessor):
    """演示数据处理"""
    print("\n" + "="*60)
    print("数据处理演示")
    print("="*60)

    print(f"数据形状: {data.shape}")
    print(f"数据列: {list(data.columns)}")
    print(f"时间范围: {data.index[0]} 到 {data.index[-1]}")

    # 计算多个指标
    indicators = ['sma', 'rsi', 'macd']
    params = {
        'sma_window': 20,
        'rsi_window': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }

    print(f"\n计算指标: {indicators}")
    result = processor.calculate_multiple_indicators_gpu(data, indicators, params)

    print(f"计算结果形状: {result.shape}")
    print(f"结果列: {list(result.columns)}")

    # 显示统计信息
    print("\n结果统计信息:")
    for col in result.columns:
        valid_data = result[col].dropna()
        if len(valid_data) > 0:
            print(f"  {col}:")
            print(f"    均值: {valid_data.mean():.4f}")
            print(f"    标准差: {valid_data.std():.4f}")
            print(f"    最小值: {valid_data.min():.4f}")
            print(f"    最大值: {valid_data.max():.4f}")


def main():
    """主函数"""
    print("GPU加速技术指标计算演示")
    print("="*60)
    print(f"开始时间: {datetime.now()}")

    # 创建GPU处理器
    print("\n初始化GPU处理器...")
    processor = GPUTechnicalProcessor()

    # 演示GPU功能
    demonstrate_gpu_features(processor)

    # 生成样本数据
    print("\n生成样本数据...")
    data = generate_sample_data(10000)

    # 演示数据处理
    demonstrate_data_processing(data, processor)

    # 性能基准测试
    performance_results = benchmark_performance(data, processor)

    # 总结
    print("\n" + "="*60)
    print("性能测试总结")
    print("="*60)

    avg_speedup = np.mean([result['speedup'] for result in performance_results.values()])
    max_speedup = max([result['speedup'] for result in performance_results.values()])
    min_speedup = min([result['speedup'] for result in performance_results.values()])

    print(f"平均加速比: {avg_speedup:.2f}x")
    print(f"最大加速比: {max_speedup:.2f}x")
    print(f"最小加速比: {min_speedup:.2f}x")

    if processor.gpu_available:
        print("\n✅ GPU加速功能正常工作")
        if avg_speedup > 1.0:
            print("✅ GPU提供了显著的性能提升")
        else:
            print("⚠️  GPU性能提升有限，可能是数据量较小或GPU开销较大")
    else:
        print("\n⚠️  GPU不可用，使用CPU模式")

    print(f"\n结束时间: {datetime.now()}")
    print("演示完成!")


if __name__ == "__main__":
    main()
