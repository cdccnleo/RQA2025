#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大数据集GPU加速性能测试脚本

测试GPU加速在不同数据规模下的性能表现
"""

from src.utils.logger import get_logger
from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor
import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logger = get_logger(__name__)


def generate_large_dataset(n_samples: int) -> pd.DataFrame:
    """生成大规模数据集"""
    np.random.seed(42)

    # 创建时间索引
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='min')

    # 生成更真实的股票数据
    base_price = 100
    returns = np.random.randn(n_samples) * 0.001  # 更小的波动
    prices = base_price * np.exp(np.cumsum(returns))

    # 添加趋势和季节性
    trend = np.linspace(0, 0.1, n_samples)
    seasonal = 0.02 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))  # 日季节性
    prices = prices * (1 + trend + seasonal)

    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_samples) * 0.0005),
        'high': prices * (1 + abs(np.random.randn(n_samples) * 0.001)),
        'low': prices * (1 - abs(np.random.randn(n_samples) * 0.001)),
        'close': prices,
        'volume': np.random.randint(1000, 100000, n_samples)
    }, index=dates)

    return data


def benchmark_indicator(processor: GPUTechnicalProcessor, data: pd.DataFrame,
                        indicator: str, params: dict) -> dict:
    """基准测试单个指标"""
    results = {}

    try:
        # GPU测试
        start_time = time.time()
        if indicator == 'sma':
            gpu_result = processor.calculate_sma_gpu(data, params.get('window', 20))
        elif indicator == 'ema':
            gpu_result = processor.calculate_ema_gpu(data, params.get('window', 20))
        elif indicator == 'rsi':
            gpu_result = processor.calculate_rsi_gpu(data, params.get('window', 14))
        elif indicator == 'macd':
            gpu_result = processor.calculate_macd_gpu(data,
                                                      params.get('fast', 12),
                                                      params.get('slow', 26),
                                                      params.get('signal', 9))
        elif indicator == 'bollinger':
            gpu_result = processor.calculate_bollinger_bands_gpu(data,
                                                                 params.get('window', 20),
                                                                 params.get('std', 2))
        elif indicator == 'atr':
            gpu_result = processor.calculate_atr_gpu(data, params.get('window', 14))

        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        results['gpu_success'] = True

    except Exception as e:
        results['gpu_time'] = float('inf')
        results['gpu_success'] = False
        results['gpu_error'] = str(e)

    try:
        # CPU测试
        start_time = time.time()
        if indicator == 'sma':
            cpu_result = processor._calculate_sma_cpu(data, params.get('window', 20))
        elif indicator == 'ema':
            cpu_result = processor._calculate_ema_cpu(data, params.get('window', 20))
        elif indicator == 'rsi':
            cpu_result = processor._calculate_rsi_cpu(data, params.get('window', 14))
        elif indicator == 'macd':
            cpu_result = processor._calculate_macd_cpu(data,
                                                       params.get('fast', 12),
                                                       params.get('slow', 26),
                                                       params.get('signal', 9))
        elif indicator == 'bollinger':
            cpu_result = processor._calculate_bollinger_bands_cpu(data,
                                                                  params.get('window', 20),
                                                                  params.get('std', 2))
        elif indicator == 'atr':
            cpu_result = processor._calculate_atr_cpu(data, params.get('window', 14))

        cpu_time = time.time() - start_time
        results['cpu_time'] = cpu_time
        results['cpu_success'] = True

    except Exception as e:
        results['cpu_time'] = float('inf')
        results['cpu_success'] = False
        results['cpu_error'] = str(e)

    # 计算加速比
    if results['gpu_success'] and results['cpu_success']:
        if results['cpu_time'] > 0:
            results['speedup'] = results['cpu_time'] / results['gpu_time']
        else:
            results['speedup'] = float('inf')
    else:
        results['speedup'] = 0

    return results


def run_scalability_test(processor: GPUTechnicalProcessor):
    """运行可扩展性测试"""
    print("\n" + "="*80)
    print("大数据集GPU加速性能测试")
    print("="*80)

    # 测试数据规模
    dataset_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    # 测试指标和参数
    indicators_config = {
        'sma': {'window': 20},
        'ema': {'window': 20},
        'rsi': {'window': 14},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'bollinger': {'window': 20, 'std': 2},
        'atr': {'window': 14}
    }

    all_results = {}

    for size in dataset_sizes:
        print(f"\n测试数据规模: {size:,} 个数据点")
        print("-" * 50)

        # 生成数据
        data = generate_large_dataset(size)
        print(f"数据形状: {data.shape}")
        print(f"时间范围: {data.index[0]} 到 {data.index[-1]}")

        size_results = {}

        for indicator, params in indicators_config.items():
            print(f"\n测试 {indicator.upper()} 指标...")

            result = benchmark_indicator(processor, data, indicator, params)

            if result['gpu_success'] and result['cpu_success']:
                print(f"  GPU时间: {result['gpu_time']:.4f}秒")
                print(f"  CPU时间: {result['cpu_time']:.4f}秒")
                print(f"  加速比: {result['speedup']:.2f}x")
            else:
                print(
                    f"  测试失败 - GPU: {result.get('gpu_error', 'N/A')}, CPU: {result.get('cpu_error', 'N/A')}")

            size_results[indicator] = result

        all_results[size] = size_results

    return all_results


def analyze_results(results: dict):
    """分析测试结果"""
    print("\n" + "="*80)
    print("性能测试结果分析")
    print("="*80)

    # 创建结果表格
    dataset_sizes = list(results.keys())
    indicators = list(results[dataset_sizes[0]].keys())

    print("\n性能测试结果汇总:")
    print("-" * 120)
    print(f"{'数据规模':<12} {'指标':<12} {'GPU时间(秒)':<12} {'CPU时间(秒)':<12} {'加速比':<10} {'状态':<10}")
    print("-" * 120)

    summary_stats = {}

    for size in dataset_sizes:
        for indicator in indicators:
            result = results[size][indicator]

            if result['gpu_success'] and result['cpu_success']:
                gpu_time = f"{result['gpu_time']:.4f}"
                cpu_time = f"{result['cpu_time']:.4f}"
                speedup = f"{result['speedup']:.2f}x"
                status = "✅"
            else:
                gpu_time = "失败"
                cpu_time = "失败"
                speedup = "N/A"
                status = "❌"

            print(f"{size:<12} {indicator:<12} {gpu_time:<12} {cpu_time:<12} {speedup:<10} {status:<10}")

            # 收集统计信息
            if result['gpu_success'] and result['cpu_success']:
                if indicator not in summary_stats:
                    summary_stats[indicator] = {'speedups': [], 'gpu_times': [], 'cpu_times': []}
                summary_stats[indicator]['speedups'].append(result['speedup'])
                summary_stats[indicator]['gpu_times'].append(result['gpu_time'])
                summary_stats[indicator]['cpu_times'].append(result['cpu_time'])

    # 计算平均统计
    print("\n" + "="*80)
    print("平均性能统计")
    print("="*80)

    for indicator, stats in summary_stats.items():
        if stats['speedups']:
            avg_speedup = np.mean(stats['speedups'])
            max_speedup = np.max(stats['speedups'])
            min_speedup = np.min(stats['speedups'])
            avg_gpu_time = np.mean(stats['gpu_times'])
            avg_cpu_time = np.mean(stats['cpu_times'])

            print(f"\n{indicator.upper()} 指标:")
            print(f"  平均加速比: {avg_speedup:.2f}x")
            print(f"  最大加速比: {max_speedup:.2f}x")
            print(f"  最小加速比: {min_speedup:.2f}x")
            print(f"  平均GPU时间: {avg_gpu_time:.4f}秒")
            print(f"  平均CPU时间: {avg_cpu_time:.4f}秒")


def create_performance_charts(results: dict):
    """创建性能图表"""
    print("\n" + "="*80)
    print("生成性能图表")
    print("="*80)

    dataset_sizes = list(results.keys())
    indicators = list(results[dataset_sizes[0]].keys())

    # 准备数据
    chart_data = []
    for size in dataset_sizes:
        for indicator in indicators:
            result = results[size][indicator]
            if result['gpu_success'] and result['cpu_success']:
                chart_data.append({
                    'dataset_size': size,
                    'indicator': indicator,
                    'gpu_time': result['gpu_time'],
                    'cpu_time': result['cpu_time'],
                    'speedup': result['speedup']
                })

    if not chart_data:
        print("没有有效数据生成图表")
        return

    df = pd.DataFrame(chart_data)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GPU加速性能测试结果', fontsize=16)

    # 1. 加速比对比
    ax1 = axes[0, 0]
    for indicator in indicators:
        indicator_data = df[df['indicator'] == indicator]
        if not indicator_data.empty:
            ax1.plot(indicator_data['dataset_size'], indicator_data['speedup'],
                     marker='o', label=indicator.upper())
    ax1.set_xscale('log')
    ax1.set_xlabel('数据规模')
    ax1.set_ylabel('加速比')
    ax1.set_title('各指标加速比对比')
    ax1.legend()
    ax1.grid(True)

    # 2. GPU vs CPU 时间对比
    ax2 = axes[0, 1]
    for indicator in indicators:
        indicator_data = df[df['indicator'] == indicator]
        if not indicator_data.empty:
            ax2.plot(indicator_data['dataset_size'], indicator_data['gpu_time'],
                     marker='s', label=f'{indicator.upper()} (GPU)')
            ax2.plot(indicator_data['dataset_size'], indicator_data['cpu_time'],
                     marker='^', label=f'{indicator.upper()} (CPU)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('数据规模')
    ax2.set_ylabel('执行时间 (秒)')
    ax2.set_title('GPU vs CPU 执行时间')
    ax2.legend()
    ax2.grid(True)

    # 3. 热力图 - 加速比
    ax3 = axes[1, 0]
    pivot_speedup = df.pivot(index='dataset_size', columns='indicator', values='speedup')
    sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3)
    ax3.set_title('加速比热力图')
    ax3.set_xlabel('指标')
    ax3.set_ylabel('数据规模')

    # 4. 热力图 - GPU时间
    ax4 = axes[1, 1]
    pivot_gpu_time = df.pivot(index='dataset_size', columns='indicator', values='gpu_time')
    sns.heatmap(pivot_gpu_time, annot=True, fmt='.4f', cmap='Blues', ax=ax4)
    ax4.set_title('GPU执行时间热力图 (秒)')
    ax4.set_xlabel('指标')
    ax4.set_ylabel('数据规模')

    plt.tight_layout()

    # 保存图表
    chart_path = 'reports/gpu_performance_charts.png'
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"性能图表已保存到: {chart_path}")

    plt.show()


def main():
    """主函数"""
    print("大数据集GPU加速性能测试")
    print("="*80)
    print(f"开始时间: {datetime.now()}")

    # 创建GPU处理器
    print("\n初始化GPU处理器...")
    processor = GPUTechnicalProcessor()

    # 获取GPU信息
    gpu_info = processor.get_gpu_info()
    print(f"GPU可用性: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"GPU设备: {gpu_info['device_name']}")
        print(f"GPU内存: {gpu_info['total_memory_gb']:.2f} GB")

    # 运行可扩展性测试
    results = run_scalability_test(processor)

    # 分析结果
    analyze_results(results)

    # 创建性能图表
    try:
        create_performance_charts(results)
    except Exception as e:
        print(f"图表生成失败: {e}")

    print(f"\n结束时间: {datetime.now()}")
    print("性能测试完成!")


if __name__ == "__main__":
    main()
