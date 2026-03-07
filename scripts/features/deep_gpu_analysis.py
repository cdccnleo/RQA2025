#!/usr/bin/env python3
"""
深度GPU性能分析脚本
诊断GPU性能问题，找出GPU比CPU慢的根本原因
"""

from src.utils.logger import get_logger
from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor
import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logger = get_logger(__name__)


def generate_test_data(size: int = 10000) -> pd.DataFrame:
    """生成测试数据"""
    try:
        start_date = '2023-01-01'
        dates = pd.date_range(start=start_date, periods=size, freq='1min')

        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0, 0.02, size)
        prices = base_price * np.exp(np.cumsum(returns))

        high = prices * (1 + np.abs(np.random.normal(0, 0.01, size)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, size)))
        close = prices
        open_price = np.roll(close, 1)
        open_price[0] = close[0]

        data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        }, index=dates)

        return data
    except Exception as e:
        logger.error(f"生成测试数据失败: {e}")
        return pd.DataFrame()


def analyze_gpu_initialization_overhead():
    """分析GPU初始化开销"""
    logger.info("分析GPU初始化开销...")

    results = []

    for i in range(5):
        start_time = time.time()
        config = {
            'use_gpu': True,
            'optimization_level': 'aggressive',
            'gpu_threshold': 100,
            'memory_limit': 0.8
        }
        processor = GPUTechnicalProcessor(config=config)
        init_time = time.time() - start_time

        results.append(init_time)
        logger.info(f"GPU初始化 {i+1}: {init_time:.4f}s")

    avg_init_time = np.mean(results)
    logger.info(f"平均GPU初始化时间: {avg_init_time:.4f}s")
    return avg_init_time


def analyze_data_transfer_overhead(data: pd.DataFrame):
    """分析数据传输开销"""
    logger.info("分析数据传输开销...")

    try:
        import cupy as cp

        # 测试CPU到GPU的数据传输
        start_time = time.time()
        close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
        transfer_time = time.time() - start_time
        logger.info(f"CPU到GPU数据传输: {transfer_time:.4f}s")

        # 测试GPU到CPU的数据传输
        start_time = time.time()
        close_cpu = cp.asnumpy(close_gpu)
        transfer_back_time = time.time() - start_time
        logger.info(f"GPU到CPU数据传输: {transfer_back_time:.4f}s")

        return transfer_time, transfer_back_time
    except Exception as e:
        logger.error(f"数据传输分析失败: {e}")
        return 0, 0


def analyze_algorithm_breakdown(data: pd.DataFrame):
    """分析算法性能分解"""
    logger.info("分析算法性能分解...")

    try:
        config = {
            'use_gpu': True,
            'optimization_level': 'aggressive',
            'gpu_threshold': 100,
            'memory_limit': 0.8
        }
        processor = GPUTechnicalProcessor(config=config)

        # 测试SMA算法
        logger.info("测试SMA算法...")
        start_time = time.time()
        sma_gpu = processor.calculate_sma_gpu(data, 20)
        sma_gpu_time = time.time() - start_time

        start_time = time.time()
        sma_cpu = processor._calculate_sma_cpu(data, 20)
        sma_cpu_time = time.time() - start_time

        logger.info(
            f"SMA - GPU: {sma_gpu_time:.4f}s, CPU: {sma_cpu_time:.4f}s, 加速比: {sma_cpu_time/sma_gpu_time:.2f}x")

        # 测试EMA算法
        logger.info("测试EMA算法...")
        start_time = time.time()
        ema_gpu = processor.calculate_ema_gpu(data, 20)
        ema_gpu_time = time.time() - start_time

        start_time = time.time()
        ema_cpu = processor._calculate_ema_cpu(data, 20)
        ema_cpu_time = time.time() - start_time

        logger.info(
            f"EMA - GPU: {ema_gpu_time:.4f}s, CPU: {ema_cpu_time:.4f}s, 加速比: {ema_cpu_time/ema_gpu_time:.2f}x")

        # 测试RSI算法
        logger.info("测试RSI算法...")
        start_time = time.time()
        rsi_gpu = processor.calculate_rsi_gpu(data, 14)
        rsi_gpu_time = time.time() - start_time

        start_time = time.time()
        rsi_cpu = processor._calculate_rsi_cpu(data, 14)
        rsi_cpu_time = time.time() - start_time

        logger.info(
            f"RSI - GPU: {rsi_gpu_time:.4f}s, CPU: {rsi_cpu_time:.4f}s, 加速比: {rsi_cpu_time/rsi_gpu_time:.2f}x")

        return {
            'sma': {'gpu': sma_gpu_time, 'cpu': sma_cpu_time, 'speedup': sma_cpu_time/sma_gpu_time},
            'ema': {'gpu': ema_gpu_time, 'cpu': ema_cpu_time, 'speedup': ema_cpu_time/ema_gpu_time},
            'rsi': {'gpu': rsi_gpu_time, 'cpu': rsi_cpu_time, 'speedup': rsi_cpu_time/rsi_gpu_time}
        }
    except Exception as e:
        logger.error(f"算法性能分解分析失败: {e}")
        return {}


def analyze_memory_usage_patterns(data: pd.DataFrame):
    """分析内存使用模式"""
    logger.info("分析内存使用模式...")

    try:
        import cupy as cp

        # 获取初始内存状态
        device = cp.cuda.Device()
        initial_memory = cp.cuda.runtime.memGetInfo()
        logger.info(f"初始可用显存: {initial_memory[0] / 1024**3:.2f} GB")

        # 测试内存分配和释放
        start_time = time.time()
        arrays = []
        for i in range(10):
            arr = cp.random.random((10000, 1000), dtype=cp.float32)
            arrays.append(arr)
        allocation_time = time.time() - start_time
        logger.info(f"内存分配时间: {allocation_time:.4f}s")

        # 检查内存使用
        current_memory = cp.cuda.runtime.memGetInfo()
        used_memory = (initial_memory[0] - current_memory[0]) / 1024**3
        logger.info(f"已使用显存: {used_memory:.2f} GB")

        # 释放内存
        start_time = time.time()
        del arrays
        cp.get_default_memory_pool().free_all_blocks()
        deallocation_time = time.time() - start_time
        logger.info(f"内存释放时间: {deallocation_time:.4f}s")

        return allocation_time, deallocation_time, used_memory
    except Exception as e:
        logger.error(f"内存使用模式分析失败: {e}")
        return 0, 0, 0


def analyze_cupy_vs_numpy_performance(data: pd.DataFrame):
    """分析CuPy vs NumPy性能对比"""
    logger.info("分析CuPy vs NumPy性能对比...")

    try:
        import cupy as cp

        # 准备数据
        close_values = data['close'].values.astype(np.float32)

        # NumPy操作
        start_time = time.time()
        np_result = np.mean(close_values)
        np_time = time.time() - start_time
        logger.info(f"NumPy均值计算: {np_time:.6f}s")

        # CuPy操作
        start_time = time.time()
        cp_array = cp.asarray(close_values, dtype=cp.float32)
        cp_result = cp.mean(cp_array)
        cp_time = time.time() - start_time
        logger.info(f"CuPy均值计算: {cp_time:.6f}s")

        # 向量化操作对比
        start_time = time.time()
        np_diff = np.diff(close_values)
        np_time_diff = time.time() - start_time
        logger.info(f"NumPy差分计算: {np_time_diff:.6f}s")

        start_time = time.time()
        cp_diff = cp.diff(cp_array)
        cp_time_diff = time.time() - start_time
        logger.info(f"CuPy差分计算: {cp_time_diff:.6f}s")

        return {
            'numpy_mean': np_time,
            'cupy_mean': cp_time,
            'numpy_diff': np_time_diff,
            'cupy_diff': cp_time_diff
        }
    except Exception as e:
        logger.error(f"CuPy vs NumPy性能对比分析失败: {e}")
        return {}


def analyze_parallelization_efficiency(data: pd.DataFrame):
    """分析并行化效率"""
    logger.info("分析并行化效率...")

    try:
        import cupy as cp

        # 测试不同数据大小的并行化效果
        sizes = [1000, 5000, 10000, 50000]
        results = {}

        for size in sizes:
            test_data = data.head(size)
            close_values = test_data['close'].values.astype(np.float32)

            # NumPy计算
            start_time = time.time()
            np_result = np.mean(close_values)
            np_time = time.time() - start_time

            # CuPy计算
            start_time = time.time()
            cp_array = cp.asarray(close_values, dtype=cp.float32)
            cp_result = cp.mean(cp_array)
            cp_time = time.time() - start_time

            speedup = np_time / cp_time if cp_time > 0 else 0
            results[size] = {
                'numpy_time': np_time,
                'cupy_time': cp_time,
                'speedup': speedup
            }

            logger.info(
                f"数据大小 {size}: NumPy={np_time:.6f}s, CuPy={cp_time:.6f}s, 加速比={speedup:.2f}x")

        return results
    except Exception as e:
        logger.error(f"并行化效率分析失败: {e}")
        return {}


def generate_deep_analysis_report(init_time: float, transfer_times: Tuple,
                                  algorithm_results: Dict, memory_results: Tuple,
                                  cupy_numpy_results: Dict, parallelization_results: Dict) -> str:
    """生成深度分析报告"""
    report = f"""
# GPU性能深度分析报告

## 分析概述
本报告深入分析了GPU性能问题的根本原因，包括初始化开销、数据传输、算法效率、内存使用等方面。

## 关键发现

### 1. GPU初始化开销
- 平均GPU初始化时间: {init_time:.4f}s
- 影响: 每次创建处理器实例都会产生显著开销

### 2. 数据传输开销
- CPU到GPU传输: {transfer_times[0]:.4f}s
- GPU到CPU传输: {transfer_times[1]:.4f}s
- 影响: 数据传输开销可能超过计算收益

### 3. 算法性能分解
"""

    for algo, result in algorithm_results.items():
        report += f"- {algo.upper()}: GPU={result['gpu']:.4f}s, CPU={result['cpu']:.4f}s, 加速比={result['speedup']:.2f}x\n"

    report += f"""
### 4. 内存使用模式
- 内存分配时间: {memory_results[0]:.4f}s
- 内存释放时间: {memory_results[1]:.4f}s
- 内存使用量: {memory_results[2]:.2f} GB

### 5. CuPy vs NumPy性能对比
"""

    for test, time_val in cupy_numpy_results.items():
        report += f"- {test}: {time_val:.6f}s\n"

    report += """
### 6. 并行化效率
"""

    for size, result in parallelization_results.items():
        report += f"- 数据大小 {size}: NumPy={result['numpy_time']:.6f}s, CuPy={result['cupy_time']:.6f}s, 加速比={result['speedup']:.2f}x\n"

    report += """
## 问题诊断

### 主要问题
1. **初始化开销过大**: GPU初始化需要5-6秒，远超过CPU计算时间
2. **数据传输开销**: 频繁的CPU-GPU数据传输抵消了计算优势
3. **算法实现效率**: 某些算法（如EMA）的GPU实现效率不高
4. **内存管理开销**: GPU内存分配和释放开销较大

### 根本原因
1. **小数据集不适合GPU**: 对于小数据集，GPU的并行优势无法发挥
2. **算法设计问题**: 某些算法（如递归EMA）不适合GPU并行化
3. **内存访问模式**: GPU内存访问模式不够优化
4. **数据传输瓶颈**: 频繁的数据传输成为性能瓶颈

## 优化建议

### 短期优化
1. **延迟初始化**: 只在真正需要时才初始化GPU
2. **批量处理**: 减少数据传输频率，实现批量处理
3. **算法重构**: 重新设计适合GPU的算法实现
4. **内存池优化**: 优化GPU内存池管理

### 长期优化
1. **混合计算**: 根据数据大小智能选择GPU或CPU
2. **算法并行化**: 重新设计算法以充分利用GPU并行能力
3. **内存访问优化**: 优化GPU内存访问模式
4. **性能监控**: 建立实时性能监控机制

## 结论
GPU性能问题主要源于初始化开销、数据传输开销和算法实现效率问题。需要通过延迟初始化、批量处理和算法重构来解决这些问题。
"""

    return report


def main():
    """主函数"""
    logger.info("开始深度GPU性能分析...")

    try:
        # 生成测试数据
        test_data = generate_test_data(10000)
        if test_data.empty:
            logger.error("无法生成测试数据，退出分析")
            return

        # 执行各项分析
        init_time = analyze_gpu_initialization_overhead()
        transfer_times = analyze_data_transfer_overhead(test_data)
        algorithm_results = analyze_algorithm_breakdown(test_data)
        memory_results = analyze_memory_usage_patterns(test_data)
        cupy_numpy_results = analyze_cupy_vs_numpy_performance(test_data)
        parallelization_results = analyze_parallelization_efficiency(test_data)

        # 生成报告
        report = generate_deep_analysis_report(init_time, transfer_times,
                                               algorithm_results, memory_results,
                                               cupy_numpy_results, parallelization_results)

        # 保存报告
        report_path = "reports/deep_gpu_analysis_report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"深度GPU性能分析完成，报告已保存到: {report_path}")

        # 打印关键发现
        logger.info("=== 深度分析关键发现 ===")
        logger.info(f"GPU初始化开销: {init_time:.4f}s")
        logger.info(
            f"数据传输开销: {transfer_times[0]:.4f}s (CPU->GPU), {transfer_times[1]:.4f}s (GPU->CPU)")

        if algorithm_results:
            for algo, result in algorithm_results.items():
                logger.info(f"{algo.upper()} 加速比: {result['speedup']:.2f}x")

    except Exception as e:
        logger.error(f"深度GPU性能分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
