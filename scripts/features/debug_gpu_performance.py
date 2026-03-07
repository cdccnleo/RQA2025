#!/usr/bin/env python3
"""
GPU性能调试脚本
详细分析GPU性能问题
"""
from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_test_data(size: int = 10000) -> pd.DataFrame:
    """生成测试数据"""
    if size > 50000:
        start_date = '2023-01-01'
    else:
        start_date = '2020-01-01'

    dates = pd.date_range(start=start_date, periods=size, freq='D')
    np.random.seed(42)

    base_price = 100.0
    returns = np.random.normal(0, 0.02, size)
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, size)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, size))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, size))),
        'close': prices
    }, index=dates)

    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


def debug_gpu_info():
    """调试GPU信息"""
    print("=== GPU信息调试 ===")

    processor = GPUTechnicalProcessor({
        'use_gpu': True,
        'gpu_threshold': 1000,
        'optimization_level': 'balanced'
    })

    gpu_info = processor.get_gpu_info()
    print(f"GPU信息: {gpu_info}")

    # 检查GPU可用性
    print(f"\nGPU可用性检查:")
    print(f"  GPU可用: {processor.gpu_available}")
    print(f"  CUDA可用: {gpu_info.get('cuda_available', False)}")
    print(f"  GPU数量: {gpu_info.get('gpu_count', 0)}")
    print(f"  内存使用: {gpu_info.get('memory_usage', 0):.2f}%")

    return processor


def debug_data_transfer():
    """调试数据传输性能"""
    print("\n=== 数据传输性能调试 ===")

    try:
        import cupy as cp

        processor = debug_gpu_info()

        # 测试不同数据规模的数据传输时间
        data_sizes = [1000, 5000, 10000, 20000]

        for size in data_sizes:
            print(f"\n测试数据规模: {size}")
            data = generate_test_data(size)

            # 测试CPU到GPU传输时间
            start_time = time.time()
            close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
            transfer_time = time.time() - start_time
            print(f"  CPU到GPU传输时间: {transfer_time:.6f}s")

            # 测试GPU到CPU传输时间
            start_time = time.time()
            close_cpu = cp.asnumpy(close_gpu)
            transfer_back_time = time.time() - start_time
            print(f"  GPU到CPU传输时间: {transfer_back_time:.6f}s")

            # 测试GPU计算时间
            start_time = time.time()
            sma_gpu = cp.convolve(close_gpu, cp.ones(20, dtype=cp.float32) / 20, mode='valid')
            compute_time = time.time() - start_time
            print(f"  GPU计算时间: {compute_time:.6f}s")

            # 测试CPU计算时间
            start_time = time.time()
            sma_cpu = data['close'].rolling(window=20).mean()
            cpu_compute_time = time.time() - start_time
            print(f"  CPU计算时间: {cpu_compute_time:.6f}s")

            # 计算总开销
            total_gpu_time = transfer_time + compute_time + transfer_back_time
            speedup = cpu_compute_time / total_gpu_time if total_gpu_time > 0 else 0
            print(f"  总GPU时间: {total_gpu_time:.6f}s")
            print(f"  加速比: {speedup:.2f}x")

    except Exception as e:
        print(f"数据传输调试失败: {e}")


def debug_memory_usage():
    """调试内存使用"""
    print("\n=== 内存使用调试 ===")

    try:
        import cupy as cp

        # 检查GPU内存信息
        device = cp.cuda.Device()
        total_memory = device.mem_info[1]
        free_memory = device.mem_info[0]
        used_memory = total_memory - free_memory

        print(f"GPU内存信息:")
        print(f"  总内存: {total_memory / (1024**3):.2f} GB")
        print(f"  已用内存: {used_memory / (1024**3):.2f} GB")
        print(f"  可用内存: {free_memory / (1024**3):.2f} GB")
        print(f"  使用率: {used_memory / total_memory * 100:.2f}%")

        # 测试内存分配
        print(f"\n内存分配测试:")
        sizes = [1024, 1024*1024, 10*1024*1024]  # 1KB, 1MB, 10MB

        for size in sizes:
            start_time = time.time()
            try:
                temp_array = cp.zeros(size, dtype=cp.float32)
                allocation_time = time.time() - start_time
                print(f"  分配 {size} 个元素: {allocation_time:.6f}s")
                del temp_array
            except Exception as e:
                print(f"  分配 {size} 个元素失败: {e}")

    except Exception as e:
        print(f"内存调试失败: {e}")


def debug_algorithm_performance():
    """调试算法性能"""
    print("\n=== 算法性能调试 ===")

    processor = debug_gpu_info()

    data = generate_test_data(10000)

    # 测试各个算法的性能
    algorithms = [
        ('SMA', lambda: processor.calculate_sma_gpu(data, window=20)),
        ('EMA', lambda: processor.calculate_ema_gpu(data, window=20)),
        ('RSI', lambda: processor.calculate_rsi_gpu(data, window=14)),
        ('ATR', lambda: processor.calculate_atr_gpu(data, window=14))
    ]

    for name, func in algorithms:
        print(f"\n测试 {name} 算法:")

        # GPU计算
        start_time = time.time()
        try:
            result_gpu = func()
            gpu_time = time.time() - start_time
            print(f"  GPU时间: {gpu_time:.6f}s")
            print(f"  结果类型: {type(result_gpu)}")
            if hasattr(result_gpu, 'shape'):
                print(f"  结果形状: {result_gpu.shape}")
        except Exception as e:
            print(f"  GPU计算失败: {e}")
            gpu_time = float('inf')

        # CPU计算
        start_time = time.time()
        if name == 'SMA':
            result_cpu = data['close'].rolling(window=20).mean()
        elif name == 'EMA':
            result_cpu = data['close'].ewm(span=20).mean()
        elif name == 'RSI':
            delta = data['close'].diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gains = gains.rolling(window=14).mean()
            avg_losses = losses.rolling(window=14).mean()
            rs = avg_gains / avg_losses
            result_cpu = 100 - (100 / (1 + rs))
        elif name == 'ATR':
            high = data['high']
            low = data['low']
            close = data['close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result_cpu = tr.rolling(window=14).mean()

        cpu_time = time.time() - start_time
        print(f"  CPU时间: {cpu_time:.6f}s")

        # 计算加速比
        if gpu_time != float('inf') and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"  加速比: {speedup:.2f}x")
        else:
            print(f"  加速比: 无法计算")


def main():
    """主函数"""
    try:
        print("开始GPU性能调试...")

        # 调试GPU信息
        debug_gpu_info()

        # 调试数据传输
        debug_data_transfer()

        # 调试内存使用
        debug_memory_usage()

        # 调试算法性能
        debug_algorithm_performance()

        print("\nGPU性能调试完成!")

    except Exception as e:
        print(f"调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
