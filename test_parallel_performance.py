#!/usr/bin/env python3
"""
并行计算性能测试脚本

测试并行特征计算的性能提升效果
"""

import asyncio
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.core.parallel_calculator import (
    ParallelFeatureCalculator,
    ParallelConfig,
    benchmark_parallel_calculation
)
from src.data_management import get_stock_pool_manager


def test_sequential_vs_parallel():
    """测试串行 vs 并行计算性能"""
    print("=" * 60)
    print("测试1: 串行 vs 并行计算性能对比")
    print("=" * 60)
    
    # 获取股票池
    pool_manager = get_stock_pool_manager()
    all_stocks = pool_manager.get_all_stocks()
    
    if len(all_stocks) == 0:
        print("⚠️ 没有可用股票，使用测试数据")
        test_symbols = ["002837", "688702"] * 5  # 10只测试股票
    else:
        # 使用前20只股票进行测试
        test_symbols = all_stocks[:min(20, len(all_stocks))]
    
    print(f"测试股票数: {len(test_symbols)}")
    print(f"股票列表: {test_symbols[:5]}...")
    
    end_date = "2026-02-13"
    
    # 测试串行计算（1个worker）
    print("\n--- 串行计算 (1 worker) ---")
    config_seq = ParallelConfig(max_workers=1, use_processes=False)
    calc_seq = ParallelFeatureCalculator(config_seq)
    
    start_time = time.time()
    result_seq = calc_seq.calculate_batch(test_symbols, end_date)
    elapsed_seq = time.time() - start_time
    calc_seq.shutdown()
    
    print(f"耗时: {elapsed_seq:.2f} 秒")
    print(f"成功: {result_seq['stats']['completed']}/{result_seq['stats']['total']}")
    print(f"平均速率: {result_seq['stats']['avg_rate']:.2f} 只/秒")
    
    # 测试并行计算（4个worker）
    print("\n--- 并行计算 (4 workers) ---")
    config_par = ParallelConfig(max_workers=4, use_processes=True)
    calc_par = ParallelFeatureCalculator(config_par)
    
    start_time = time.time()
    result_par = calc_par.calculate_batch(test_symbols, end_date)
    elapsed_par = time.time() - start_time
    calc_par.shutdown()
    
    print(f"耗时: {elapsed_par:.2f} 秒")
    print(f"成功: {result_par['stats']['completed']}/{result_par['stats']['total']}")
    print(f"平均速率: {result_par['stats']['avg_rate']:.2f} 只/秒")
    
    # 计算加速比
    speedup = elapsed_seq / elapsed_par if elapsed_par > 0 else 0
    print(f"\n加速比: {speedup:.2f}x")
    print(f"效率提升: {(speedup - 1) * 100:.1f}%")
    
    return {
        "sequential": {
            "elapsed": elapsed_seq,
            "rate": result_seq['stats']['avg_rate']
        },
        "parallel": {
            "elapsed": elapsed_par,
            "rate": result_par['stats']['avg_rate']
        },
        "speedup": speedup
    }


def test_different_worker_counts():
    """测试不同worker数量的性能"""
    print("\n" + "=" * 60)
    print("测试2: 不同Worker数量的性能对比")
    print("=" * 60)
    
    # 获取股票池
    pool_manager = get_stock_pool_manager()
    all_stocks = pool_manager.get_all_stocks()
    
    if len(all_stocks) == 0:
        print("⚠️ 没有可用股票，跳过测试")
        return None
    
    # 使用50只股票进行测试
    test_symbols = all_stocks[:min(50, len(all_stocks))]
    end_date = "2026-02-13"
    
    worker_counts = [1, 2, 4, 8]
    results = []
    
    for workers in worker_counts:
        print(f"\n--- 测试 {workers} 个workers ---")
        config = ParallelConfig(max_workers=workers)
        calculator = ParallelFeatureCalculator(config)
        
        start_time = time.time()
        result = calculator.calculate_batch(test_symbols, end_date)
        elapsed = time.time() - start_time
        calculator.shutdown()
        
        stats = result['stats']
        results.append({
            'workers': workers,
            'elapsed': elapsed,
            'rate': stats['avg_rate'],
            'success_rate': stats['success_rate']
        })
        
        print(f"耗时: {elapsed:.2f} 秒")
        print(f"速率: {stats['avg_rate']:.2f} 只/秒")
        print(f"成功率: {stats['success_rate'] * 100:.1f}%")
    
    # 找出最优配置
    best = max(results, key=lambda x: x['rate'])
    print(f"\n最优配置: {best['workers']} 个workers")
    print(f"最优速率: {best['rate']:.2f} 只/秒")
    
    return results


def test_async_performance():
    """测试异步IO性能"""
    print("\n" + "=" * 60)
    print("测试3: 异步IO性能测试")
    print("=" * 60)
    
    # 获取股票池
    pool_manager = get_stock_pool_manager()
    all_stocks = pool_manager.get_all_stocks()
    
    if len(all_stocks) == 0:
        print("⚠️ 没有可用股票，跳过测试")
        return None
    
    # 使用20只股票进行测试
    test_symbols = all_stocks[:min(20, len(all_stocks))]
    end_date = "2026-02-13"
    
    async def run_async_test():
        config = ParallelConfig(max_workers=4)
        calculator = ParallelFeatureCalculator(config)
        
        print("开始异步计算...")
        start_time = time.time()
        result = await calculator.calculate_with_async_io(test_symbols, end_date)
        elapsed = time.time() - start_time
        
        if 'error' in result:
            print(f"异步计算出错: {result['error']}")
            return None
        
        stats = result['stats']
        print(f"耗时: {elapsed:.2f} 秒")
        print(f"成功: {stats['completed']}/{stats['total']}")
        print(f"平均速率: {stats['avg_rate']:.2f} 只/秒")
        
        return {
            'elapsed': elapsed,
            'rate': stats['avg_rate']
        }
    
    return asyncio.run(run_async_test())


def estimate_full_market_performance():
    """估算全市场计算性能"""
    print("\n" + "=" * 60)
    print("测试4: 全市场计算性能估算")
    print("=" * 60)
    
    # 获取股票池
    pool_manager = get_stock_pool_manager()
    all_stocks = pool_manager.get_all_stocks()
    total_stocks = len(all_stocks)
    
    if total_stocks == 0:
        total_stocks = 5000  # 假设A股5000只
    
    print(f"全市场股票数: {total_stocks}")
    
    # 使用并行计算测试小样本
    test_count = min(20, total_stocks)
    test_symbols = all_stocks[:test_count] if all_stocks else ["002837"] * test_count
    
    config = ParallelConfig(max_workers=4)
    calculator = ParallelFeatureCalculator(config)
    
    start_time = time.time()
    result = calculator.calculate_batch(test_symbols, "2026-02-13")
    elapsed = time.time() - start_time
    calculator.shutdown()
    
    rate = result['stats']['avg_rate']
    
    # 估算全市场计算时间
    estimated_time = total_stocks / rate if rate > 0 else 0
    
    print(f"\n小样本测试结果:")
    print(f"  测试股票数: {test_count}")
    print(f"  实际耗时: {elapsed:.2f} 秒")
    print(f"  计算速率: {rate:.2f} 只/秒")
    
    print(f"\n全市场估算:")
    print(f"  估算耗时: {estimated_time:.2f} 秒 ({estimated_time/60:.2f} 分钟)")
    print(f"  日均计算: {86400 / estimated_time * total_stocks:.0f} 只/天")
    
    return {
        'total_stocks': total_stocks,
        'estimated_time': estimated_time,
        'rate': rate
    }


if __name__ == "__main__":
    print("开始并行计算性能测试...")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 测试1: 串行 vs 并行
        result1 = test_sequential_vs_parallel()
        
        # 测试2: 不同worker数量
        result2 = test_different_worker_counts()
        
        # 测试3: 异步IO
        result3 = test_async_performance()
        
        # 测试4: 全市场估算
        result4 = estimate_full_market_performance()
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
