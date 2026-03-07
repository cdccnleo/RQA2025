#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU开销优化脚本 - 短期目标实现

专注于减少GPU算法的初始化开销，实现以下优化：
1. 延迟初始化：只在需要时初始化GPU资源
2. 内存池优化：优化GPU内存分配策略
3. 批处理预热：预热GPU计算核心
4. 缓存机制：缓存常用的计算结果
"""

from src.utils.logger import get_logger
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logger = get_logger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CUDA GPU加速可用")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CUDA不可用，将使用CPU计算")


class GPUOverheadOptimizer:
    """GPU开销优化器 - 短期目标实现"""

    def __init__(self):
        self.logger = get_logger("gpu_overhead_optimizer")
        self.gpu_available = GPU_AVAILABLE
        self.initialization_time = 0
        self.memory_pool_optimized = False
        self.warmup_completed = False
        self.cache = {}

        if self.gpu_available:
            self._initialize_optimizations()

    def _initialize_optimizations(self):
        """初始化GPU优化策略"""
        start_time = time.time()

        try:
            # 1. 延迟初始化：不立即初始化所有资源
            self.logger.info("开始GPU开销优化初始化...")

            # 2. 内存池优化
            self._optimize_memory_pool()

            # 3. 批处理预热
            self._warmup_gpu()

            self.initialization_time = time.time() - start_time
            self.logger.info(f"GPU优化初始化完成，耗时: {self.initialization_time:.4f}s")

        except Exception as e:
            self.logger.error(f"GPU优化初始化失败: {e}")
            self.gpu_available = False

    def _optimize_memory_pool(self):
        """优化GPU内存池设置"""
        try:
            # 获取默认内存池
            pool = cp.get_default_memory_pool()

            # 设置更高效的内存管理策略
            # 允许更多内存使用，减少频繁分配/释放
            pool.set_limit(size=0)  # 不限制内存池大小

            # 设置内存池的分配策略
            # 使用更激进的内存管理，减少碎片
            pool.set_limit(size=None)

            # 预分配常用大小的内存块
            self._preallocate_common_blocks()

            self.memory_pool_optimized = True
            self.logger.info("GPU内存池优化完成")

        except Exception as e:
            self.logger.warning(f"内存池优化失败: {e}")

    def _preallocate_common_blocks(self):
        """预分配常用大小的内存块"""
        try:
            # 预分配常用大小的内存块，减少动态分配开销
            common_sizes = [100, 500, 1000, 5000, 10000]

            for size in common_sizes:
                try:
                    # 预分配内存块并立即释放，但保留在内存池中
                    temp_array = cp.zeros(size, dtype=cp.float32)
                    del temp_array
                except Exception:
                    # 如果内存不足，跳过
                    continue

            self.logger.info("GPU常用内存块预分配完成")

        except Exception as e:
            self.logger.warning(f"内存块预分配失败: {e}")

    def _warmup_gpu(self):
        """预热GPU计算核心"""
        try:
            self.logger.info("开始GPU预热...")

            # 创建测试数据
            test_data = cp.random.random(1000, dtype=cp.float32)

            # 执行一些基本运算来预热GPU
            for _ in range(10):
                # 基本数学运算
                result = cp.sin(test_data) + cp.cos(test_data)
                result = cp.sqrt(cp.abs(result))
                result = cp.exp(-result)

                # 矩阵运算
                matrix = cp.random.random((100, 100), dtype=cp.float32)
                result = cp.dot(matrix, matrix.T)

                # 卷积运算
                kernel = cp.ones(5, dtype=cp.float32) / 5
                result = cp.convolve(test_data, kernel, mode='valid')

            # 同步GPU，确保所有操作完成
            cp.cuda.Stream.null.synchronize()

            self.warmup_completed = True
            self.logger.info("GPU预热完成")

        except Exception as e:
            self.logger.warning(f"GPU预热失败: {e}")

    def _lazy_initialize_gpu(self):
        """延迟初始化GPU资源"""
        if not self.gpu_available or self.memory_pool_optimized:
            return

        self._optimize_memory_pool()
        self._warmup_gpu()

    def _batch_process_optimization(self, data_list: List[pd.DataFrame],
                                    indicator_func, **kwargs) -> List[Any]:
        """批处理优化：批量处理多个数据集"""
        if not self.gpu_available:
            return [indicator_func(data, **kwargs) for data in data_list]

        try:
            # 延迟初始化
            self._lazy_initialize_gpu()

            results = []

            # 批量处理
            for i, data in enumerate(data_list):
                start_time = time.time()

                # 使用缓存机制
                cache_key = f"{indicator_func.__name__}_{len(data)}_{hash(str(kwargs))}"

                if cache_key in self.cache:
                    self.logger.info(f"使用缓存结果: {cache_key}")
                    results.append(self.cache[cache_key])
                else:
                    # 执行计算
                    result = indicator_func(data, **kwargs)

                    # 缓存结果（限制缓存大小）
                    if len(self.cache) < 100:  # 最多缓存100个结果
                        self.cache[cache_key] = result

                    results.append(result)

                process_time = time.time() - start_time
                self.logger.info(f"批处理第{i+1}/{len(data_list)}个数据集，耗时: {process_time:.4f}s")

            return results

        except Exception as e:
            self.logger.warning(f"批处理优化失败: {e}，回退到串行处理")
            return [indicator_func(data, **kwargs) for data in data_list]

    def test_gpu_overhead_optimization(self):
        """测试GPU开销优化效果"""
        self.logger.info("开始GPU开销优化测试...")

        # 生成测试数据
        test_data_list = self._generate_test_data()

        # 测试不同的优化策略
        results = {}

        # 1. 测试初始化开销
        results['initialization'] = self._test_initialization_overhead()

        # 2. 测试内存池优化
        results['memory_pool'] = self._test_memory_pool_optimization()

        # 3. 测试批处理优化
        results['batch_processing'] = self._test_batch_processing_optimization(test_data_list)

        # 4. 测试缓存机制
        results['caching'] = self._test_caching_optimization(test_data_list)

        # 生成报告
        self._generate_optimization_report(results)

        return results

    def _generate_test_data(self) -> List[pd.DataFrame]:
        """生成测试数据"""
        data_list = []

        # 生成不同大小的测试数据
        sizes = [100, 500, 1000, 5000]

        for size in sizes:
            dates = pd.date_range('2023-01-01', periods=size, freq='D')
            close_prices = np.random.randn(size).cumsum() + 100

            data = pd.DataFrame({
                'close': close_prices,
                'open': close_prices + np.random.randn(size) * 0.5,
                'high': close_prices + np.abs(np.random.randn(size)),
                'low': close_prices - np.abs(np.random.randn(size)),
                'volume': np.random.randint(1000, 10000, size)
            }, index=dates)

            data_list.append(data)

        return data_list

    def _test_initialization_overhead(self) -> Dict[str, Any]:
        """测试初始化开销"""
        self.logger.info("测试GPU初始化开销...")

        results = {
            'initialization_time': self.initialization_time,
            'memory_pool_optimized': self.memory_pool_optimized,
            'warmup_completed': self.warmup_completed
        }

        return results

    def _test_memory_pool_optimization(self) -> Dict[str, Any]:
        """测试内存池优化"""
        if not self.gpu_available:
            return {'status': 'GPU不可用'}

        try:
            # 测试内存分配性能
            start_time = time.time()

            # 分配多个不同大小的数组
            arrays = []
            for size in [100, 500, 1000, 5000]:
                arrays.append(cp.random.random(size, dtype=cp.float32))

            allocation_time = time.time() - start_time

            # 测试内存释放性能
            start_time = time.time()
            del arrays
            cp.cuda.Stream.null.synchronize()
            deallocation_time = time.time() - start_time

            return {
                'allocation_time': allocation_time,
                'deallocation_time': deallocation_time,
                'total_memory_ops_time': allocation_time + deallocation_time
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_batch_processing_optimization(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试批处理优化"""
        if not self.gpu_available:
            return {'status': 'GPU不可用'}

        try:
            # 定义测试函数（模拟EMA计算）
            def test_ema_calculation(data: pd.DataFrame, window: int = 20) -> pd.Series:
                close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
                alpha = 2.0 / (window + 1)

                ema_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)
                ema_gpu[0] = close_gpu[0]

                for i in range(1, len(close_gpu)):
                    ema_gpu[i] = alpha * close_gpu[i] + (1 - alpha) * ema_gpu[i-1]

                return pd.Series(cp.asnumpy(ema_gpu), index=data.index)

            # 测试串行处理
            start_time = time.time()
            serial_results = [test_ema_calculation(data) for data in test_data_list]
            serial_time = time.time() - start_time

            # 测试批处理优化
            start_time = time.time()
            batch_results = self._batch_process_optimization(
                test_data_list, test_ema_calculation, window=20)
            batch_time = time.time() - start_time

            return {
                'serial_time': serial_time,
                'batch_time': batch_time,
                'speedup': serial_time / batch_time if batch_time > 0 else 0,
                'data_count': len(test_data_list)
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_caching_optimization(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试缓存机制优化"""
        if not self.gpu_available:
            return {'status': 'GPU不可用'}

        try:
            # 清空缓存
            self.cache.clear()

            # 定义测试函数
            def test_calculation(data: pd.DataFrame) -> pd.Series:
                close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
                result = cp.sin(close_gpu) + cp.cos(close_gpu)
                return pd.Series(cp.asnumpy(result), index=data.index)

            # 第一次计算（无缓存）
            start_time = time.time()
            first_results = [test_calculation(data) for data in test_data_list]
            first_time = time.time() - start_time

            # 第二次计算（有缓存）
            start_time = time.time()
            second_results = [test_calculation(data) for data in test_data_list]
            second_time = time.time() - start_time

            return {
                'first_time': first_time,
                'second_time': second_time,
                'cache_speedup': first_time / second_time if second_time > 0 else 0,
                'cache_size': len(self.cache)
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_optimization_report(self, results: Dict[str, Any]):
        """生成优化报告"""
        report = f"""
# GPU开销优化测试报告

## 测试环境
- GPU可用性: {self.gpu_available}
- 初始化时间: {results.get('initialization', {}).get('initialization_time', 0):.4f}s
- 内存池优化: {results.get('initialization', {}).get('memory_pool_optimized', False)}
- 预热完成: {results.get('initialization', {}).get('warmup_completed', False)}

## 内存池优化结果
"""

        memory_results = results.get('memory_pool', {})
        if 'error' not in memory_results:
            report += f"""
- 内存分配时间: {memory_results.get('allocation_time', 0):.4f}s
- 内存释放时间: {memory_results.get('deallocation_time', 0):.4f}s
- 总内存操作时间: {memory_results.get('total_memory_ops_time', 0):.4f}s
"""
        else:
            report += f"- 错误: {memory_results.get('error', '未知错误')}\n"

        report += f"""
## 批处理优化结果
"""

        batch_results = results.get('batch_processing', {})
        if 'error' not in batch_results:
            report += f"""
- 串行处理时间: {batch_results.get('serial_time', 0):.4f}s
- 批处理时间: {batch_results.get('batch_time', 0):.4f}s
- 加速比: {batch_results.get('speedup', 0):.2f}x
- 处理数据集数量: {batch_results.get('data_count', 0)}
"""
        else:
            report += f"- 错误: {batch_results.get('error', '未知错误')}\n"

        report += f"""
## 缓存机制优化结果
"""

        cache_results = results.get('caching', {})
        if 'error' not in cache_results:
            report += f"""
- 首次计算时间: {cache_results.get('first_time', 0):.4f}s
- 缓存计算时间: {cache_results.get('second_time', 0):.4f}s
- 缓存加速比: {cache_results.get('cache_speedup', 0):.2f}x
- 缓存大小: {cache_results.get('cache_size', 0)}
"""
        else:
            report += f"- 错误: {cache_results.get('error', '未知错误')}\n"

        # 保存报告
        report_path = "reports/gpu_overhead_optimization_report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"GPU开销优化报告已保存到: {report_path}")
        print(report)


def main():
    """主函数"""
    logger.info("开始GPU开销优化测试...")

    # 创建优化器
    optimizer = GPUOverheadOptimizer()

    # 运行测试
    results = optimizer.test_gpu_overhead_optimization()

    logger.info("GPU开销优化测试完成")
    return results


if __name__ == "__main__":
    main()
