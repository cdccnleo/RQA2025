#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理优化脚本 - 短期目标实现

专注于实现批量计算，提高GPU利用率，包括：
1. 数据批处理：将多个数据集合并处理
2. 并行计算：利用GPU并行能力
3. 内存优化：优化批处理内存使用
4. 动态批大小：根据数据大小动态调整批大小
"""

from src.utils.logger import get_logger
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
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


class BatchProcessingOptimizer:
    """批处理优化器 - 短期目标实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'batch_size': 1000,
            'max_batch_size': 10000,
            'min_batch_size': 100,
            'memory_limit': 0.8,
            'parallel_workers': 4,
            'optimization_level': 'balanced'
        }
        self.logger = get_logger("batch_processing_optimizer")
        self.gpu_available = GPU_AVAILABLE

        if self.gpu_available:
            self._initialize_batch_processing()

    def _initialize_batch_processing(self):
        """初始化批处理优化"""
        try:
            self.logger.info("初始化批处理优化...")

            # 获取GPU信息
            if self.gpu_available:
                gpu_info = self._get_gpu_info()
                total_memory = gpu_info.get('total_memory_gb', 0)

                # 根据GPU内存动态调整批大小
                if total_memory >= 8:
                    self.config['batch_size'] = 5000
                    self.config['max_batch_size'] = 20000
                elif total_memory >= 4:
                    self.config['batch_size'] = 2000
                    self.config['max_batch_size'] = 10000
                else:
                    self.config['batch_size'] = 1000
                    self.config['max_batch_size'] = 5000

                self.logger.info(f"GPU内存: {total_memory:.1f}GB，批大小: {self.config['batch_size']}")

        except Exception as e:
            self.logger.warning(f"批处理初始化失败: {e}")

    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        try:
            if not self.gpu_available:
                return {}

            device = cp.cuda.Device()
            memory_info = cp.cuda.runtime.memGetInfo()
            total_memory = memory_info[1]
            free_memory = memory_info[0]

            return {
                'device': str(device),
                'total_memory_gb': total_memory / 1024**3,
                'free_memory_gb': free_memory / 1024**3,
                'memory_usage_percent': (1 - free_memory / total_memory) * 100
            }
        except Exception as e:
            self.logger.warning(f"获取GPU信息失败: {e}")
            return {}

    def _calculate_optimal_batch_size(self, data_size: int, memory_usage: float = 0) -> int:
        """计算最优批大小"""
        if not self.gpu_available:
            return min(data_size, self.config['batch_size'])

        # 根据数据大小和内存使用情况动态调整
        base_batch_size = self.config['batch_size']

        # 如果内存使用率超过80%，减少批大小
        if memory_usage > 80:
            base_batch_size = int(base_batch_size * 0.5)

        # 根据数据大小调整
        if data_size < 1000:
            optimal_size = min(data_size, base_batch_size // 2)
        elif data_size < 10000:
            optimal_size = base_batch_size
        else:
            optimal_size = min(self.config['max_batch_size'], data_size)

        return max(self.config['min_batch_size'], optimal_size)

    def _batch_process_data(self, data_list: List[pd.DataFrame],
                            indicator_func, **kwargs) -> List[Any]:
        """批处理数据"""
        if not data_list:
            return []

        if not self.gpu_available:
            # CPU模式：串行处理
            return [indicator_func(data, **kwargs) for data in data_list]

        try:
            # GPU模式：批处理优化
            results = []
            total_data_size = sum(len(data) for data in data_list)

            # 获取GPU内存使用情况
            gpu_info = self._get_gpu_info()
            memory_usage = gpu_info.get('memory_usage_percent', 0)

            # 计算最优批大小
            optimal_batch_size = self._calculate_optimal_batch_size(total_data_size, memory_usage)

            self.logger.info(f"批处理优化: 总数据量={total_data_size}, 批大小={optimal_batch_size}")

            # 分批处理
            current_batch = []
            current_batch_size = 0

            for i, data in enumerate(data_list):
                data_size = len(data)

                # 检查是否需要开始新的批次
                if current_batch_size + data_size > optimal_batch_size and current_batch:
                    # 处理当前批次
                    batch_results = self._process_batch(current_batch, indicator_func, **kwargs)
                    results.extend(batch_results)

                    # 重置批次
                    current_batch = []
                    current_batch_size = 0

                # 添加到当前批次
                current_batch.append(data)
                current_batch_size += data_size

            # 处理最后一个批次
            if current_batch:
                batch_results = self._process_batch(current_batch, indicator_func, **kwargs)
                results.extend(batch_results)

            return results

        except Exception as e:
            self.logger.warning(f"批处理失败: {e}，回退到串行处理")
            return [indicator_func(data, **kwargs) for data in data_list]

    def _process_batch(self, batch_data: List[pd.DataFrame],
                       indicator_func, **kwargs) -> List[Any]:
        """处理单个批次"""
        try:
            # 合并批次数据
            combined_data = self._combine_batch_data(batch_data)

            # 在GPU上处理合并数据
            combined_result = indicator_func(combined_data, **kwargs)

            # 分割结果
            return self._split_batch_results(combined_result, batch_data)

        except Exception as e:
            self.logger.warning(f"批次处理失败: {e}，回退到单独处理")
            return [indicator_func(data, **kwargs) for data in batch_data]

    def _combine_batch_data(self, batch_data: List[pd.DataFrame]) -> pd.DataFrame:
        """合并批次数据"""
        try:
            # 创建合并的DataFrame
            combined_data = pd.concat(batch_data, ignore_index=True)

            # 添加批次标识
            batch_ids = []
            for i, data in enumerate(batch_data):
                batch_ids.extend([i] * len(data))

            combined_data['batch_id'] = batch_ids

            return combined_data

        except Exception as e:
            self.logger.warning(f"数据合并失败: {e}")
            return batch_data[0] if batch_data else pd.DataFrame()

    def _split_batch_results(self, combined_result: pd.DataFrame,
                             batch_data: List[pd.DataFrame]) -> List[Any]:
        """分割批次结果"""
        try:
            results = []
            start_idx = 0

            for data in batch_data:
                end_idx = start_idx + len(data)

                # 提取对应批次的结果
                if isinstance(combined_result, pd.DataFrame):
                    batch_result = combined_result.iloc[start_idx:end_idx].copy()
                    batch_result = batch_result.drop('batch_id', axis=1, errors='ignore')
                else:
                    batch_result = combined_result.iloc[start_idx:end_idx]

                results.append(batch_result)
                start_idx = end_idx

            return results

        except Exception as e:
            self.logger.warning(f"结果分割失败: {e}")
            return [combined_result] * len(batch_data)

    def _parallel_batch_processing(self, data_list: List[pd.DataFrame],
                                   indicator_func, **kwargs) -> List[Any]:
        """并行批处理"""
        if not self.gpu_available:
            return self._batch_process_data(data_list, indicator_func, **kwargs)

        try:
            # 将数据分成多个并行批次
            num_workers = min(self.config['parallel_workers'], len(data_list))
            batch_size = len(data_list) // num_workers

            if batch_size == 0:
                return self._batch_process_data(data_list, indicator_func, **kwargs)

            # 创建并行批次
            parallel_batches = []
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                parallel_batches.append(batch)

            # 并行处理（这里简化为串行，实际可以使用多进程）
            results = []
            for batch in parallel_batches:
                batch_results = self._batch_process_data(batch, indicator_func, **kwargs)
                results.extend(batch_results)

            return results

        except Exception as e:
            self.logger.warning(f"并行批处理失败: {e}，回退到串行批处理")
            return self._batch_process_data(data_list, indicator_func, **kwargs)

    def test_batch_processing_optimization(self):
        """测试批处理优化效果"""
        self.logger.info("开始批处理优化测试...")

        # 生成测试数据
        test_data_list = self._generate_test_data()

        # 测试不同的批处理策略
        results = {}

        # 1. 测试基本批处理
        results['basic_batch'] = self._test_basic_batch_processing(test_data_list)

        # 2. 测试并行批处理
        results['parallel_batch'] = self._test_parallel_batch_processing(test_data_list)

        # 3. 测试动态批大小
        results['dynamic_batch'] = self._test_dynamic_batch_sizing(test_data_list)

        # 4. 测试内存优化
        results['memory_optimization'] = self._test_memory_optimization(test_data_list)

        # 生成报告
        self._generate_batch_optimization_report(results)

        return results

    def _generate_test_data(self) -> List[pd.DataFrame]:
        """生成测试数据"""
        data_list = []

        # 生成不同大小的测试数据
        sizes = [100, 500, 1000, 2000, 5000, 10000]

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

    def _test_basic_batch_processing(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试基本批处理"""
        self.logger.info("测试基本批处理...")

        try:
            # 定义测试函数（模拟EMA计算）
            def test_ema_calculation(data: pd.DataFrame, window: int = 20) -> pd.Series:
                if not self.gpu_available:
                    # CPU版本
                    alpha = 2.0 / (window + 1)
                    ema = pd.Series(index=data.index, dtype=float)
                    ema.iloc[0] = data['close'].iloc[0]

                    for i in range(1, len(data)):
                        ema.iloc[i] = alpha * data['close'].iloc[i] + (1 - alpha) * ema.iloc[i-1]

                    return ema
                else:
                    # GPU版本
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

            # 测试批处理
            start_time = time.time()
            batch_results = self._batch_process_data(
                test_data_list, test_ema_calculation, window=20)
            batch_time = time.time() - start_time

            return {
                'serial_time': serial_time,
                'batch_time': batch_time,
                'speedup': serial_time / batch_time if batch_time > 0 else 0,
                'data_count': len(test_data_list),
                'total_data_size': sum(len(data) for data in test_data_list)
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_parallel_batch_processing(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试并行批处理"""
        self.logger.info("测试并行批处理...")

        try:
            # 定义测试函数
            def test_calculation(data: pd.DataFrame) -> pd.Series:
                if not self.gpu_available:
                    return data['close'].rolling(window=20).mean()
                else:
                    close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
                    weights = cp.ones(20, dtype=cp.float32) / 20
                    result = cp.convolve(close_gpu, weights, mode='valid')
                    padding = cp.full(19, cp.nan, dtype=cp.float32)
                    result = cp.concatenate([padding, result])
                    return pd.Series(cp.asnumpy(result), index=data.index)

            # 测试串行批处理
            start_time = time.time()
            serial_batch_results = self._batch_process_data(test_data_list, test_calculation)
            serial_batch_time = time.time() - start_time

            # 测试并行批处理
            start_time = time.time()
            parallel_batch_results = self._parallel_batch_processing(
                test_data_list, test_calculation)
            parallel_batch_time = time.time() - start_time

            return {
                'serial_batch_time': serial_batch_time,
                'parallel_batch_time': parallel_batch_time,
                'speedup': serial_batch_time / parallel_batch_time if parallel_batch_time > 0 else 0,
                'data_count': len(test_data_list)
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_dynamic_batch_sizing(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试动态批大小"""
        self.logger.info("测试动态批大小...")

        try:
            # 测试不同批大小的性能
            batch_sizes = [100, 500, 1000, 2000, 5000]
            results = {}

            for batch_size in batch_sizes:
                # 临时设置批大小
                original_batch_size = self.config['batch_size']
                self.config['batch_size'] = batch_size

                # 定义测试函数
                def test_calculation(data: pd.DataFrame) -> pd.Series:
                    if not self.gpu_available:
                        return data['close'].rolling(window=10).std()
                    else:
                        close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
                        result = cp.std(close_gpu)
                        return pd.Series([result] * len(data), index=data.index)

                # 测试批处理性能
                start_time = time.time()
                batch_results = self._batch_process_data(test_data_list, test_calculation)
                process_time = time.time() - start_time

                results[batch_size] = {
                    'process_time': process_time,
                    'batch_size': batch_size
                }

                # 恢复原始批大小
                self.config['batch_size'] = original_batch_size

            # 找到最优批大小
            optimal_batch_size = min(results.keys(), key=lambda x: results[x]['process_time'])

            return {
                'results': results,
                'optimal_batch_size': optimal_batch_size,
                'optimal_time': results[optimal_batch_size]['process_time']
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_memory_optimization(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试内存优化"""
        self.logger.info("测试内存优化...")

        try:
            if not self.gpu_available:
                return {'status': 'GPU不可用'}

            # 获取初始内存信息
            initial_memory = self._get_gpu_info()

            # 执行内存密集型操作
            def memory_intensive_calculation(data: pd.DataFrame) -> pd.Series:
                close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)

                # 创建多个大型数组
                arrays = []
                for i in range(5):
                    arrays.append(cp.random.random_like(close_gpu))
                    arrays.append(cp.sin(close_gpu) + cp.cos(close_gpu))

                # 执行复杂计算
                result = cp.zeros_like(close_gpu)
                for arr in arrays:
                    result += arr

                return pd.Series(cp.asnumpy(result), index=data.index)

            # 测试批处理内存使用
            start_time = time.time()
            batch_results = self._batch_process_data(test_data_list, memory_intensive_calculation)
            process_time = time.time() - start_time

            # 获取最终内存信息
            final_memory = self._get_gpu_info()

            # 清理GPU内存
            cp.get_default_memory_pool().free_all_blocks()

            return {
                'process_time': process_time,
                'initial_memory_gb': initial_memory.get('free_memory_gb', 0),
                'final_memory_gb': final_memory.get('free_memory_gb', 0),
                'memory_used_gb': initial_memory.get('free_memory_gb', 0) - final_memory.get('free_memory_gb', 0),
                'data_count': len(test_data_list)
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_batch_optimization_report(self, results: Dict[str, Any]):
        """生成批处理优化报告"""
        report = f"""
# 批处理优化测试报告

## 测试环境
- GPU可用性: {self.gpu_available}
- 默认批大小: {self.config['batch_size']}
- 最大批大小: {self.config['max_batch_size']}
- 并行工作数: {self.config['parallel_workers']}

## 基本批处理结果
"""

        basic_results = results.get('basic_batch', {})
        if 'error' not in basic_results:
            report += f"""
- 串行处理时间: {basic_results.get('serial_time', 0):.4f}s
- 批处理时间: {basic_results.get('batch_time', 0):.4f}s
- 加速比: {basic_results.get('speedup', 0):.2f}x
- 数据集数量: {basic_results.get('data_count', 0)}
- 总数据量: {basic_results.get('total_data_size', 0)}
"""
        else:
            report += f"- 错误: {basic_results.get('error', '未知错误')}\n"

        report += f"""
## 并行批处理结果
"""

        parallel_results = results.get('parallel_batch', {})
        if 'error' not in parallel_results:
            report += f"""
- 串行批处理时间: {parallel_results.get('serial_batch_time', 0):.4f}s
- 并行批处理时间: {parallel_results.get('parallel_batch_time', 0):.4f}s
- 加速比: {parallel_results.get('speedup', 0):.2f}x
- 数据集数量: {parallel_results.get('data_count', 0)}
"""
        else:
            report += f"- 错误: {parallel_results.get('error', '未知错误')}\n"

        report += f"""
## 动态批大小结果
"""

        dynamic_results = results.get('dynamic_batch', {})
        if 'error' not in dynamic_results:
            optimal_size = dynamic_results.get('optimal_batch_size', 0)
            optimal_time = dynamic_results.get('optimal_time', 0)
            report += f"""
- 最优批大小: {optimal_size}
- 最优处理时间: {optimal_time:.4f}s
"""
        else:
            report += f"- 错误: {dynamic_results.get('error', '未知错误')}\n"

        report += f"""
## 内存优化结果
"""

        memory_results = results.get('memory_optimization', {})
        if 'error' not in memory_results:
            report += f"""
- 处理时间: {memory_results.get('process_time', 0):.4f}s
- 初始内存: {memory_results.get('initial_memory_gb', 0):.2f}GB
- 最终内存: {memory_results.get('final_memory_gb', 0):.2f}GB
- 内存使用: {memory_results.get('memory_used_gb', 0):.2f}GB
- 数据集数量: {memory_results.get('data_count', 0)}
"""
        else:
            report += f"- 错误: {memory_results.get('error', '未知错误')}\n"

        # 保存报告
        report_path = "reports/batch_processing_optimization_report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"批处理优化报告已保存到: {report_path}")
        print(report)


def main():
    """主函数"""
    logger.info("开始批处理优化测试...")

    # 创建优化器
    optimizer = BatchProcessingOptimizer()

    # 运行测试
    results = optimizer.test_batch_processing_optimization()

    logger.info("批处理优化测试完成")
    return results


if __name__ == "__main__":
    main()
