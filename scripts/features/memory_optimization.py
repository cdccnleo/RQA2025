#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化脚本 - 短期目标实现

专注于优化GPU内存分配和使用策略，包括：
1. 内存池管理：优化内存池配置和策略
2. 内存碎片整理：减少内存碎片，提高分配效率
3. 智能内存分配：根据数据大小智能分配内存
4. 内存监控：实时监控内存使用情况
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


class MemoryOptimizer:
    """内存优化器 - 短期目标实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'memory_limit_ratio': 0.8,
            'fragmentation_threshold': 0.3,
            'cleanup_interval': 100,
            'preallocation_sizes': [100, 500, 1000, 5000, 10000],
            'optimization_level': 'balanced'
        }
        self.logger = get_logger("memory_optimizer")
        self.gpu_available = GPU_AVAILABLE
        self.memory_operations = 0
        self.last_cleanup = 0

        if self.gpu_available:
            self._initialize_memory_optimization()

    def _initialize_memory_optimization(self):
        """初始化内存优化"""
        try:
            self.logger.info("初始化内存优化...")

            # 获取GPU信息
            gpu_info = self._get_gpu_info()
            total_memory = gpu_info.get('total_memory_gb', 0)

            # 根据GPU内存调整配置
            if total_memory >= 8:
                self.config['memory_limit_ratio'] = 0.9
                self.config['preallocation_sizes'] = [100, 500, 1000, 5000, 10000, 20000]
            elif total_memory >= 4:
                self.config['memory_limit_ratio'] = 0.8
                self.config['preallocation_sizes'] = [100, 500, 1000, 5000, 10000]
            else:
                self.config['memory_limit_ratio'] = 0.7
                self.config['preallocation_sizes'] = [100, 500, 1000, 5000]

            # 优化内存池设置
            self._optimize_memory_pool()

            # 预分配内存块
            self._preallocate_memory_blocks()

            self.logger.info(f"内存优化初始化完成，GPU内存: {total_memory:.1f}GB")

        except Exception as e:
            self.logger.warning(f"内存优化初始化失败: {e}")

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
                'used_memory_gb': (total_memory - free_memory) / 1024**3,
                'memory_usage_percent': (1 - free_memory / total_memory) * 100
            }
        except Exception as e:
            self.logger.warning(f"获取GPU信息失败: {e}")
            return {}

    def _optimize_memory_pool(self):
        """优化内存池设置"""
        try:
            # 获取默认内存池
            pool = cp.get_default_memory_pool()

            # 获取GPU信息
            gpu_info = self._get_gpu_info()
            total_memory = gpu_info.get('total_memory_gb', 0)

            # 设置内存限制
            if total_memory > 0:
                memory_limit = int(total_memory * 1024**3 * self.config['memory_limit_ratio'])
                pool.set_limit(size=memory_limit)
                self.logger.info(f"设置内存池限制: {memory_limit / 1024**3:.1f}GB")

            # 设置内存池策略
            # 允许更多内存使用，减少频繁分配/释放
            pool.set_limit(size=0)  # 不限制内存池大小

            self.logger.info("内存池优化完成")

        except Exception as e:
            self.logger.warning(f"内存池优化失败: {e}")

    def _preallocate_memory_blocks(self):
        """预分配内存块"""
        try:
            self.logger.info("开始预分配内存块...")

            for size in self.config['preallocation_sizes']:
                try:
                    # 预分配内存块并立即释放，但保留在内存池中
                    temp_array = cp.zeros(size, dtype=cp.float32)
                    del temp_array

                    # 预分配不同数据类型的数组
                    temp_array = cp.zeros(size, dtype=cp.float64)
                    del temp_array

                except Exception as e:
                    self.logger.warning(f"预分配大小{size}失败: {e}")
                    continue

            self.logger.info("内存块预分配完成")

        except Exception as e:
            self.logger.warning(f"内存块预分配失败: {e}")

    def _check_memory_fragmentation(self) -> float:
        """检查内存碎片化程度"""
        try:
            if not self.gpu_available:
                return 0.0

            # 获取内存池信息
            pool = cp.get_default_memory_pool()

            # 尝试分配不同大小的内存块来检测碎片
            fragmentation_scores = []

            for size in [100, 500, 1000, 5000]:
                try:
                    start_time = time.time()
                    temp_array = cp.zeros(size, dtype=cp.float32)
                    allocation_time = time.time() - start_time
                    del temp_array

                    # 分配时间越长，碎片化越严重
                    if allocation_time > 0.001:  # 1ms阈值
                        fragmentation_scores.append(1.0)
                    else:
                        fragmentation_scores.append(0.0)

                except Exception:
                    fragmentation_scores.append(1.0)

            # 计算平均碎片化分数
            fragmentation_ratio = np.mean(fragmentation_scores)

            return fragmentation_ratio

        except Exception as e:
            self.logger.warning(f"内存碎片检查失败: {e}")
            return 0.0

    def _cleanup_memory(self):
        """清理内存"""
        try:
            if not self.gpu_available:
                return

            # 获取内存池
            pool = cp.get_default_memory_pool()

            # 释放所有未使用的内存块
            pool.free_all_blocks()

            # 同步GPU
            cp.cuda.Stream.null.synchronize()

            self.logger.info("内存清理完成")

        except Exception as e:
            self.logger.warning(f"内存清理失败: {e}")

    def _smart_memory_allocation(self, size: int, dtype: np.dtype = np.float32) -> Optional[cp.ndarray]:
        """智能内存分配"""
        try:
            if not self.gpu_available:
                return None

            # 检查内存使用情况
            gpu_info = self._get_gpu_info()
            memory_usage = gpu_info.get('memory_usage_percent', 0)

            # 如果内存使用率过高，先清理
            if memory_usage > 90:
                self._cleanup_memory()

            # 检查碎片化
            fragmentation = self._check_memory_fragmentation()
            if fragmentation > self.config['fragmentation_threshold']:
                self.logger.info(f"检测到内存碎片化({fragmentation:.2f})，执行清理")
                self._cleanup_memory()

            # 智能分配内存
            start_time = time.time()
            array = cp.zeros(size, dtype=dtype)
            allocation_time = time.time() - start_time

            # 记录内存操作
            self.memory_operations += 1

            # 定期清理
            if self.memory_operations - self.last_cleanup > self.config['cleanup_interval']:
                self._cleanup_memory()
                self.last_cleanup = self.memory_operations

            if allocation_time > 0.01:  # 10ms阈值
                self.logger.warning(f"内存分配时间过长: {allocation_time:.4f}s")

            return array

        except Exception as e:
            self.logger.warning(f"智能内存分配失败: {e}")
            return None

    def _optimize_data_transfer(self, data: pd.DataFrame) -> cp.ndarray:
        """优化数据传输"""
        try:
            if not self.gpu_available:
                return None

            # 选择最优的数据类型
            if data.dtypes['close'].kind in 'fc':  # float类型
                dtype = cp.float32
            else:
                dtype = cp.float64

            # 智能内存分配
            close_values = data['close'].values
            gpu_array = self._smart_memory_allocation(len(close_values), dtype)

            if gpu_array is None:
                # 回退到标准分配
                gpu_array = cp.asarray(close_values, dtype=dtype)
            else:
                # 使用预分配的数组
                gpu_array[:] = close_values

            return gpu_array

        except Exception as e:
            self.logger.warning(f"数据传输优化失败: {e}")
            return cp.asarray(data['close'].values, dtype=cp.float32)

    def test_memory_optimization(self):
        """测试内存优化效果"""
        self.logger.info("开始内存优化测试...")

        # 生成测试数据
        test_data_list = self._generate_test_data()

        # 测试不同的内存优化策略
        results = {}

        # 1. 测试内存池优化
        results['memory_pool'] = self._test_memory_pool_optimization()

        # 2. 测试智能内存分配
        results['smart_allocation'] = self._test_smart_memory_allocation(test_data_list)

        # 3. 测试数据传输优化
        results['data_transfer'] = self._test_data_transfer_optimization(test_data_list)

        # 4. 测试内存碎片管理
        results['fragmentation'] = self._test_fragmentation_management(test_data_list)

        # 生成报告
        self._generate_memory_optimization_report(results)

        return results

    def _generate_test_data(self) -> List[pd.DataFrame]:
        """生成测试数据"""
        data_list = []

        # 生成不同大小的测试数据
        sizes = [100, 500, 1000, 5000, 10000]

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

    def _test_memory_pool_optimization(self) -> Dict[str, Any]:
        """测试内存池优化"""
        self.logger.info("测试内存池优化...")

        try:
            if not self.gpu_available:
                return {'status': 'GPU不可用'}

            # 获取初始内存信息
            initial_memory = self._get_gpu_info()

            # 测试内存分配性能
            allocation_times = []
            for size in [100, 500, 1000, 5000]:
                start_time = time.time()
                array = cp.zeros(size, dtype=cp.float32)
                allocation_time = time.time() - start_time
                allocation_times.append(allocation_time)
                del array

            # 测试内存释放性能
            arrays = []
            for size in [100, 500, 1000, 5000]:
                arrays.append(cp.zeros(size, dtype=cp.float32))

            start_time = time.time()
            del arrays
            cp.cuda.Stream.null.synchronize()
            deallocation_time = time.time() - start_time

            # 获取最终内存信息
            final_memory = self._get_gpu_info()

            return {
                'initial_memory_gb': initial_memory.get('free_memory_gb', 0),
                'final_memory_gb': final_memory.get('free_memory_gb', 0),
                'allocation_times': allocation_times,
                'deallocation_time': deallocation_time,
                'avg_allocation_time': np.mean(allocation_times)
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_smart_memory_allocation(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试智能内存分配"""
        self.logger.info("测试智能内存分配...")

        try:
            if not self.gpu_available:
                return {'status': 'GPU不可用'}

            # 测试标准分配
            standard_times = []
            for data in test_data_list:
                start_time = time.time()
                array = cp.asarray(data['close'].values, dtype=cp.float32)
                standard_times.append(time.time() - start_time)
                del array

            # 测试智能分配
            smart_times = []
            for data in test_data_list:
                start_time = time.time()
                array = self._smart_memory_allocation(len(data), cp.float32)
                smart_times.append(time.time() - start_time)
                if array is not None:
                    del array

            return {
                'standard_times': standard_times,
                'smart_times': smart_times,
                'avg_standard_time': np.mean(standard_times),
                'avg_smart_time': np.mean(smart_times),
                'speedup': np.mean(standard_times) / np.mean(smart_times) if np.mean(smart_times) > 0 else 0
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_data_transfer_optimization(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试数据传输优化"""
        self.logger.info("测试数据传输优化...")

        try:
            if not self.gpu_available:
                return {'status': 'GPU不可用'}

            # 测试标准传输
            standard_times = []
            for data in test_data_list:
                start_time = time.time()
                array = cp.asarray(data['close'].values, dtype=cp.float32)
                standard_times.append(time.time() - start_time)
                del array

            # 测试优化传输
            optimized_times = []
            for data in test_data_list:
                start_time = time.time()
                array = self._optimize_data_transfer(data)
                optimized_times.append(time.time() - start_time)
                if array is not None:
                    del array

            return {
                'standard_times': standard_times,
                'optimized_times': optimized_times,
                'avg_standard_time': np.mean(standard_times),
                'avg_optimized_time': np.mean(optimized_times),
                'speedup': np.mean(standard_times) / np.mean(optimized_times) if np.mean(optimized_times) > 0 else 0
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_fragmentation_management(self, test_data_list: List[pd.DataFrame]) -> Dict[str, Any]:
        """测试内存碎片管理"""
        self.logger.info("测试内存碎片管理...")

        try:
            if not self.gpu_available:
                return {'status': 'GPU不可用'}

            # 获取初始碎片化程度
            initial_fragmentation = self._check_memory_fragmentation()

            # 执行多次内存分配/释放操作来模拟碎片
            for _ in range(10):
                arrays = []
                for data in test_data_list:
                    arrays.append(cp.asarray(data['close'].values, dtype=cp.float32))
                del arrays

            # 获取碎片化后的程度
            fragmented_fragmentation = self._check_memory_fragmentation()

            # 执行清理
            self._cleanup_memory()

            # 获取清理后的碎片化程度
            cleaned_fragmentation = self._check_memory_fragmentation()

            return {
                'initial_fragmentation': initial_fragmentation,
                'fragmented_fragmentation': fragmented_fragmentation,
                'cleaned_fragmentation': cleaned_fragmentation,
                'fragmentation_reduction': fragmented_fragmentation - cleaned_fragmentation
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_memory_optimization_report(self, results: Dict[str, Any]):
        """生成内存优化报告"""
        report = f"""
# 内存优化测试报告

## 测试环境
- GPU可用性: {self.gpu_available}
- 内存限制比例: {self.config['memory_limit_ratio']}
- 碎片化阈值: {self.config['fragmentation_threshold']}
- 清理间隔: {self.config['cleanup_interval']}

## 内存池优化结果
"""

        memory_pool_results = results.get('memory_pool', {})
        if 'error' not in memory_pool_results:
            report += f"""
- 初始内存: {memory_pool_results.get('initial_memory_gb', 0):.2f}GB
- 最终内存: {memory_pool_results.get('final_memory_gb', 0):.2f}GB
- 平均分配时间: {memory_pool_results.get('avg_allocation_time', 0):.4f}s
- 释放时间: {memory_pool_results.get('deallocation_time', 0):.4f}s
"""
        else:
            report += f"- 错误: {memory_pool_results.get('error', '未知错误')}\n"

        report += f"""
## 智能内存分配结果
"""

        smart_allocation_results = results.get('smart_allocation', {})
        if 'error' not in smart_allocation_results:
            report += f"""
- 标准分配平均时间: {smart_allocation_results.get('avg_standard_time', 0):.4f}s
- 智能分配平均时间: {smart_allocation_results.get('avg_smart_time', 0):.4f}s
- 加速比: {smart_allocation_results.get('speedup', 0):.2f}x
"""
        else:
            report += f"- 错误: {smart_allocation_results.get('error', '未知错误')}\n"

        report += f"""
## 数据传输优化结果
"""

        data_transfer_results = results.get('data_transfer', {})
        if 'error' not in data_transfer_results:
            report += f"""
- 标准传输平均时间: {data_transfer_results.get('avg_standard_time', 0):.4f}s
- 优化传输平均时间: {data_transfer_results.get('avg_optimized_time', 0):.4f}s
- 加速比: {data_transfer_results.get('speedup', 0):.2f}x
"""
        else:
            report += f"- 错误: {data_transfer_results.get('error', '未知错误')}\n"

        report += f"""
## 内存碎片管理结果
"""

        fragmentation_results = results.get('fragmentation', {})
        if 'error' not in fragmentation_results:
            report += f"""
- 初始碎片化: {fragmentation_results.get('initial_fragmentation', 0):.2f}
- 碎片化后: {fragmentation_results.get('fragmented_fragmentation', 0):.2f}
- 清理后: {fragmentation_results.get('cleaned_fragmentation', 0):.2f}
- 碎片减少: {fragmentation_results.get('fragmentation_reduction', 0):.2f}
"""
        else:
            report += f"- 错误: {fragmentation_results.get('error', '未知错误')}\n"

        # 保存报告
        report_path = "reports/memory_optimization_report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"内存优化报告已保存到: {report_path}")
        print(report)


def main():
    """主函数"""
    logger.info("开始内存优化测试...")

    # 创建优化器
    optimizer = MemoryOptimizer()

    # 运行测试
    results = optimizer.test_memory_optimization()

    logger.info("内存优化测试完成")
    return results


if __name__ == "__main__":
    main()
