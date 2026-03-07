from .gpu_technical_processor import GPUTechnicalProcessor
from src.infrastructure.utils.logging.logger import get_logger
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多GPU并行技术指标处理器

实现多GPU设备检测、数据分发、结果聚合和负载均衡
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

try:
    pass
    # 使用依赖管理器安全导入torch
    from ...dependency_manager import dependency_manager
    torch = dependency_manager.safe_import('torch')
    MULTI_GPU_AVAILABLE = True
    logger.info("多GPU支持可用")
except ImportError:
    MULTI_GPU_AVAILABLE = False
    logger.warning("多GPU支持不可用")
    # 创建mock torch模块
    from unittest.mock import MagicMock
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.cuda.device_count.return_value = 0


class MultiGPUProcessor:

    """多GPU并行技术指标处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):

        default_config = {
            'use_multi_gpu': True,
            'max_gpus': 4,  # 最大使用GPU数量
            'load_balancing': 'round_robin',  # round_robin, memory_based, performance_based
            'chunk_size': 50000,  # 每个GPU处理的数据块大小
            'memory_threshold': 0.8,  # GPU内存使用阈值
            'fallback_to_single_gpu': True,
            'fallback_to_cpu': True,
            'sync_mode': True,  # 同步模式，确保结果一致性
            'warmup_iterations': 3  # GPU预热迭代次数
        }

        if config:

            default_config.update(config)
        self.config = default_config
        self.logger = logger or get_logger("multi_gpu_processor")
        self.available_gpus = []
        self.gpu_processors = {}
        self.gpu_info = {}
        self.lock = threading.Lock()

        if MULTI_GPU_AVAILABLE:
            self._initialize_multi_gpu()
        else:
            self.logger.warn("多GPU不可用，将使用单GPU或CPU模式")
            self._initialize_fallback()

    def _initialize_multi_gpu(self):
        """初始化多GPU环境"""
        try:
            # 检测可用GPU
            self._detect_available_gpus()

            if not self.available_gpus:
                self.logger.warn("未检测到可用GPU，回退到CPU模式")
                self._initialize_fallback()
                return

            # 为每个GPU创建处理器
            self._create_gpu_processors()

            # GPU预热
            self._warmup_gpus()

            self.logger.info(f"多GPU初始化完成，可用GPU数量: {len(self.available_gpus)}")

        except Exception as e:
            self.logger.error(f"多GPU初始化失败: {e}")
            self._initialize_fallback()

    def _detect_available_gpus(self):
        """检测可用GPU设备"""
        try:
            # 使用PyTorch检测GPU
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"检测到 {gpu_count} 个GPU设备")

                for i in range(min(gpu_count, self.config['max_gpus'])):
                    try:
                        # 获取GPU信息
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(
                            i).total_memory / 1024 ** 3  # GB

                        # 检查GPU是否可用
                        torch.cuda.set_device(i)
                        test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                        del test_tensor
                        torch.cuda.empty_cache()

                        self.available_gpus.append(i)
                        self.gpu_info[i] = {
                            'name': gpu_name,
                            'memory_gb': gpu_memory,
                            'device_id': i,
                            'status': 'available'
                        }

                        self.logger.info(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f}GB")

                    except Exception as e:
                        self.logger.warn(f"GPU {i} 不可用: {e}")
                        continue
            else:
                self.logger.warn("CUDA不可用")

        except Exception as e:
            self.logger.error(f"GPU检测失败: {e}")

    def _create_gpu_processors(self):
        """为每个GPU创建处理器"""
        for gpu_id in self.available_gpus:
            try:
                # 为每个GPU创建独立的配置
                gpu_config = self.config.copy()
                gpu_config['device_id'] = gpu_id
                gpu_config['memory_limit'] = self.config['memory_threshold']

                # 创建GPU处理器
                processor = GPUTechnicalProcessor(config=gpu_config)
                self.gpu_processors[gpu_id] = processor

                self.logger.info(f"GPU {gpu_id} 处理器创建成功")

            except Exception as e:
                self.logger.error(f"GPU {gpu_id} 处理器创建失败: {e}")
                self.available_gpus.remove(gpu_id)

    def _warmup_gpus(self):
        """GPU预热"""
        self.logger.info("开始GPU预热...")

        for gpu_id in self.available_gpus:
            try:
                processor = self.gpu_processors[gpu_id]

                # 创建测试数据
                test_data = pd.DataFrame({
                    'close': np.random.randn(1000) * 100 + 1000,
                    'high': np.random.randn(1000) * 10 + 1010,
                    'low': np.random.randn(1000) * 10 + 990,
                    'volume': np.random.randint(1000000, 10000000, 1000)
                })

                # 预热迭代
                for _ in range(self.config['warmup_iterations']):
                    processor.calculate_multiple_indicators_gpu(
                        test_data,
                        ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands']
                    )

                self.logger.info(f"GPU {gpu_id} 预热完成")

            except Exception as e:
                self.logger.warn(f"GPU {gpu_id} 预热失败: {e}")

    def _initialize_fallback(self):
        """初始化回退模式"""
        if self.config['fallback_to_single_gpu']:
            try:
                # 尝试创建单GPU处理器
                self.single_gpu_processor = GPUTechnicalProcessor()
                self.logger.info("回退到单GPU模式")
            except Exception as e:
                self.logger.warn(f"单GPU初始化失败: {e}")
                self.single_gpu_processor = None

        if not self.single_gpu_processor and self.config['fallback_to_cpu']:
            self.logger.info("回退到CPU模式")
            # CPU模式将在具体计算时实现

    def _split_data_for_gpus(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """将数据分割给多个GPU"""
        total_rows = len(data)
        gpu_count = len(self.available_gpus)

        if gpu_count == 0:
            return {}

        # 计算每个GPU的数据块大小
        chunk_size = max(self.config['chunk_size'], total_rows // gpu_count)

        data_chunks = {}
        for i, gpu_id in enumerate(self.available_gpus):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)

            if start_idx < total_rows:
                chunk = data.iloc[start_idx:end_idx].copy()
                data_chunks[gpu_id] = chunk

        return data_chunks

    def _load_balance_data(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """基于负载均衡策略分发数据"""
        if self.config['load_balancing'] == 'memory_based':
            return self._memory_based_distribution(data)
        elif self.config['load_balancing'] == 'performance_based':
            return self._performance_based_distribution(data)
        else:  # round_robin
            return self._split_data_for_gpus(data)

    def _memory_based_distribution(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """基于内存使用情况的数据分发"""
        # 获取每个GPU的内存使用情况
        gpu_memory_usage = {}
        for gpu_id in self.available_gpus:
            try:
                torch.cuda.set_device(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3  # GB
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024 ** 3  # GB
                gpu_memory_usage[gpu_id] = memory_allocated + memory_reserved
            except Exception as e:
                self.logger.warn(f"获取GPU {gpu_id} 内存信息失败: {e}")
                gpu_memory_usage[gpu_id] = 0

        # 按内存使用率排序，优先分配给内存使用较少的GPU
        sorted_gpus = sorted(gpu_memory_usage.items(), key=lambda x: x[1])
        available_gpus = [gpu_id for gpu_id, _ in sorted_gpus]

        return self._split_data_for_gpus(data, available_gpus)

    def _performance_based_distribution(self, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """基于性能的数据分发"""
        # 这里可以实现基于历史性能数据的分发策略
        # 暂时使用轮询分发
        return self._split_data_for_gpus(data)

    def _split_data_for_gpus(self, data: pd.DataFrame, gpu_list: List[int] = None) -> Dict[int, pd.DataFrame]:
        """将数据分割给指定的GPU列表"""
        if gpu_list is None:
            gpu_list = self.available_gpus

        total_rows = len(data)
        gpu_count = len(gpu_list)

        if gpu_count == 0:
            return {}

        chunk_size = max(self.config['chunk_size'], total_rows // gpu_count)
        data_chunks = {}

        for i, gpu_id in enumerate(gpu_list):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)

            if start_idx < total_rows:
                chunk = data.iloc[start_idx:end_idx].copy()
                data_chunks[gpu_id] = chunk

        return data_chunks

    def _process_chunk_on_gpu(self, gpu_id: int, chunk: pd.DataFrame,


                              indicators: List[str], params: Dict[str, Any] = None) -> pd.DataFrame:
        """在指定GPU上处理数据块"""
        try:
            processor = self.gpu_processors[gpu_id]

            # 设置GPU设备
            torch.cuda.set_device(gpu_id)

            # 处理数据
            result = processor.calculate_multiple_indicators_gpu(chunk, indicators, params)

            return result

        except Exception as e:
            self.logger.error(f"GPU {gpu_id} 处理失败: {e}")
            # 返回空DataFrame，后续会处理
            return pd.DataFrame()

    def calculate_multiple_indicators_multi_gpu(self, data: pd.DataFrame,


                                                indicators: List[str],
                                                params: Dict[str, Any] = None) -> pd.DataFrame:
        """多GPU并行计算多个技术指标"""
        if not self.available_gpus:
            # 回退到单GPU或CPU
            if hasattr(self, 'single_gpu_processor') and self.single_gpu_processor:
                return self.single_gpu_processor.calculate_multiple_indicators_gpu(data, indicators, params)
            else:
                # CPU模式
                return self._calculate_multiple_indicators_cpu(data, indicators, params)

        # 数据分发
        data_chunks = self._load_balance_data(data)

        if not data_chunks:
            self.logger.warn("没有可用的GPU处理器")
            return pd.DataFrame()

        # 并行处理
        results = {}

        if self.config['sync_mode']:
            # 同步模式：使用ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(self.available_gpus)) as executor:
                future_to_gpu = {}

                for gpu_id, chunk in data_chunks.items():
                    future = executor.submit(
                        self._process_chunk_on_gpu,
                        gpu_id, chunk, indicators, params
                    )
                    future_to_gpu[future] = gpu_id

                for future in as_completed(future_to_gpu):
                    gpu_id = future_to_gpu[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            results[gpu_id] = result
                    except Exception as e:
                        self.logger.error(f"GPU {gpu_id} 处理异常: {e}")
        else:
            # 异步模式：直接并行处理
            threads = []
            for gpu_id, chunk in data_chunks.items():
                thread = threading.Thread(
                    target=lambda: results.update(
                        {gpu_id: self._process_chunk_on_gpu(gpu_id, chunk, indicators, params)})
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        # 结果聚合
        return self._aggregate_results(results, data.index)

    def _aggregate_results(self, results: Dict[int, pd.DataFrame], original_index: pd.Index) -> pd.DataFrame:
        """聚合多个GPU的结果"""
        if not results:
            return pd.DataFrame()

        # 收集所有结果
        all_results = []
        for gpu_id, result in results.items():
            if not result.empty:
                all_results.append(result)

        if not all_results:
            return pd.DataFrame()

        # 合并结果
        combined_result = pd.concat(all_results, axis=0)

        # 确保索引顺序正确，避免重复索引问题
        if len(combined_result) == len(original_index):
            try:
                # 重置索引以避免重复标签问题
                combined_result = combined_result.reset_index(drop=True)
                combined_result.index = original_index[:len(combined_result)]
            except Exception as e:
                self.logger.warn(f"结果聚合时索引处理失败: {e}")
                # 如果索引处理失败，直接返回合并结果

        return combined_result

    def _calculate_multiple_indicators_cpu(self, data: pd.DataFrame,


                                           indicators: List[str],
                                           params: Dict[str, Any] = None) -> pd.DataFrame:
        """CPU模式计算多个技术指标"""
        # 这里实现CPU版本的计算
        # 暂时返回空DataFrame
        return pd.DataFrame()

    def get_multi_gpu_info(self) -> Dict[str, Any]:
        """获取多GPU信息"""
        info = {
            'available_gpus': len(self.available_gpus),
            'total_gpus': len(self.gpu_processors),
            'gpu_details': {}
        }

        for gpu_id in self.available_gpus:
            try:
                torch.cuda.set_device(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3  # GB
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024 ** 3  # GB

                info['gpu_details'][gpu_id] = {
                    'name': self.gpu_info[gpu_id]['name'],
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'memory_usage_percent': (memory_allocated / self.gpu_info[gpu_id]['memory_gb']) * 100
                }
            except Exception as e:
                self.logger.warn(f"获取GPU {gpu_id} 信息失败: {e}")

        return info

    def clear_multi_gpu_memory(self):
        """清理多GPU内存"""
        for gpu_id in self.available_gpus:
            try:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()

                if gpu_id in self.gpu_processors:
                    self.gpu_processors[gpu_id].clear_gpu_memory()

            except Exception as e:
                self.logger.warn(f"清理GPU {gpu_id} 内存失败: {e}")

    def get_available_gpus(self) -> List[int]:
        """获取可用GPU列表"""
        return self.available_gpus.copy()

    def is_multi_gpu_available(self) -> bool:
        """检查多GPU是否可用"""
        return len(self.available_gpus) > 1
