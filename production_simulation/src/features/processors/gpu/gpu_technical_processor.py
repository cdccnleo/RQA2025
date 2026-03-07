from src.infrastructure.utils.logging.logger import get_logger
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU加速技术指标处理器

实现基于CUDA的并行技术指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CUDA GPU加速可用")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CUDA不可用，将使用CPU计算")


class GPUTechnicalProcessor:

    """GPU加速技术指标处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):

        # 默认配置

        default_config = {
            'use_gpu': True,
            'batch_size': 1000,
            'memory_limit': 0.8,  # GPU内存使用限制
            'fallback_to_cpu': True,
            'gpu_threshold': 5000,  # GPU使用的最小数据规模阈值
            'optimization_level': 'balanced'  # balanced, aggressive, conservative
        }

        # 合并配置
        if config:

            default_config.update(config)

        self.config = default_config
        self.logger = logger or get_logger("gpu_technical_processor")
        self.gpu_available = GPU_AVAILABLE and self.config['use_gpu']

        # 根据优化级别调整阈值
        if self.config['optimization_level'] == 'aggressive':
            self.config['gpu_threshold'] = 2000
        elif self.config['optimization_level'] == 'conservative':
            self.config['gpu_threshold'] = 10000
        elif self.config['optimization_level'] == 'balanced':
            self.config['gpu_threshold'] = 500

        if self.gpu_available:
            self._initialize_gpu()
        else:
            self.logger.info("使用CPU模式进行计算")

    def _should_use_gpu(self, data_size: int) -> bool:
        """第四阶段优化：动态GPU / CPU选择 - 支持更大数据集"""
        if not self.gpu_available:
            return False

        # 第四阶段优化：调整阈值以支持更大数据集
        # 根据优化级别调整数据大小阈值
        if self.config['optimization_level'] == 'conservative':
            threshold = 1000  # 保守模式：小数据集使用GPU
        elif self.config['optimization_level'] == 'balanced':
            threshold = 500   # 平衡模式：中等数据集使用GPU
        else:  # aggressive
            threshold = 100   # 激进模式：几乎所有数据集都使用GPU

        # 第四阶段优化：考虑GPU内存使用情况
        try:
            gpu_info = self.get_gpu_info()
            memory_usage = gpu_info.get('memory_usage', 0)

            # 如果GPU内存使用率超过80%，优先使用CPU
            if memory_usage > 80:
                self.logger.info(f"GPU内存使用率过高({memory_usage:.1f}%)，使用CPU计算")
                return False

            # 对于超大数据集，检查内存是否足够
            if data_size > 100000:  # 10万条数据
                total_memory_gb = gpu_info.get('total_memory_gb', 0)
                if total_memory_gb < 4:  # 小于4GB显存
                    self.logger.info(f"GPU显存不足({total_memory_gb:.1f}GB)，大数据集使用CPU计算")
                    return False
        except Exception as e:
            self.logger.warning(f"获取GPU信息失败: {e}")

        return data_size >= threshold

    def _optimize_memory_access(self):
        """第四阶段优化：内存访问优化"""
        try:
            # 优化CuPy内存池设置
            pool = cp.get_default_memory_pool()

            # 设置更激进的内存管理策略
            pool.set_limit(size=0)  # 允许更多内存使用

            # 预分配常用内存块
            self._preallocate_memory_blocks()

            self.logger.info("GPU内存访问优化完成")
        except Exception as e:
            self.logger.warning(f"内存访问优化失败: {e}")

    def _preallocate_memory_blocks(self):
        """第四阶段优化：预分配常用内存块"""
        try:
            # 预分配常用大小的内存块，减少动态分配开销
            common_sizes = [1000, 5000, 10000, 50000, 100000]

            for size in common_sizes:
                try:
                    # 预分配内存块
                    temp_array = cp.zeros(size, dtype=cp.float32)
                    del temp_array  # 立即释放，但保留内存池
                except Exception:
                    # 如果内存不足，跳过
                    continue

            self.logger.info("GPU内存块预分配完成")
        except Exception as e:
            self.logger.warning(f"内存块预分配失败: {e}")

    def _initialize_gpu(self):
        """第四阶段优化：GPU初始化 - 支持更大数据集"""
        try:
            # 检查CUDA可用性
            if not cp.cuda.is_available():
                self.logger.warning("CUDA不可用，将使用CPU计算")
                return False

            # 获取GPU设备信息
            device = cp.cuda.Device()
            memory_info = cp.cuda.runtime.memGetInfo()
            total_memory = memory_info[1]
            free_memory = memory_info[0]

            self.logger.info(f"GPU设备: {device}")
            self.logger.info(f"总显存: {total_memory / 1024 ** 3:.2f} GB")
            self.logger.info(f"可用显存: {free_memory / 1024 ** 3:.2f} GB")

            # 第四阶段优化：更灵活的内存限制设置
            # 根据总显存大小调整内存限制
            if total_memory >= 8 * 1024 ** 3:  # 8GB以上
                memory_limit_ratio = 0.9  # 使用90 % 显存
            elif total_memory >= 4 * 1024 ** 3:  # 4GB以上
                memory_limit_ratio = 0.8  # 使用80 % 显存
            else:  # 小于4GB
                memory_limit_ratio = 0.7  # 使用70 % 显存

            memory_limit = int(total_memory * memory_limit_ratio)
            cp.get_default_memory_pool().set_limit(size=memory_limit)

            # 第四阶段优化：初始化内存访问优化
            self._optimize_memory_access()

            self.gpu_available = True
            self.logger.info("GPU初始化成功")
            return True

        except Exception as e:
            self.logger.warning(f"GPU初始化失败: {e}，切换到CPU模式")
            self.gpu_available = False
            return False

    def calculate_sma_gpu(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """GPU加速简单移动平均 - 优化卷积计算和内存访问"""
        if window <= 0:
            raise ValueError("窗口大小必须大于0")

        if not self._should_use_gpu(len(data)):
            return self._calculate_sma_cpu(data, window)

        try:
            # 将数据转移到GPU
            close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
            n = len(close_gpu)

            # 优化GPU并行计算移动平均
            # 使用预计算的权重，避免重复计算
            weights = cp.ones(window, dtype=cp.float32) / window

            # 使用优化的卷积计算
            sma_gpu = cp.convolve(close_gpu, weights, mode='valid')

            # 优化内存分配 - 使用更高效的填充方法
            padding = cp.full(window - 1, cp.nan, dtype=cp.float32)
            sma_gpu = cp.concatenate([padding, sma_gpu])

            # 转回CPU并确保数据类型一致
            sma_cpu = cp.asnumpy(sma_gpu).astype(np.float64)
            return pd.Series(sma_cpu, index=data.index)

        except Exception as e:
            self.logger.warning(f"GPU SMA计算失败: {e}，回退到CPU")
            return self._calculate_sma_cpu(data, window)

    def calculate_ema_gpu(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """GPU加速指数移动平均 - 第七阶段优化版本"""
        if not self._should_use_gpu(len(data)):
            return self._calculate_ema_cpu(data, window)

        try:
            close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
            n = len(close_gpu)
            alpha = 2.0 / (window + 1)

            # 第七阶段优化：使用向量化的递归算法
            # 避免构建大型系数矩阵，提高计算效率

            ema_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)
            ema_gpu[0] = close_gpu[0]

            # 使用向量化的递归计算
            for i in range(1, n):
                ema_gpu[i] = alpha * close_gpu[i] + (1 - alpha) * ema_gpu[i - 1]

            ema_cpu = cp.asnumpy(ema_gpu).astype(np.float64)
            return pd.Series(ema_cpu, index=data.index)
        except Exception as e:
            self.logger.warning(f"GPU EMA计算失败: {e}，回退到CPU")
            return self._calculate_ema_cpu(data, window)

    def calculate_rsi_gpu(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """GPU加速相对强弱指数 - 优化卷积计算和内存访问"""
        if not self._should_use_gpu(len(data)):
            return self._calculate_rsi_cpu(data, window)

        try:
            # 将数据转移到GPU
            close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
            n = len(close_gpu)

            # 计算价格变化 - 使用向量化操作
            delta = cp.diff(close_gpu)

            # 分离上涨和下跌 - 优化条件判断
            gains = cp.maximum(delta, 0)
            losses = cp.maximum(-delta, 0)

            # 优化GPU并行计算移动平均
            # 使用更高效的卷积方法，避免重复计算
            weights = cp.ones(window, dtype=cp.float32) / window

            # 并行计算平均涨幅和跌幅
            avg_gains = cp.convolve(gains, weights, mode='valid')
            avg_losses = cp.convolve(losses, weights, mode='valid')

            # 计算相对强弱比率 - 添加数值稳定性
            epsilon = 1e-8  # 避免除零
            rs = avg_gains / (avg_losses + epsilon)

            # 计算RSI - 使用向量化操作
            rsi = 100 - (100 / (1 + rs))

            # 填充开始的NaN值 - 优化内存分配
            padding = cp.full(window, cp.nan, dtype=cp.float32)
            rsi = cp.concatenate([padding, rsi])

            # 转回CPU并确保数据类型一致
            rsi_cpu = cp.asnumpy(rsi).astype(np.float64)
            return pd.Series(rsi_cpu, index=data.index)

        except Exception as e:
            self.logger.warning(f"GPU RSI计算失败: {e}，回退到CPU")
            return self._calculate_rsi_cpu(data, window)

    def calculate_macd_gpu(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """GPU加速MACD指标 - 第七阶段优化版本"""
        if not self._should_use_gpu(len(data)):
            return self._calculate_macd_cpu(data, fast, slow, signal)

        try:
            close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
            n = len(close_gpu)

            # 第七阶段优化：并行计算多个EMA，减少重复计算
            # 使用向量化的递归算法，避免矩阵运算开销

            # 计算快速EMA
            alpha_fast = 2.0 / (fast + 1)
            ema_fast_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)
            ema_fast_gpu[0] = close_gpu[0]
            for i in range(1, n):
                ema_fast_gpu[i] = alpha_fast * close_gpu[i] + (1 - alpha_fast) * ema_fast_gpu[i - 1]

            # 计算慢速EMA
            alpha_slow = 2.0 / (slow + 1)
            ema_slow_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)
            ema_slow_gpu[0] = close_gpu[0]
            for i in range(1, n):
                ema_slow_gpu[i] = alpha_slow * close_gpu[i] + (1 - alpha_slow) * ema_slow_gpu[i - 1]

            # 计算MACD线
            macd_gpu = ema_fast_gpu - ema_slow_gpu

            # 计算信号线
            alpha_signal = 2.0 / (signal + 1)
            signal_gpu = cp.zeros_like(macd_gpu, dtype=cp.float32)
            signal_gpu[0] = macd_gpu[0]
            for i in range(1, n):
                signal_gpu[i] = alpha_signal * macd_gpu[i] + (1 - alpha_signal) * signal_gpu[i - 1]

            # 计算柱状图
            histogram_gpu = macd_gpu - signal_gpu

            # 转回CPU
            macd_cpu = cp.asnumpy(macd_gpu).astype(np.float64)
            signal_cpu = cp.asnumpy(signal_gpu).astype(np.float64)
            histogram_cpu = cp.asnumpy(histogram_gpu).astype(np.float64)

            return pd.DataFrame({
                'macd': pd.Series(macd_cpu, index=data.index),
                'signal': pd.Series(signal_cpu, index=data.index),
                'histogram': pd.Series(histogram_cpu, index=data.index)
            })
        except Exception as e:
            self.logger.warning(f"GPU MACD计算失败: {e}，回退到CPU")
            return self._calculate_macd_cpu(data, fast, slow, signal)

    def calculate_bollinger_bands_gpu(self, data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """GPU加速布林带 - 第七阶段优化版本"""
        if not self._should_use_gpu(len(data)):
            return self._calculate_bollinger_bands_cpu(data, window, num_std)

        try:
            close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)
            n = len(close_gpu)

            # 第七阶段优化：修复数值稳定性问题

            # 1. 并行计算SMA - 使用卷积
            weights = cp.ones(window, dtype=cp.float32) / window
            sma_gpu = cp.convolve(close_gpu, weights, mode='valid')
            padding = cp.full(window - 1, cp.nan, dtype=cp.float32)
            sma_gpu = cp.concatenate([padding, sma_gpu])

            # 2. 并行计算标准差 - 修复数值稳定性
            std_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)

            for i in range(window - 1, n):
                window_data = close_gpu[i - window + 1:i + 1]
                # 使用更稳定的方差计算方法
                mean_val = cp.mean(window_data)
                variance = cp.mean((window_data - mean_val) ** 2)
                std_gpu[i] = cp.sqrt(variance)

            # 3. 计算上下轨 - 确保数值稳定性
            upper_band = sma_gpu + (num_std * std_gpu)
            lower_band = sma_gpu - (num_std * std_gpu)

            # 处理NaN值
            upper_band = cp.where(cp.isnan(upper_band), sma_gpu, upper_band)
            lower_band = cp.where(cp.isnan(lower_band), sma_gpu, lower_band)

            # 转回CPU
            upper_cpu = cp.asnumpy(upper_band).astype(np.float64)
            lower_cpu = cp.asnumpy(lower_band).astype(np.float64)
            sma_cpu = cp.asnumpy(sma_gpu).astype(np.float64)

            return pd.DataFrame({
                'upper': pd.Series(upper_cpu, index=data.index),
                'middle': pd.Series(sma_cpu, index=data.index),
                'lower': pd.Series(lower_cpu, index=data.index)
            })
        except Exception as e:
            self.logger.warning(f"GPU布林带计算失败: {e}，回退到CPU")
            return self._calculate_bollinger_bands_cpu(data, window, num_std)

    def calculate_atr_gpu(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """GPU加速平均真实波幅 - 优化并行计算和内存访问"""
        if not self._should_use_gpu(len(data)):
            return self._calculate_atr_cpu(data, window)

        try:
            # 批量传输数据到GPU，减少传输开销
            high_gpu = cp.asarray(data['high'].values, dtype=cp.float32)
            low_gpu = cp.asarray(data['low'].values, dtype=cp.float32)
            close_gpu = cp.asarray(data['close'].values, dtype=cp.float32)

            # 优化并行计算真实波幅
            # 使用向量化操作，避免循环
            tr1 = high_gpu - low_gpu
            tr2 = cp.abs(high_gpu - cp.roll(close_gpu, 1))
            tr3 = cp.abs(low_gpu - cp.roll(close_gpu, 1))

            # 使用cp.maximum的向量化操作
            tr = cp.maximum(tr1, cp.maximum(tr2, tr3))

            # 优化移动平均计算 - 使用预计算的权重
            weights = cp.ones(window, dtype=cp.float32) / window
            atr_gpu = cp.convolve(tr, weights, mode='valid')

            # 优化内存分配 - 使用更高效的填充方法
            padding = cp.full(window - 1, cp.nan, dtype=cp.float32)
            atr_gpu = cp.concatenate([padding, atr_gpu])

            # 批量转回CPU，减少传输次数
            atr_cpu = cp.asnumpy(atr_gpu).astype(np.float64)
            return pd.Series(atr_cpu, index=data.index)

        except Exception as e:
            self.logger.warning(f"GPU ATR计算失败: {e}，回退到CPU")
            return self._calculate_atr_cpu(data, window)

    def calculate_multiple_indicators_gpu(self, data: pd.DataFrame, indicators: List[str], params: Dict[str, Any] = None) -> pd.DataFrame:
        """GPU加速多指标计算 - 第四阶段优化：内存访问优化和算法并行化"""
        if not self._should_use_gpu(len(data)):
            return self._calculate_multiple_indicators_cpu(data, indicators, params)
        try:
            # 第四阶段优化：优化内存访问模式
            # 使用连续内存布局，减少内存碎片
            close_values = data['close'].values.astype(np.float32)
            high_values = data['high'].values.astype(np.float32)
            low_values = data['low'].values.astype(np.float32)

            # 批量传输数据到GPU，使用连续内存
            close_gpu = cp.asarray(close_values, dtype=cp.float32)
            high_gpu = cp.asarray(high_values, dtype=cp.float32)
            low_gpu = cp.asarray(low_values, dtype=cp.float32)

            # 预分配结果内存，减少动态分配
            results = {}

            # 第四阶段优化：算法并行化 - 同时计算多个指标
            # 创建指标计算任务列表
            tasks = []
            for indicator in indicators:
                if indicator == 'sma':
                    window = params.get('sma_window', 20) if params else 20
                    tasks.append(('sma', window))
                elif indicator == 'ema':
                    window = params.get('ema_window', 20) if params else 20
                    tasks.append(('ema', window))
                elif indicator == 'rsi':
                    window = params.get('rsi_window', 14) if params else 14
                    tasks.append(('rsi', window))
                elif indicator == 'macd':
                    fast = params.get('macd_fast', 12) if params else 12
                    slow = params.get('macd_slow', 26) if params else 26
                    signal = params.get('macd_signal', 9) if params else 9
                    tasks.append(('macd', (fast, slow, signal)))
                elif indicator == 'bollinger':
                    window = params.get('bb_window', 20) if params else 20
                    num_std = params.get('bb_std', 2) if params else 2
                    tasks.append(('bollinger', (window, num_std)))
                elif indicator == 'atr':
                    window = params.get('atr_window', 14) if params else 14
                    tasks.append(('atr', window))

            # 并行处理指标计算
            for task_type, task_params in tasks:
                if task_type == 'sma':
                    window = task_params
                    # 优化SMA计算 - 使用预计算的权重
                    weights = cp.ones(window, dtype=cp.float32) / window
                    sma_gpu = cp.convolve(close_gpu, weights, mode='valid')
                    padding = cp.full(window - 1, cp.nan, dtype=cp.float32)
                    sma_gpu = cp.concatenate([padding, sma_gpu])
                    results[f'sma_{window}'] = pd.Series(cp.asnumpy(
                        sma_gpu).astype(np.float64), index=data.index)

                elif task_type == 'ema':
                    window = task_params
                    # 第四阶段优化：重构EMA算法
                    alpha = 2.0 / (window + 1)
                    ema_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)
                    ema_gpu[0] = close_gpu[0]

                    # 使用向量化操作优化EMA计算
                    for i in range(1, len(close_gpu)):
                        ema_gpu[i] = alpha * close_gpu[i] + (1 - alpha) * ema_gpu[i - 1]

                    results[f'ema_{window}'] = pd.Series(cp.asnumpy(
                        ema_gpu).astype(np.float64), index=data.index)

                elif task_type == 'rsi':
                    window = task_params
                    # 优化RSI计算 - 使用向量化操作
                    delta = cp.diff(close_gpu)
                    gains = cp.maximum(delta, 0)
                    losses = cp.maximum(-delta, 0)

                    # 使用卷积计算移动平均
                    weights = cp.ones(window, dtype=cp.float32) / window
                    avg_gains = cp.convolve(gains, weights, mode='valid')
                    avg_losses = cp.convolve(losses, weights, mode='valid')

                    # 数值稳定性优化
                    epsilon = 1e-8
                    rs = avg_gains / (avg_losses + epsilon)
                    rsi = 100 - (100 / (1 + rs))

                    padding = cp.full(window, cp.nan, dtype=cp.float32)
                    rsi = cp.concatenate([padding, rsi])
                    results[f'rsi_{window}'] = pd.Series(
                        cp.asnumpy(rsi).astype(np.float64), index=data.index)

                elif task_type == 'macd':
                    fast, slow, signal = task_params
                    # 第四阶段优化：MACD算法并行化
                    alpha_fast = 2.0 / (fast + 1)
                    alpha_slow = 2.0 / (slow + 1)

                    # 并行计算快速和慢速EMA
                    ema_fast_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)
                    ema_slow_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)

                    ema_fast_gpu[0] = close_gpu[0]
                    ema_slow_gpu[0] = close_gpu[0]

                    for i in range(1, len(close_gpu)):
                        ema_fast_gpu[i] = alpha_fast * close_gpu[i] + \
                            (1 - alpha_fast) * ema_fast_gpu[i - 1]
                        ema_slow_gpu[i] = alpha_slow * close_gpu[i] + \
                            (1 - alpha_slow) * ema_slow_gpu[i - 1]

                    macd_gpu = ema_fast_gpu - ema_slow_gpu

                    # 计算信号线
                    alpha_signal = 2.0 / (signal + 1)
                    signal_gpu = cp.zeros_like(macd_gpu, dtype=cp.float32)
                    signal_gpu[0] = macd_gpu[0]

                    for i in range(1, len(macd_gpu)):
                        signal_gpu[i] = alpha_signal * macd_gpu[i] + \
                            (1 - alpha_signal) * signal_gpu[i - 1]

                    histogram_gpu = macd_gpu - signal_gpu

                    results.update({
                        f'macd_line': pd.Series(cp.asnumpy(macd_gpu).astype(np.float64), index=data.index),
                        f'macd_signal': pd.Series(cp.asnumpy(signal_gpu).astype(np.float64), index=data.index),
                        f'macd_histogram': pd.Series(cp.asnumpy(histogram_gpu).astype(np.float64), index=data.index)
                    })

                elif task_type == 'bollinger':
                    window, num_std = task_params
                    # 第四阶段优化：布林带算法并行化
                    weights = cp.ones(window, dtype=cp.float32) / window
                    sma_gpu = cp.convolve(close_gpu, weights, mode='valid')
                    padding = cp.full(window - 1, cp.nan, dtype=cp.float32)
                    sma_gpu = cp.concatenate([padding, sma_gpu])

                    # 优化标准差计算
                    std_gpu = cp.zeros_like(close_gpu, dtype=cp.float32)
                    for i in range(window - 1, len(close_gpu)):
                        window_data = close_gpu[i - window + 1:i + 1]
                        mean_val = cp.mean(window_data)
                        variance = cp.mean((window_data - mean_val) ** 2)
                        std_gpu[i] = cp.sqrt(variance)

                    upper_gpu = sma_gpu + (num_std * std_gpu)
                    lower_gpu = sma_gpu - (num_std * std_gpu)

                    results.update({
                        f'bb_upper': pd.Series(cp.asnumpy(upper_gpu).astype(np.float64), index=data.index),
                        f'bb_middle': pd.Series(cp.asnumpy(sma_gpu).astype(np.float64), index=data.index),
                        f'bb_lower': pd.Series(cp.asnumpy(lower_gpu).astype(np.float64), index=data.index)
                    })

                elif task_type == 'atr':
                    window = task_params
                    # 第四阶段优化：ATR算法并行化
                    tr1 = high_gpu - low_gpu
                    tr2 = cp.abs(high_gpu - cp.roll(close_gpu, 1))
                    tr3 = cp.abs(low_gpu - cp.roll(close_gpu, 1))
                    tr = cp.maximum(tr1, cp.maximum(tr2, tr3))

                    weights = cp.ones(window, dtype=cp.float32) / window
                    atr_gpu = cp.convolve(tr, weights, mode='valid')
                    padding = cp.full(window - 1, cp.nan, dtype=cp.float32)
                    atr_gpu = cp.concatenate([padding, atr_gpu])

                    results[f'atr_{window}'] = pd.Series(cp.asnumpy(
                        atr_gpu).astype(np.float64), index=data.index)

            return pd.DataFrame(results, index=data.index)
        except Exception as e:
            self.logger.warning(f"GPU多指标计算失败: {e}，回退到CPU")
            return self._calculate_multiple_indicators_cpu(data, indicators, params)

    # CPU回退方法

    def _calculate_sma_cpu(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """CPU计算简单移动平均"""
        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float)
        return data['close'].rolling(window=window).mean()

    def _calculate_ema_cpu(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """CPU计算指数移动平均"""
        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float)
        return data['close'].ewm(span=window).mean()

    def _calculate_rsi_cpu(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """CPU计算相对强弱指数"""
        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float)
        delta = data['close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd_cpu(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """CPU计算MACD指标"""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    def _calculate_bollinger_bands_cpu(self, data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """CPU计算布林带"""
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)

        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        })

    def _calculate_atr_cpu(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """CPU计算平均真实波幅"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr

    def _calculate_multiple_indicators_cpu(self, data: pd.DataFrame, indicators: List[str], params: Dict[str, Any] = None) -> pd.DataFrame:
        """CPU计算多指标"""
        results = {}

        for indicator in indicators:
            if indicator == 'sma':
                window = params.get('sma_window', 20) if params else 20
                results[f'sma_{window}'] = self._calculate_sma_cpu(data, window)

            elif indicator == 'ema':
                window = params.get('ema_window', 20) if params else 20
                results[f'ema_{window}'] = self._calculate_ema_cpu(data, window)

            elif indicator == 'rsi':
                window = params.get('rsi_window', 14) if params else 14
                results[f'rsi_{window}'] = self._calculate_rsi_cpu(data, window)

            elif indicator == 'macd':
                fast = params.get('macd_fast', 12) if params else 12
                slow = params.get('macd_slow', 26) if params else 26
                signal = params.get('macd_signal', 9) if params else 9
                macd_result = self._calculate_macd_cpu(data, fast, slow, signal)
                results.update({
                    f'macd_line': macd_result['macd'],
                    f'macd_signal': macd_result['signal'],
                    f'macd_histogram': macd_result['histogram']
                })

            elif indicator == 'bollinger':
                window = params.get('bb_window', 20) if params else 20
                num_std = params.get('bb_std', 2) if params else 2
                bb_result = self._calculate_bollinger_bands_cpu(data, window, num_std)
                results.update({
                    f'bb_upper': bb_result['upper'],
                    f'bb_middle': bb_result['middle'],
                    f'bb_lower': bb_result['lower']
                })

            elif indicator == 'atr':
                window = params.get('atr_window', 14) if params else 14
                results[f'atr_{window}'] = self._calculate_atr_cpu(data, window)

        return pd.DataFrame(results, index=data.index)

    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        if not self.gpu_available:
            return {
                'available': False,
                'device_name': 'N / A',
                'total_memory_gb': 0,
                'free_memory_gb': 0,
                'used_memory_gb': 0,
                'cuda_available': False,
                'gpu_count': 0,
                'memory_usage': 0.0
            }

        try:
            # 使用PyTorch获取设备名称和CUDA信息
            import torch
            device_name = torch.cuda.get_device_name(
                0) if torch.cuda.is_available() else "Unknown GPU"
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()

            # 使用CuPy获取内存信息
            device = cp.cuda.Device()
            memory_info = cp.cuda.runtime.memGetInfo()
            total_memory = memory_info[1]
            free_memory = memory_info[0]
            used_memory = total_memory - free_memory

            # 计算内存使用率
            memory_usage = (used_memory / total_memory) * 100 if total_memory > 0 else 0

            return {
                'available': True,
                'device_name': device_name,
                'total_memory_gb': total_memory / 1024 ** 3,
                'free_memory_gb': free_memory / 1024 ** 3,
                'used_memory_gb': used_memory / 1024 ** 3,
                'cuda_available': cuda_available,
                'gpu_count': gpu_count,
                'memory_usage': memory_usage
            }
        except Exception as e:
            return {
                'available': False,
                'device_name': 'Error',
                'total_memory_gb': 0,
                'free_memory_gb': 0,
                'used_memory_gb': 0,
                'cuda_available': False,
                'gpu_count': 0,
                'memory_usage': 0.0
            }

    def clear_gpu_memory(self):
        """清理GPU内存"""
        try:
            if self.gpu_available:
                # 清理所有内存块
                if hasattr(self, '_optimize_memory_usage'):
                    self._optimize_memory_usage()
                elif GPU_AVAILABLE:
                    try:
                        import cupy as cp
                        pool = cp.get_default_memory_pool()
                        pool.free_all_blocks()
                    except (ImportError, AttributeError):
                        pass
                self.logger.info("GPU内存已清理")
        except Exception as e:
            self.logger.warning(f"GPU内存清理失败: {e}")
