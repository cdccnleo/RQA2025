import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式特征处理器

实现大规模特征计算的分布式处理
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

try:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    DISTRIBUTED_AVAILABLE = True
    logger.info("分布式处理可用")
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    logger.warning("分布式处理不可用")


class DistributedFeatureProcessor:

    """分布式特征处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {
            'use_distributed': True,
            'max_workers': None,  # 自动检测CPU核心数
            'chunk_size': 1000,
            'memory_limit_mb': 1024,
            'timeout_seconds': 300,
            'fallback_to_sequential': True
        }
        self.logger = logger
        self.distributed_available = DISTRIBUTED_AVAILABLE and self.config['use_distributed']

        if self.distributed_available:
            self._initialize_distributed()
        else:
            logger.info("使用顺序处理模式")

    def _initialize_distributed(self):
        """初始化分布式环境"""
        try:
            if self.config['max_workers'] is None:
                self.config['max_workers'] = mp.cpu_count()

            logger.info(f"分布式处理器初始化完成，工作进程数: {self.config['max_workers']}")
        except Exception as e:
            self.distributed_available = False
            logger.warning(f"分布式初始化失败: {e}，切换到顺序模式")

    def _chunk_data(self, data: pd.DataFrame, chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """将数据分块"""
        chunk_size = chunk_size or self.config['chunk_size']
        chunks = []

        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)

        return chunks

    def _process_chunk_technical(self, chunk_data: pd.DataFrame, indicators: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """处理单个数据块的技术指标"""
        try:
            # 直接计算指标，避免序列化问题
            results = {}

            for indicator in indicators:
                if indicator == 'sma':
                    window = params.get('sma_window', 20)
                    results['sma'] = chunk_data['close'].rolling(window=window).mean()

                elif indicator == 'ema':
                    window = params.get('ema_window', 20)
                    results['ema'] = chunk_data['close'].ewm(span=window).mean()

                elif indicator == 'rsi':
                    window = params.get('rsi_window', 14)
                    delta = chunk_data['close'].diff()
                    gains = delta.where(delta > 0, 0)
                    losses = -delta.where(delta < 0, 0)
                    avg_gains = gains.rolling(window=window).mean()
                    avg_losses = losses.rolling(window=window).mean()
                    rs = avg_gains / avg_losses
                    results['rsi'] = 100 - (100 / (1 + rs))

                elif indicator == 'macd':
                    fast = params.get('macd_fast', 12)
                    slow = params.get('macd_slow', 26)
                    signal = params.get('macd_signal', 9)
                    ema_fast = chunk_data['close'].ewm(span=fast).mean()
                    ema_slow = chunk_data['close'].ewm(span=slow).mean()
                    macd_line = ema_fast - ema_slow
                    signal_line = macd_line.ewm(span=signal).mean()
                    histogram = macd_line - signal_line
                    results['macd_macd'] = macd_line
                    results['macd_signal'] = signal_line
                    results['macd_histogram'] = histogram

            return pd.DataFrame(results, index=chunk_data.index)

        except Exception as e:
            logger.error(f"处理数据块失败: {e}")
            # 返回空结果
            return pd.DataFrame(index=chunk_data.index)

    def _process_chunk_gpu(self, chunk_data: pd.DataFrame, indicators: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """处理单个数据块的GPU加速指标"""
        try:
            from ...gpu.gpu_technical_processor import GPUTechnicalProcessor

            processor = GPUTechnicalProcessor()
            result = processor.calculate_multiple_indicators_gpu(chunk_data, indicators, params)
            return result
        except Exception as e:
            logger.error(f"GPU处理数据块失败: {e}")
            # 回退到CPU处理
            return self._process_chunk_technical(chunk_data, indicators, params)

    def calculate_distributed_technical_features(self,


                                                 data: pd.DataFrame,
                                                 indicators: List[str],
                                                 params: Dict[str, Any] = None,
                                                 use_gpu: bool = False) -> pd.DataFrame:
        """分布式计算技术指标特征"""
        if not self.distributed_available:
            return self._calculate_sequential_technical_features(data, indicators, params, use_gpu)

        try:
            # 数据分块
            chunks = self._chunk_data(data)
            logger.info(f"数据已分块，共 {len(chunks)} 个块")

            # 选择处理函数
            process_func = self._process_chunk_gpu if use_gpu else self._process_chunk_technical

            # 使用进程池并行处理
            results = []
            with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
                # 提交任务
                future_to_chunk = {
                    executor.submit(process_func, chunk, indicators, params or {}): i
                    for i, chunk in enumerate(chunks)
                }

                # 收集结果
                for future in as_completed(future_to_chunk, timeout=self.config['timeout_seconds']):
                    chunk_index = future_to_chunk[future]
                    try:
                        result = future.result()
                        results.append((chunk_index, result))
                        logger.info(f"数据块 {chunk_index + 1}/{len(chunks)} 处理完成")
                    except Exception as e:
                        logger.error(f"数据块 {chunk_index + 1} 处理失败: {e}")
                        # 使用顺序处理作为回退
                        chunk = chunks[chunk_index]
                        fallback_result = self._calculate_sequential_technical_features(
                            chunk, indicators, params, use_gpu)
                        results.append((chunk_index, fallback_result))

            # 按原始顺序合并结果
            results.sort(key=lambda x: x[0])
            combined_result = pd.concat([result for _, result in results], axis=0)

            # 确保索引顺序正确
            combined_result = combined_result.reindex(data.index)

            logger.info(f"分布式处理完成，结果形状: {combined_result.shape}")
            return combined_result

        except Exception as e:
            logger.warning(f"分布式处理失败: {e}，回退到顺序处理")
            return self._calculate_sequential_technical_features(data, indicators, params, use_gpu)

    def calculate_distributed_quality_features(self,


                                               data: pd.DataFrame,
                                               target: pd.Series,
                                               quality_metrics: List[str] = None) -> pd.DataFrame:
        """分布式计算质量特征"""
        if not self.distributed_available:
            return self._calculate_sequential_quality_features(data, target, quality_metrics)

        try:
            quality_metrics = quality_metrics or ['importance', 'correlation', 'stability']

            # 数据分块
            chunks = self._chunk_data(data)
            target_chunks = [target.iloc[i:i + self.config['chunk_size']]
                             for i in range(0, len(target), self.config['chunk_size'])]

            logger.info(f"质量特征数据已分块，共 {len(chunks)} 个块")

            # 使用进程池并行处理
            results = []
            with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
                # 提交任务
                future_to_chunk = {
                    executor.submit(self._process_chunk_quality, chunk, target_chunk, quality_metrics): i
                    for i, (chunk, target_chunk) in enumerate(zip(chunks, target_chunks))
                }

                # 收集结果
                for future in as_completed(future_to_chunk, timeout=self.config['timeout_seconds']):
                    chunk_index = future_to_chunk[future]
                    try:
                        result = future.result()
                        results.append((chunk_index, result))
                        logger.info(f"质量特征块 {chunk_index + 1}/{len(chunks)} 处理完成")
                    except Exception as e:
                        logger.error(f"质量特征块 {chunk_index + 1} 处理失败: {e}")
                        # 使用顺序处理作为回退
                        chunk = chunks[chunk_index]
                        target_chunk = target_chunks[chunk_index]
                        fallback_result = self._calculate_sequential_quality_features(
                            chunk, target_chunk, quality_metrics)
                        results.append((chunk_index, fallback_result))

            # 按原始顺序合并结果
            results.sort(key=lambda x: x[0])
            combined_result = pd.concat([result for _, result in results], axis=0)

            # 确保索引顺序正确
            combined_result = combined_result.reindex(data.index)

            logger.info(f"分布式质量特征处理完成，结果形状: {combined_result.shape}")
            return combined_result

        except Exception as e:
            logger.warning(f"分布式质量特征处理失败: {e}，回退到顺序处理")
            return self._calculate_sequential_quality_features(data, target, quality_metrics)

    def _process_chunk_quality(self, chunk_data: pd.DataFrame, target_chunk: pd.Series, quality_metrics: List[str]) -> pd.DataFrame:
        """处理单个数据块的质量特征"""
        try:
            results = {}

            if 'importance' in quality_metrics:
                from ...feature_importance import FeatureImportanceAnalyzer
                analyzer = FeatureImportanceAnalyzer()
                importance_result = analyzer.analyze_feature_importance(
                    chunk_data, target_chunk, 'regression')
                results['importance_score'] = pd.Series(
                    importance_result['combined_importance'], index=chunk_data.index)

            if 'correlation' in quality_metrics:
                from ...feature_correlation import FeatureCorrelationAnalyzer
                analyzer = FeatureCorrelationAnalyzer()
                correlation_result = analyzer.analyze_feature_correlation(chunk_data)
                # 简化相关性结果
                results['correlation_score'] = pd.Series(0.5, index=chunk_data.index)  # 占位符

            if 'stability' in quality_metrics:
                from ...feature_stability import FeatureStabilityAnalyzer
                analyzer = FeatureStabilityAnalyzer()
                stability_result = analyzer.analyze_feature_stability(chunk_data)
                results['stability_score'] = pd.Series(
                    stability_result['combined_stability'], index=chunk_data.index)

            return pd.DataFrame(results, index=chunk_data.index)

        except Exception as e:
            logger.error(f"处理质量特征块失败: {e}")
            return pd.DataFrame(index=chunk_data.index)

    def _calculate_sequential_technical_features(self,


                                                 data: pd.DataFrame,
                                                 indicators: List[str],
                                                 params: Dict[str, Any] = None,
                                                 use_gpu: bool = False) -> pd.DataFrame:
        """顺序计算技术指标特征"""
        try:
            if use_gpu:
                from ...gpu.gpu_technical_processor import GPUTechnicalProcessor
                processor = GPUTechnicalProcessor()
                return processor.calculate_multiple_indicators_gpu(data, indicators, params or {})
            else:
                from ...technical.technical_processor import TechnicalProcessor
                processor = TechnicalProcessor()
                return processor.calculate_multiple_indicators(data, indicators, params or {})
        except Exception as e:
            logger.error(f"顺序处理失败: {e}")
            return pd.DataFrame(index=data.index)

    def _calculate_sequential_quality_features(self,


                                               data: pd.DataFrame,
                                               target: pd.Series,
                                               quality_metrics: List[str] = None) -> pd.DataFrame:
        """顺序计算质量特征"""
        try:
            quality_metrics = quality_metrics or ['importance', 'correlation', 'stability']
            results = {}

            if 'importance' in quality_metrics:
                from ...feature_importance import FeatureImportanceAnalyzer
                analyzer = FeatureImportanceAnalyzer()
                importance_result = analyzer.analyze_feature_importance(data, target, 'regression')
                results['importance_score'] = pd.Series(
                    importance_result['combined_importance'], index=data.index)

            if 'correlation' in quality_metrics:
                from ...feature_correlation import FeatureCorrelationAnalyzer
                analyzer = FeatureCorrelationAnalyzer()
                correlation_result = analyzer.analyze_feature_correlation(data)
                # 简化相关性结果
                results['correlation_score'] = pd.Series(0.5, index=data.index)  # 占位符

            if 'stability' in quality_metrics:
                from ...feature_stability import FeatureStabilityAnalyzer
                analyzer = FeatureStabilityAnalyzer()
                stability_result = analyzer.analyze_feature_stability(data)
                results['stability_score'] = pd.Series(
                    stability_result['combined_stability'], index=data.index)

            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            logger.error(f"顺序质量特征处理失败: {e}")
            return pd.DataFrame(index=data.index)

    def get_distributed_info(self) -> Dict[str, Any]:
        """获取分布式处理信息"""
        if not self.distributed_available:
            return {'available': False, 'reason': '分布式处理不可用'}

        try:
            cpu_count = mp.cpu_count()
            return {
                'available': True,
                'cpu_count': cpu_count,
                'max_workers': self.config['max_workers'],
                'chunk_size': self.config['chunk_size'],
                'memory_limit_mb': self.config['memory_limit_mb']
            }
        except Exception as e:
            return {'available': False, 'reason': str(e)}

    def optimize_chunk_size(self, data_size: int, memory_limit_mb: Optional[int] = None) -> int:
        """优化分块大小"""
        memory_limit_mb = memory_limit_mb or self.config['memory_limit_mb']

        # 估算每个数据点的内存使用（字节）
        estimated_bytes_per_point = 100  # 保守估计

        # 计算最佳分块大小
        optimal_chunk_size = (memory_limit_mb * 1024 * 1024) // estimated_bytes_per_point

        # 确保分块大小在合理范围内
        min_chunk_size = 100
        max_chunk_size = 10000

        optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))

        # 如果数据量较小，使用较小的分块
        if data_size < optimal_chunk_size:
            optimal_chunk_size = max(min_chunk_size, data_size // 2)

        return optimal_chunk_size

    def estimate_processing_time(self, data_size: int, indicators_count: int, use_gpu: bool = False) -> Dict[str, float]:
        """估算处理时间"""
        # 基准性能指标（毫秒 / 数据点）
        cpu_base_time = 0.001  # 1ms per point
        gpu_base_time = 0.0001  # 0.1ms per point

        base_time = gpu_base_time if use_gpu else cpu_base_time

        # 考虑指标数量
        total_time_per_point = base_time * indicators_count

        # 考虑分布式加速
        if self.distributed_available:
            speedup_factor = min(self.config['max_workers'], 8)  # 最大8倍加速
        else:
            speedup_factor = 1

        estimated_time = (data_size * total_time_per_point) / speedup_factor

        return {
            'estimated_time_seconds': estimated_time,
            'estimated_time_minutes': estimated_time / 60,
            'speedup_factor': speedup_factor,
            'data_size': data_size,
            'indicators_count': indicators_count
        }
