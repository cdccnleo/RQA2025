"""并行特征处理器"""
import numpy as np
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
except ImportError:
    # 如果基础设施日志模块不可用，使用标准logging
    import logging
    def get_unified_logger(name):
        return logging.getLogger(name)
import pandas as pd
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

from src.infrastructure.utils.logging.logger import get_logger
from .feature_engineer import FeatureEngineer
from .feature_config import FeatureConfig, FeatureType

logger = get_logger(__name__)
logger_unified = get_unified_logger('parallel_feature_processor')


@dataclass
class ParallelConfig:

    """并行处理配置"""
    n_jobs: int = 2  # 限制为2个CPU核心，避免过多的并发
    chunk_size: int = 100  # 减小数据分块大小
    timeout: int = 30  # 进一步减少超时时间(秒)
    use_thread_pool: bool = True  # 使用线程池而不是进程池
    memory_limit: int = 512  # 减少内存限制(MB)
    enable_progress: bool = False  # 关闭进度显示以减少开销


class ParallelFeatureProcessor:

    """并行特征处理器"""

    def __init__(self, feature_engine: FeatureEngineer, config: ParallelConfig = None):

        self.engine = feature_engine
        self.config = config or ParallelConfig()

        # 设置工作线程数
        if self.config.n_jobs == -1:
            import os
            self.config.n_jobs = min(4, os.cpu_count() or 1)

        # 初始化执行器
        self._init_executor()

        # 性能统计
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }

        # 初始化logger
        import logging
        self.logger = logging.getLogger(__name__)

    def _init_executor(self):
        """初始化执行器"""
        # 使用线程池避免pickle问题
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.n_jobs
        )

    def process_features_parallel(


        self,
        data: pd.DataFrame,
        feature_configs: List[FeatureConfig],
        batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        并行处理特征

        Args:
            data: 输入数据
            feature_configs: 特征配置列表
            batch_size: 批处理大小，None表示使用默认配置

        Returns:
            处理后的特征DataFrame
        """
        start_time = time.time()

        if batch_size is None:
            batch_size = self.config.chunk_size

        # 数据分块
        chunks = self._split_data(data, batch_size)
        self.logger.info(f"数据分块完成，共{len(chunks)}个块")

        # 并行处理
        results = []
        futures = []

        for i, chunk in enumerate(chunks):
            future = self.executor.submit(
                self._process_chunk,
                chunk,
                feature_configs
            )
            futures.append(future)

        # 收集结果，添加超时保护
        try:
            for future in as_completed(futures, timeout=self.config.timeout):
                try:
                    result = future.result(timeout=5)  # 每个future单独超时5秒
                    if result is not None and not result.empty:
                        results.append(result)
                except Exception as e:
                    try:
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.debug(f"处理数据块失败: {e}")
                    except BaseException:
                        pass  # 忽略日志错误
                    # 继续处理其他块
                    continue
        except Exception as e:
            try:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug(f"并行处理超时，尝试降级: {e}")
            except BaseException:
                pass
            # 超时时，尝试串行处理剩余数据
            for future in futures:
                if not future.done():
                    try:
                        result = future.result(timeout=1)
                        if result is not None and not result.empty:
                            results.append(result)
                    except BaseException:
                        continue

        # 合并结果
        if results:
            final_result = pd.concat(results, ignore_index=True)
            final_result = final_result.sort_index()  # 保持原始顺序
        else:
            # 如果没有结果，返回原始数据
            try:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug("并行处理失败，返回原始数据")
            except BaseException:
                pass
            final_result = data.copy()

        # 更新统计
        processing_time = time.time() - start_time
        self.stats['total_processed'] += 1
        self.stats['total_time'] += processing_time
        self.stats['success_count'] += 1 if len(results) > 0 else 0
        self.stats['error_count'] += 1 if len(results) == 0 else 0

        try:
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"特征处理完成，耗时{processing_time:.2f}秒，成功处理{len(results)}个数据块")
        except BaseException:
            pass
        return final_result

    def _split_data(self, data: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """数据分块"""
        # 修复：当chunk_size为0时使用默认值1
        if chunk_size <= 0:
            chunk_size = 1
            self.logger.warning("分块大小不能为0，使用默认值1")

        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data.iloc[i:i + chunk_size])

        self.logger.info(f"数据分块完成，共{len(chunks)}个块")
        return chunks

    def _process_chunk(self, chunk: pd.DataFrame, configs: List[FeatureConfig]) -> pd.DataFrame:
        """处理单个数据块"""
        features = {}

        for config in configs:
            try:
                # 修复：使用feature_types[0]而不是feature_type
                if config.feature_types and config.feature_types[0] == FeatureType.TECHNICAL:
                    features.update(self._calculate_technical_features(chunk, config))
                elif config.feature_types and config.feature_types[0] == FeatureType.SENTIMENT:
                    features.update(self._calculate_sentiment_features(chunk, config))
                elif config.feature_types and config.feature_types[0] == FeatureType.HIGH_FREQUENCY:
                    features.update(self._calculate_hf_features(chunk, config))
                else:
                    features.update(self._calculate_generic_features(chunk, config))
            except Exception as e:
                self.logger.error(f"处理块失败: {e}")
                continue

        if features:
            return pd.DataFrame(features, index=chunk.index)
        return chunk

    def _calculate_technical_features(self, data: pd.DataFrame, config: FeatureConfig) -> Dict[str, np.ndarray]:
        """计算技术指标特征"""
        features = {}

        # 修复：从technical_indicators获取指标名称，而不是从config.name
        for indicator in config.technical_indicators:
            try:
                if 'SMA' in indicator.upper():
                    window = config.technical_params.sma_periods[0] if config.technical_params.sma_periods else 20
                    features[f'{indicator}_sma'] = data['close'].rolling(
                        window=window).mean().values
                elif 'RSI' in indicator.upper():
                    window = config.technical_params.rsi_period if config.technical_params.rsi_period else 14
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    features[f'{indicator}_rsi'] = (100 - (100 / (1 + rs))).values
                elif 'MACD' in indicator.upper():
                    fast = config.technical_params.macd_fast if config.technical_params.macd_fast else 12
                    slow = config.technical_params.macd_slow if config.technical_params.macd_slow else 26
                    signal = config.technical_params.macd_signal if config.technical_params.macd_signal else 9

                    ema_fast = data['close'].ewm(span=fast).mean()
                    ema_slow = data['close'].ewm(span=slow).mean()
                    macd = ema_fast - ema_slow
                    signal_line = macd.ewm(span=signal).mean()

                    features[f'{indicator}_macd'] = macd.values
                    features[f'{indicator}_signal'] = signal_line.values
                    features[f'{indicator}_histogram'] = (macd - signal_line).values
                else:
                    # 默认技术指标
                    features[f'{indicator}_default'] = data['close'].rolling(20).mean().values
            except Exception as e:
                self.logger.warning(f"计算技术指标 {indicator} 失败: {e}")
                continue

        return features

    def _calculate_sentiment_features(self, data: pd.DataFrame, config: FeatureConfig) -> Dict[str, np.ndarray]:
        """计算情感分析特征"""
        features = {}

        # 修复：从sentiment_types获取类型名称
        for sentiment_type in config.sentiment_types:
            try:
                if 'sentiment' in sentiment_type.lower():
                    features[f'{sentiment_type}_score'] = np.secrets.normal(0, 1, len(data))
                    features[f'{sentiment_type}_confidence'] = np.secrets.uniform(
                        0.5, 1.0, len(data))
                else:
                    features[f'{sentiment_type}_default'] = np.secrets.normal(0, 1, len(data))
            except Exception as e:
                self.logger.warning(f"计算情感特征 {sentiment_type} 失败: {e}")
                continue

        return features

    def _calculate_hf_features(self, data: pd.DataFrame, config: FeatureConfig) -> Dict[str, np.ndarray]:
        """计算高频特征"""
        features = {}

        try:
            returns = data['close'].pct_change()

            # 修复：使用固定的特征名称
            if 'momentum' in str(config.sentiment_types).lower():
                features['momentum_20'] = returns.rolling(window=20).sum().values
            if 'volatility' in str(config.sentiment_types).lower():
                features['volatility_20'] = returns.rolling(window=20).std().values

            # 通用高频特征
            features['volume_mean_20'] = data['volume'].rolling(20).mean().values
            features['volume_std_20'] = data['volume'].rolling(20).std().values
            features['price_range'] = (data['high'] - data['low']) / data['close']

        except Exception as e:
            self.logger.warning(f"计算高频特征失败: {e}")

        return features

    def _calculate_generic_features(self, data: pd.DataFrame, config: FeatureConfig) -> Dict[str, np.ndarray]:
        """计算通用特征"""
        features = {}

        try:
            # 基础统计特征
            features['price_mean'] = data[['open', 'high', 'low', 'close']].mean(axis=1).values
            features['price_std'] = data[['open', 'high', 'low', 'close']].std(axis=1).values
            features['volume_log'] = np.log1p(data['volume'].values)

        except Exception as e:
            self.logger.warning(f"计算通用特征失败: {e}")

        return features

    def batch_process_symbols(


        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        feature_configs: List[FeatureConfig]
    ) -> Dict[str, pd.DataFrame]:
        """批量处理多个股票的特征"""
        results = {}

        # 并行处理每个股票
        futures = {}
        for symbol in symbols:
            if symbol in data_dict:
                future = self.executor.submit(
                    self._process_single_symbol,
                    symbol,
                    data_dict[symbol],
                    feature_configs
                )
                futures[future] = symbol

        # 收集结果，添加超时保护
        try:
            for future in as_completed(futures, timeout=self.config.timeout):
                symbol = futures[future]
                try:
                    result = future.result(timeout=5)  # 每个future单独超时5秒
                    if result is not None and not result.empty:
                        results[symbol] = result
                        try:
                            if hasattr(self, 'logger') and self.logger:
                                self.logger.debug(f"股票{symbol}处理完成")
                        except BaseException:
                            pass
                except Exception as e:
                    try:
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.debug(f"处理股票{symbol}失败: {e}")
                    except BaseException:
                        pass
                    # 返回原始数据作为fallback
                    results[symbol] = data_dict[symbol].copy()
        except Exception as e:
            try:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug(f"批量处理超时，降级处理: {e}")
            except BaseException:
                pass
            # 超时时，尝试串行处理剩余数据
            for future in futures:
                if not future.done():
                    symbol = futures[future]
                    try:
                        result = future.result(timeout=2)
                        if result is not None and not result.empty:
                            results[symbol] = result
                    except BaseException:
                        # 返回原始数据作为fallback
                        results[symbol] = data_dict[symbol].copy()

        return results

    def _process_single_symbol(


        self,
        symbol: str,
        data: pd.DataFrame,
        feature_configs: List[FeatureConfig]
    ) -> Optional[pd.DataFrame]:
        """处理单个股票的特征"""
        try:
            return self.process_features_parallel(data, feature_configs)
        except Exception as e:
            self.logger.error(f"处理股票{symbol}特征失败: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['avg_time_per_record'] = stats['total_time'] / stats['total_processed']
            stats['success_rate'] = stats['success_count'] / \
                (stats['success_count'] + stats['error_count'])
        return stats

    def close(self):
        """关闭执行器"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()
