import logging
"""优化的特征管理器"""
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import threading

from .feature_engineer import FeatureEngineer
from .config import FeatureRegistrationConfig, FeatureType
from .parallel_feature_processor import ParallelFeatureProcessor, ParallelConfig
from ..processors.quality_assessor import FeatureQualityAssessor, AssessmentConfig
from .feature_store import FeatureStore, StoreConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizedManagerConfig:

    """优化管理器配置"""
    # 并行处理配置
    parallel_config: ParallelConfig = None
    # 质量评估配置
    assessment_config: AssessmentConfig = None
    # 存储配置
    store_config: StoreConfig = None
    # 性能配置
    enable_parallel: bool = True
    enable_quality_assessment: bool = True
    enable_caching: bool = True
    enable_auto_cleanup: bool = True
    max_workers: int = 4


class OptimizedFeatureManager:

    """优化的特征管理器"""

    def __init__(self, config: OptimizedManagerConfig = None):

        self.config = config or OptimizedManagerConfig()

        # 初始化组件
        self.feature_engine = FeatureEngineer()

        # 并行处理器
        if self.config.enable_parallel:
            parallel_config = self.config.parallel_config or ParallelConfig()
            self.parallel_processor = ParallelFeatureProcessor(
                self.feature_engine, parallel_config
            )
        else:
            self.parallel_processor = None

        # 质量评估器
        if self.config.enable_quality_assessment:
            assessment_config = self.config.assessment_config or AssessmentConfig()
            self.quality_assessor = FeatureQualityAssessor(assessment_config)
        else:
            self.quality_assessor = None

        # 特征存储
        if self.config.enable_caching:
            store_config = self.config.store_config or StoreConfig()
            self.feature_store = FeatureStore(store_config)
        else:
            self.feature_store = None

        # 线程锁
        self._lock = threading.Lock()

        # 性能统计
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_processed': 0,
            'quality_assessed': 0,
            'total_time': 0.0
        }

    def generate_features_optimized(


        self,
        data: pd.DataFrame,
        feature_configs: List[FeatureRegistrationConfig],
        target: Optional[pd.Series] = None,
        use_cache: bool = True,
        assess_quality: bool = True,
        min_quality_score: float = 0.6
    ) -> pd.DataFrame:
        """
        优化的特征生成

        Args:
            data: 输入数据
            feature_configs: 特征配置列表
            target: 目标变量（用于质量评估）
            use_cache: 是否使用缓存
            assess_quality: 是否进行质量评估
            min_quality_score: 最小质量评分

        Returns:
            生成的特征DataFrame
        """
        start_time = time.time()

        try:
            # 1. 尝试从缓存加载
            if use_cache and self.feature_store:
                cached_features = self._load_from_cache(data, feature_configs)
                if cached_features is not None:
                    self.stats['cache_hits'] += 1
                    logger.info("从缓存加载特征成功")
                    return cached_features

            self.stats['cache_misses'] += 1

            # 2. 并行生成特征
            if self.parallel_processor and self.config.enable_parallel:
                features = self.parallel_processor.process_features_parallel(
                    data, feature_configs
                )
                self.stats['parallel_processed'] += 1
            else:
                # 串行生成特征
                features = self._generate_features_serial(data, feature_configs)

            # 3. 质量评估和过滤
            if assess_quality and self.quality_assessor and target is not None:
                features = self._assess_and_filter_features(
                    features, target, min_quality_score
                )
                self.stats['quality_assessed'] += 1

            # 4. 存储到缓存
            if use_cache and self.feature_store:
                self._save_to_cache(features, feature_configs, data)

            # 更新统计
            self.stats['total_processed'] += 1
            self.stats['total_time'] += time.time() - start_time

            logger.info(f"特征生成完成，耗时{time.time() - start_time:.2f}秒")
            return features

        except Exception as e:
            logger.error(f"特征生成失败: {e}")
            return pd.DataFrame()

    def batch_generate_features(


        self,
        data_dict: Dict[str, pd.DataFrame],
        feature_configs: List[FeatureRegistrationConfig],
        targets: Optional[Dict[str, pd.Series]] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量生成特征

        Args:
            data_dict: 股票数据字典 {symbol: data}
            feature_configs: 特征配置列表
            targets: 目标变量字典
            use_cache: 是否使用缓存

        Returns:
            特征字典 {symbol: features}
        """
        results = {}

        # 简化处理逻辑，避免嵌套并行处理
        for symbol, data in data_dict.items():
            try:
                target = targets.get(symbol) if targets else None
                # 直接调用单股票特征生成，避免嵌套并行
                features = self.generate_features_optimized(
                    data, feature_configs, target, use_cache, assess_quality=False
                )
                if not features.empty:
                    results[symbol] = features
                logger.info(f"股票{symbol}特征生成完成")
            except Exception as e:
                logger.error(f"处理股票{symbol}失败: {e}")
                # 返回原始数据作为fallback
                results[symbol] = data.copy()

        return results

    def get_feature_quality_report(self) -> Dict[str, Any]:
        """获取特征质量报告"""
        if self.quality_assessor:
            return self.quality_assessor.get_quality_report()
        return {}

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if self.feature_store:
            return self.feature_store.get_store_stats()
        return {}

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.stats.copy()

        # 计算平均处理时间
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = stats['total_time'] / stats['total_processed']
        else:
            stats['avg_processing_time'] = 0.0

        # 计算缓存命中率
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0

        return stats

    def cleanup_cache(self) -> int:
        """清理缓存"""
        if self.feature_store:
            return self.feature_store.cleanup_expired()
        return 0

    def _load_from_cache(


        self,
        data: pd.DataFrame,
        feature_configs: List[FeatureRegistrationConfig]
    ) -> Optional[pd.DataFrame]:
        """从缓存加载特征"""
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(data, feature_configs)

            # 尝试加载
            result = self.feature_store.load_feature(cache_key)
            if result:
                cached_data, metadata = result
                # 验证数据完整性
                if self._validate_cached_data(cached_data, data):
                    logger.info(f"从缓存加载特征成功: {cache_key}")
                    return cached_data

            return None
        except Exception as e:
            logger.error(f"从缓存加载失败: {e}")
            return None

    def _save_to_cache(


        self,
        features: pd.DataFrame,
        feature_configs: List[FeatureRegistrationConfig],
        original_data: pd.DataFrame = None
    ):
        """保存特征到缓存"""
        try:
            # 使用原始数据生成缓存键
            if original_data is not None:
                cache_key = self._generate_cache_key(original_data, feature_configs)
            else:
                cache_key = self._generate_cache_key(features, feature_configs)

            # 创建特征配置
            config = FeatureRegistrationConfig(
                name=cache_key,
                feature_type=FeatureType.TECHNICAL,
                params={'cached': True},
                dependencies=[]
            )

            # 保存到存储
            success = self.feature_store.store_feature(
                cache_key, features, config,
                description="Cached features",
                tags=['cached', 'optimized']
            )

            if success:
                logger.info(f"特征已保存到缓存: {cache_key}")
        except Exception as e:
            logger.error(f"保存到缓存失败: {e}")

    def _generate_cache_key(


        self,
        data: pd.DataFrame,
        feature_configs: List[FeatureRegistrationConfig]
    ) -> str:
        """生成缓存键"""
        import hashlib
        import json

        # 创建数据摘要
        data_summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
        }

        # 创建配置摘要
        config_summary = []
        for config in feature_configs:
            config_summary.append({
                'name': config.name,
                'type': config.feature_type.value,
                'params': config.params,
                'dependencies': config.dependencies
            })

        # 生成哈希
        content = json.dumps([data_summary, config_summary], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _validate_cached_data(


        self,
        cached_data: pd.DataFrame,
        original_data: pd.DataFrame
    ) -> bool:
        """验证缓存数据的有效性"""
        try:
            # 检查数据长度
            if len(cached_data) != len(original_data):
                logger.warning(f"缓存数据长度不匹配: {len(cached_data)} vs {len(original_data)}")
                return False

            # 检查索引
            if not cached_data.index.equals(original_data.index):
                logger.warning("缓存数据索引不匹配")
                return False

            # 检查是否包含原始列
            original_cols = set(original_data.columns)
            cached_cols = set(cached_data.columns)
            if not original_cols.issubset(cached_cols):
                logger.warning(f"缓存数据缺少原始列: {original_cols - cached_cols}")
                return False

            return True
        except Exception as e:
            logger.error(f"验证缓存数据失败: {e}")
            return False

    def _generate_features_serial(


        self,
        data: pd.DataFrame,
        feature_configs: List[FeatureRegistrationConfig]
    ) -> pd.DataFrame:
        """串行生成特征"""
        result = data.copy()

        for config in feature_configs:
            try:
                if config.feature_type == FeatureType.TECHNICAL:
                    features = self.feature_engine.generate_technical_features(
                        result, [config.name], config.params
                    )
                elif config.feature_type == FeatureType.SENTIMENT:
                    features = self.feature_engine.generate_sentiment_features(
                        result, text_col="content", date_col="date"
                    )
                else:
                    # 通用特征生成
                    features = self._generate_generic_features(result, config)

                # 合并特征
                for col in features.columns:
                    if col not in result.columns:
                        result[col] = features[col]

            except Exception as e:
                logger.error(f"生成特征 {config.name} 失败: {e}")

        return result

    def _generate_generic_features(


        self,
        data: pd.DataFrame,
        config: FeatureRegistrationConfig
    ) -> pd.DataFrame:
        """生成通用特征"""
        features = pd.DataFrame(index=data.index)

        try:
            if 'volume' in data.columns:
                features[f'{config.name}_volume_mean'] = data['volume'].rolling(20).mean()
                features[f'{config.name}_volume_std'] = data['volume'].rolling(20).std()

            if 'close' in data.columns:
                features[f'{config.name}_price_range'] = (
                    data['high'] - data['low']) / data['close']
                features[f'{config.name}_returns'] = data['close'].pct_change()

        except Exception as e:
            logger.error(f"生成通用特征失败: {e}")

        return features

    def _assess_and_filter_features(


        self,
        features: pd.DataFrame,
        target: pd.Series,
        min_quality_score: float
    ) -> pd.DataFrame:
        """评估和过滤特征"""
        try:
            # 分离原始列和新生成的特征列
            original_columns = ['open', 'high', 'low', 'close', 'volume']
            original_features = features[original_columns].copy()
            new_features = features.drop(columns=original_columns, errors='ignore')

            # 只对新生成的特征进行质量评估
            if not new_features.empty:
                quality_metrics = self.quality_assessor.assess_feature_quality(
                    new_features, target
                )

                # 过滤低质量特征
                filtered_new_features = self.quality_assessor.filter_features(
                    new_features, min_quality_score
                )

                logger.info(
                    f"质量评估完成，从{len(new_features.columns)}个新特征中保留{len(filtered_new_features.columns)}个")

                # 合并原始列和过滤后的新特征
                result = pd.concat([original_features, filtered_new_features], axis=1)
            else:
                # 如果没有新特征，直接返回原始特征
                result = original_features
                logger.info("没有新生成的特征，保留所有原始列")

            return result

        except Exception as e:
            logger.error(f"特征质量评估失败: {e}")
            return features

    def close(self):
        """关闭管理器"""
        if self.parallel_processor:
            self.parallel_processor.close()

        if self.feature_store:
            self.feature_store.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()
