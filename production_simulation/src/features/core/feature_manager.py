import logging
"""
特征管理器模块
提供特征工程的整体管理和协调功能
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import time

from .feature_config import FeatureConfig, FeatureType
from .config import DefaultConfigs
from .feature_engineer import FeatureEngineer
from ..processors.feature_processor import FeatureProcessor
from ..selection.feature_selector import FeatureSelector, SelectionMethod
from ..standardization.feature_standardizer import FeatureStandardizer, StandardizationMethod
from .feature_saver import FeatureSaver
from .config_integration import get_config_integration_manager, ConfigScope


logger = logging.getLogger(__name__)


class FeatureManager:

    """特征管理器"""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        初始化特征管理器

        Args:
            config: 特征配置，如果为None则使用默认配置
        """
        # 初始化logger
        self.logger = logger

        # 初始化配置集成管理器
        self.config_manager = get_config_integration_manager()

        # 获取配置
        self.config = config or DefaultConfigs.get_basic_config()

        # 从配置管理器获取处理配置
        processing_config = self.config_manager.get_config(ConfigScope.PROCESSING)
        if processing_config:
            # 更新配置
            if 'max_workers' in processing_config:
                self.config.max_workers = processing_config['max_workers']
            if 'batch_size' in processing_config:
                self.config.batch_size = processing_config['batch_size']
            if 'timeout' in processing_config:
                self.config.timeout = processing_config['timeout']
            if 'feature_selection_method' in processing_config:
                self.config.feature_selection_method = processing_config['feature_selection_method']
            if 'max_features' in processing_config:
                self.config.max_features = processing_config['max_features']
            if 'min_feature_importance' in processing_config:
                self.config.min_feature_importance = processing_config['min_feature_importance']
            if 'standardization_method' in processing_config:
                self.config.standardization_method = processing_config['standardization_method']
            if 'robust_scaling' in processing_config:
                self.config.robust_scaling = processing_config['robust_scaling']

        # 初始化组件
        self.feature_engineer = FeatureEngineer()
        self.feature_processor = FeatureProcessor()
        self.feature_selector = FeatureSelector(config={'max_features': getattr(self.config, 'max_features', None)})
        self.feature_standardizer = FeatureStandardizer(config={'method': getattr(self.config, 'standardization_method', 'zscore')})

        self.feature_saver = FeatureSaver()

        # 缓存
        self._feature_cache = {}
        self._last_cache_cleanup = time.time()

        # 注册配置变更监听器
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

        self.logger.info("特征管理器初始化完成")

    def _on_config_change(self, scope: ConfigScope, key: str, old_value: Any, new_value: Any):
        """配置变更处理"""
        logger.info(f"特征管理器配置变更: {scope.value}.{key} = {old_value} -> {new_value}")

        if scope == ConfigScope.PROCESSING:
            if key == "max_workers":
                self.config.max_workers = new_value
            elif key == "batch_size":
                self.config.batch_size = new_value
            elif key == "timeout":
                self.config.timeout = new_value
            elif key == "feature_selection_method":
                self.config.feature_selection_method = new_value
            elif key == "max_features":
                self.config.max_features = new_value
            elif key == "min_feature_importance":
                self.config.min_feature_importance = new_value
            elif key == "standardization_method":
                self.config.standardization_method = new_value
            elif key == "robust_scaling":
                self.config.robust_scaling = new_value

    def process_features(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],


                         feature_types: Optional[List[FeatureType]] = None) -> pd.DataFrame:
        """
        处理特征

        Args:
            data: 输入数据，可以是DataFrame或包含多个DataFrame的字典
            feature_types: 要处理的特征类型列表，如果为None则使用配置中的类型

        Returns:
            处理后的特征DataFrame
        """
        start_time = time.time()
        logger.info(f"开始处理特征，数据类型: {type(data)}")

        # 确定要处理的特征类型
        if feature_types is None:
            feature_types = self.config.feature_types

        # 处理数据
        if isinstance(data, pd.DataFrame):
            processed_data = self._process_single_dataframe(data, feature_types)
        elif isinstance(data, dict):
            processed_data = self._process_multiple_dataframes(data, feature_types)
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")

        # 特征选择
        if self.config.max_features and not processed_data.empty:
            try:
                # 将字符串方法转换为枚举
                selection_method = SelectionMethod.VARIANCE
                if hasattr(self.config, 'feature_selection_method'):
                    method_str = self.config.feature_selection_method.lower()
                    if method_str == 'correlation':
                        selection_method = SelectionMethod.CORRELATION
                    elif method_str == 'importance':
                        selection_method = SelectionMethod.IMPORTANCE
                    elif method_str == 'mutual_info':
                        selection_method = SelectionMethod.MUTUAL_INFO
                    elif method_str == 'none':
                        selection_method = SelectionMethod.NONE
                
                processed_data = self.feature_selector.select_features(
                    processed_data,
                    method=selection_method,
                    max_features=self.config.max_features,
                    min_importance=getattr(self.config, 'min_feature_importance', 0.01)
                )
                logger.info(f"特征选择完成，保留 {len(processed_data.columns)} 个特征")
            except Exception as e:
                logger.warning(f"特征选择失败: {e}，保留所有特征")

        # 特征标准化
        if not processed_data.empty:
            try:
                # 将字符串方法转换为枚举
                std_method = StandardizationMethod.ZSCORE
                if hasattr(self.config, 'standardization_method'):
                    method_str = self.config.standardization_method.lower()
                    if method_str == 'minmax':
                        std_method = StandardizationMethod.MINMAX
                    elif method_str == 'robust':
                        std_method = StandardizationMethod.ROBUST
                    elif method_str == 'log':
                        std_method = StandardizationMethod.LOG
                    elif method_str == 'none':
                        std_method = StandardizationMethod.NONE
                
                processed_data = self.feature_standardizer.standardize_features(
                    processed_data,
                    method=std_method,
                    robust=getattr(self.config, 'robust_scaling', False)
                )
                logger.info(f"特征标准化完成，方法: {std_method.value}")
            except Exception as e:
                logger.warning(f"特征标准化失败: {e}，保留原始特征")

        processing_time = time.time() - start_time
        logger.info(f"特征处理完成，耗时: {processing_time:.2f}秒，特征数量: {len(processed_data.columns)}")

        return processed_data

    def _process_single_dataframe(self, data: pd.DataFrame,


                                  feature_types: List[FeatureType]) -> pd.DataFrame:
        """处理单个DataFrame"""
        processed_features = []

        for feature_type in feature_types:
            if feature_type == FeatureType.TECHNICAL:
                features = self.feature_engineer.generate_technical_features(data)
                processed_features.append(features)
            elif feature_type == FeatureType.SENTIMENT:
                # 检查是否启用了情感分析（通过检查情感类型列表是否为空）
                if hasattr(self.config, 'sentiment_types') and self.config.sentiment_types:
                    features = self.feature_engineer.generate_sentiment_features(data)
                    processed_features.append(features)
            elif feature_type == FeatureType.ORDERBOOK:
                # 暂时跳过orderbook特征，因为FeatureEngineer没有这个方法
                logger.warning("Orderbook特征处理暂未实现")
                continue
            elif feature_type == FeatureType.FUNDAMENTAL:
                # 暂时跳过fundamental特征，因为FeatureEngineer没有这个方法
                logger.warning("Fundamental特征处理暂未实现")
                continue
            else:
                logger.warning(f"未支持的特征类型: {feature_type}")

        # 合并所有特征
        if processed_features:
            result = pd.concat(processed_features, axis=1)
            return result
        else:
            return pd.DataFrame()

    def _process_multiple_dataframes(self, data_dict: Dict[str, pd.DataFrame],


                                     feature_types: List[FeatureType]) -> pd.DataFrame:
        """处理多个DataFrame"""
        processed_features = []

        for name, data in data_dict.items():
            logger.info(f"处理数据集: {name}")
            features = self._process_single_dataframe(data, feature_types)
            if not features.empty:
                # 添加数据集标识
                features.columns = [f"{name}_{col}" for col in features.columns]
                processed_features.append(features)

        # 合并所有特征
        if processed_features:
            result = pd.concat(processed_features, axis=1)
            return result
        else:
            return pd.DataFrame()

    def process_streaming(self, data_stream: Any) -> pd.DataFrame:
        """
        流式处理特征

        Args:
            data_stream: 数据流

        Returns:
            处理后的特征
        """
        self.logger.info("开始流式特征处理")

        # 这里应该实现流式处理逻辑
        # 目前返回空DataFrame作为占位符
        return pd.DataFrame()

    def save_features(self, features: pd.DataFrame,


                      output_path: Union[str, Path],
                      format: str = "parquet") -> None:
        """
        保存特征

        Args:
            features: 特征DataFrame
            output_path: 输出路径
            format: 保存格式
        """
        self.feature_saver.save_features(features, output_path, format)
        self.logger.info(f"特征已保存到: {output_path}")

    def load_features(self, input_path: Union[str, Path],


                      format: str = "parquet") -> pd.DataFrame:
        """
        加载特征

        Args:
            input_path: 输入路径
            format: 文件格式

        Returns:
            加载的特征DataFrame
        """
        return self.feature_saver.load_features(input_path, format)

    def get_feature_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        获取特征信息

        Args:
            features: 特征DataFrame

        Returns:
            特征信息字典
        """
        info = {
            'feature_count': len(features.columns),
            'sample_count': len(features),
            'memory_usage': features.memory_usage(deep=True).sum(),
            'feature_names': list(features.columns),
            'data_types': features.dtypes.to_dict(),
            'missing_values': features.isnull().sum().to_dict(),
            'numeric_features': features.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': features.select_dtypes(include=['object']).columns.tolist()
        }

        return info

    def validate_features(self, features: pd.DataFrame) -> Dict[str, bool]:
        """
        验证特征

        Args:
            features: 特征DataFrame

        Returns:
            验证结果字典
        """
        validation_results = {
            'has_features': len(features.columns) > 0,
            'has_samples': len(features) > 0,
            'no_duplicate_columns': len(features.columns) == len(set(features.columns)),
            'no_all_null_columns': not features.isnull().all().any(),
            'no_infinite_values': not np.isinf(features.select_dtypes(include=[np.number])).any().any()
        }

        return validation_results

    def cleanup_cache(self) -> None:
        """清理缓存"""
        current_time = time.time()
        if current_time - self._last_cache_cleanup > self.config.cache_ttl:
            self._feature_cache.clear()
            self._last_cache_cleanup = current_time
            self.logger.info("特征缓存已清理")

    def update_config(self, new_config: FeatureConfig) -> None:
        """
        更新配置

        Args:
            new_config: 新配置
        """
        self.config = new_config
        self.logger.info("特征管理器配置已更新")

    def get_status(self) -> Dict[str, Any]:
        """
        获取管理器状态

        Returns:
            状态信息字典
        """
        return {
            'config': self.config.to_dict(),
            'cache_size': len(self._feature_cache),
            'last_cache_cleanup': self._last_cache_cleanup,
            'components_initialized': all([
                self.feature_engineer is not None,
                self.feature_processor is not None,
                self.feature_selector is not None,
                self.feature_standardizer is not None,
                self.feature_saver is not None
            ])
        }

    def run(self, data_source: str) -> Dict[str, Any]:
        """
        运行特征处理流程

        Args:
            data_source: 数据源标识

        Returns:
            处理结果字典
        """
        try:
            self.logger.info(f"开始运行特征处理流程，数据源: {data_source}")

            # 生成技术特征
            technical_features = self.feature_engineer.generate_technical_features(data_source)

            # 生成情感特征
            sentiment_features = self.feature_engineer.generate_sentiment_features(data_source)

            # 合并特征
            merged_features = self.feature_engineer.merge_features(
                [technical_features, sentiment_features])

            # 保存元数据
            self.feature_engineer.save_metadata(merged_features, data_source)

            # 加载元数据（用于验证）
            metadata = self.feature_engineer.load_metadata(data_source)

            return {
                'result': merged_features,
                'metadata': metadata,
                'status': 'success',
                'data_source': data_source
            }

        except Exception as e:
            self.logger.error(f"特征处理流程失败: {e}")
            return {
                'result': None,
                'error': str(e),
                'status': 'error',
                'data_source': data_source
            }
