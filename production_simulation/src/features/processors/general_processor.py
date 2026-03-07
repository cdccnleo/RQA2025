import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征处理器

from src.infrastructure.logging.core.unified_logger import get_unified_logger
负责处理特征数据。
"""

import pandas as pd
from typing import Optional, Dict, Any, List

# from .feature_config import FeatureConfig
from .base_processor import BaseFeatureProcessor, ProcessorConfig


logger = logging.getLogger(__name__)


class FeatureProcessor(BaseFeatureProcessor):

    """特征处理器"""

    def __init__(self, config: Optional[ProcessorConfig] = None):

        if config is None:
            config = ProcessorConfig(
                processor_type="general",
                feature_params={
                    "handle_missing_values": True,
                    "remove_duplicates": True
                }
            )

        super().__init__(config)
        # 初始化logger
        self.logger = logger

    def process_features(self, features: pd.DataFrame, config: Optional[Any] = None) -> pd.DataFrame:
        """
        处理特征

        Args:
            features: 输入特征
            config: 配置

        Returns:
            处理后的特征
        """
        if features is None or features.empty:
            return pd.DataFrame()

        try:
            # 复制数据避免修改原始数据
            processed_features = features.copy()

            # 基础处理：移除重复行
            processed_features = processed_features.drop_duplicates()

            # 处理缺失值
            if config and hasattr(config, 'handle_missing_values') and config.handle_missing_values:
                processed_features = self._handle_missing_values(processed_features)

            self.logger.info(f"特征处理完成，处理了 {len(processed_features)} 行数据")
            return processed_features

        except Exception as e:
            self.logger.error(f"特征处理失败: {e}")
            return features

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            features: 特征数据

        Returns:
            处理后的特征
        """
        try:
            result = features.copy()

            # 对于数值列，用中位数填充
            numeric_columns = result.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if result[col].isna().any():
                    result[col] = result[col].fillna(result[col].median())

            # 对于分类列，用众数填充
            categorical_columns = result.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if result[col].isna().any():
                    result[col] = result[col].fillna(
                        result[col].mode().iloc[0] if not result[col].mode().empty else 'Unknown')

            return result

        except Exception as e:
            self.logger.error(f"缺失值处理失败: {e}")
            return features

    def _compute_feature(self, data: pd.DataFrame, feature_name: str,


                         params: Dict[str, Any]) -> pd.Series:
        """计算单个特征"""
        # 通用处理器不计算新特征，只处理现有特征
        if feature_name in data.columns:
            return data[feature_name]
        else:
            return pd.Series(index=data.index, dtype=float)

    def _get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """获取特征元数据"""
        return {
            "name": feature_name,
            "type": "general_feature",
            "description": f"通用特征: {feature_name}",
            "parameters": self.config.feature_params
        }

    def _get_available_features(self) -> List[str]:
        """获取可用特征列表"""
        # 通用处理器处理所有现有特征
        return []
