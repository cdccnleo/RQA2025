import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择器

负责选择重要特征。
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
except ImportError:
    get_unified_logger = logging.getLogger

try:
    from ..core.feature_config import FeatureConfig
except ImportError:
    from ..core.config import FeatureConfig


logger = get_unified_logger('__name__')


class FeatureSelector:

    """特征选择器"""

    def __init__(self):
        """初始化特征选择器"""
        self.logger = logging.getLogger(__name__)

    def select_features(self, features: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        选择特征

        Args:
            features: 输入特征
            config: 配置

        Returns:
            选择后的特征
        """
        if features is None or features.empty:
            return pd.DataFrame()

        try:
            # 复制数据避免修改原始数据
            selected_features = features.copy()

            # 获取数值列
            numeric_columns = selected_features.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) == 0:
                self.logger.warning("没有数值列可以选择")
                return selected_features

            # 移除常数列
            constant_columns = []
            for col in numeric_columns:
                if selected_features[col].std() == 0:
                    constant_columns.append(col)

            if constant_columns:
                selected_features = selected_features.drop(columns=constant_columns)
                self.logger.info(f"移除了 {len(constant_columns)} 个常数列")

            # 移除高度相关的特征
            if config and hasattr(config, 'max_features') and config.max_features:
                selected_features = self._remove_correlated_features(
                    selected_features, config.max_features)

            self.logger.info(f"特征选择完成，选择了 {len(selected_features.columns)} 个特征")
            return selected_features

        except Exception as e:
            self.logger.error(f"特征选择失败: {e}")
            return features

    def _remove_correlated_features(self, features: pd.DataFrame, max_features: int) -> pd.DataFrame:
        """
        移除高度相关的特征

        Args:
            features: 特征数据
            max_features: 最大特征数

        Returns:
            处理后的特征
        """
        try:
            numeric_columns = features.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) <= max_features:
                return features

            # 计算相关性矩阵
            corr_matrix = features[numeric_columns].corr().abs()

            # 上三角矩阵
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # 找到高度相关的特征对
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

            # 保留最重要的特征（这里简单地保留前max_features个）
            remaining_features = [col for col in numeric_columns if col not in to_drop]
            if len(remaining_features) > max_features:
                remaining_features = remaining_features[:max_features]

            return features[remaining_features]

        except Exception as e:
            self.logger.error(f"移除相关特征失败: {e}")
            return features
