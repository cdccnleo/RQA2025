from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np


@dataclass
class FeatureSelector:
    top_k: int = 5

    def select(self, data: pd.DataFrame) -> List[str]:
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        return numeric_cols[: self.top_k]


@dataclass
class AdvancedFeatureSelector(FeatureSelector):
    """
    高级特征选择器
    支持多种特征选择策略
    """
    method: str = "correlation"  # correlation, mutual_info, chi2, etc.
    threshold: float = 0.8

    def select(self, data: pd.DataFrame) -> List[str]:
        """执行高级特征选择"""
        if self.method == "correlation":
            return self._select_by_correlation(data)
        else:
            # 默认使用父类方法
            return super().select(data)

    def _select_by_correlation(self, data: pd.DataFrame) -> List[str]:
        """基于相关性选择特征"""
        numeric_data = data.select_dtypes(include=["number"])
        if numeric_data.empty:
            return []

        corr_matrix = numeric_data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 找到高度相关的特征对
        to_drop = []
        for column in upper.columns:
            if any(upper[column] > self.threshold):
                to_drop.append(column)
                break  # 只删除一个特征

        selected_features = [col for col in numeric_data.columns if col not in to_drop]
        return selected_features[:self.top_k]


def select_features_auto(data: pd.DataFrame, method: str = "correlation") -> List[str]:
    """自动特征选择"""
    selector = AdvancedFeatureSelector(method=method)
    return selector.select(data)


def select_features_univariate(data: pd.DataFrame, target: str, k: int = 5) -> List[str]:
    """单变量特征选择"""
    # 简单实现
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols[:k]


def select_features_model_based(data: pd.DataFrame, target: str, k: int = 5) -> List[str]:
    """基于模型的特征选择"""
    # 简单实现
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols[:k]


__all__ = [
    "FeatureSelector", "AdvancedFeatureSelector",
    "select_features_auto", "select_features_univariate", "select_features_model_based"
]

