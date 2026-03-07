#!/usr/bin/env python3
"""
特征选择器
提供特征选择和降维功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
try:
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SelectKBest = None
    f_regression = None
    mutual_info_regression = None
    PCA = None
    StandardScaler = None

logger = logging.getLogger(__name__)


class FeatureSelector:

    """特征选择器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        # correlation, mutual_info, kbest, pca
        self.method = self.config.get('method', 'correlation')
        self.k_features = self.config.get('k_features', 10)

    def select_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None,


                        method: Optional[str] = None) -> Dict[str, Any]:
        """
        选择最优特征

        Args:
            X: 特征数据
            y: 目标变量 (某些方法需要)
            method: 选择方法

        Returns:
            包含选择结果的字典
        """
        try:
            if method:
                self.method = method

            if self.method == 'correlation':
                return self._select_by_correlation(X, y)
            elif self.method == 'mutual_info':
                return self._select_by_mutual_info(X, y)
            elif self.method == 'kbest':
                return self._select_by_kbest(X, y)
            elif self.method == 'pca':
                return self._select_by_pca(X)
            else:
                raise ValueError(f"不支持的特征选择方法: {self.method}")

        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': self.method,
                'error': str(e)
            }

    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """基于相关性选择特征"""
        try:
            if y is None:
                raise ValueError("相关性选择需要目标变量")

            # 计算相关性
            correlations = {}
            for column in X.columns:
                if X[column].dtype in ['float64', 'int64']:
                    corr = abs(X[column].corr(y))
                    if not np.isnan(corr):
                        correlations[column] = corr

            # 按相关性排序
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, _ in sorted_features[:self.k_features]]

            return {
                'selected_features': selected_features,
                'scores': correlations,
                'method': 'correlation',
                'feature_count': len(selected_features)
            }

        except Exception as e:
            logger.error(f"相关性特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'correlation',
                'error': str(e)
            }

    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """基于互信息选择特征"""
        try:
            if y is None:
                raise ValueError("互信息选择需要目标变量")

            # 标准化数据
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # 计算互信息
            mi_scores = mutual_info_regression(X_scaled, y)

            # 创建特征 - 分数映射
            feature_scores = dict(zip(X.columns, mi_scores))

            # 按互信息排序
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, _ in sorted_features[:self.k_features]]

            return {
                'selected_features': selected_features,
                'scores': feature_scores,
                'method': 'mutual_info',
                'feature_count': len(selected_features)
            }

        except Exception as e:
            logger.error(f"互信息特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'mutual_info',
                'error': str(e)
            }

    def _select_by_kbest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """基于SelectKBest选择特征"""
        try:
            if y is None:
                raise ValueError("KBest选择需要目标变量")

            # 使用F检验选择特征
            selector = SelectKBest(score_func=f_regression, k=self.k_features)
            X_selected = selector.fit_transform(X, y)

            # 获取选择的特征名称
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()

            # 获取特征分数
            feature_scores = dict(zip(X.columns, selector.scores_))

            return {
                'selected_features': selected_features,
                'scores': feature_scores,
                'method': 'kbest',
                'feature_count': len(selected_features)
            }

        except Exception as e:
            logger.error(f"KBest特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'kbest',
                'error': str(e)
            }

    def _select_by_pca(self, X: pd.DataFrame) -> Dict[str, Any]:
        """基于PCA进行特征降维"""
        try:
            # 标准化数据
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 执行PCA
            pca = PCA(n_components=self.k_features)
            X_pca = pca.fit_transform(X_scaled)

            # 创建新的特征名称
            selected_features = [f'pca_{i + 1}' for i in range(self.k_features)]

            # 计算解释方差比
            explained_variance_ratio = pca.explained_variance_ratio_
            feature_scores = dict(zip(selected_features, explained_variance_ratio))

            return {
                'selected_features': selected_features,
                'scores': feature_scores,
                'method': 'pca',
                'feature_count': len(selected_features),
                'total_explained_variance': np.sum(explained_variance_ratio),
                'pca_components': pca.components_
            }

        except Exception as e:
            logger.error(f"PCA特征选择失败: {e}")
            return {
                'selected_features': X.columns.tolist(),
                'scores': {},
                'method': 'pca',
                'error': str(e)
            }

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            result = self.select_features(X, y, method='correlation')

            if 'scores' in result:
                return result['scores']
            else:
                return {}

        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}
