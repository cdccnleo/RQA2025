import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征处理器

from src.infrastructure.logging.core.unified_logger import get_unified_logger
支持更多特征类型，包括时间序列特征、统计特征、机器学习特征等。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import warnings

from ..base_processor import BaseFeatureProcessor, ProcessorConfig


logger = logging.getLogger(__name__)


class TimeSeriesFeatureProcessor(BaseFeatureProcessor):

    """时间序列特征处理器"""

    def __init__(self, config: ProcessorConfig):

        super().__init__(config)

    def _compute_feature(self, data: pd.DataFrame, feature_name: str, params: Dict[str, Any]) -> pd.Series:
        """计算时间序列特征"""
        if feature_name == 'rolling_mean':
            return self._compute_rolling_mean(data, params)
        elif feature_name == 'rolling_std':
            return self._compute_rolling_std(data, params)
        elif feature_name == 'rolling_skew':
            return self._compute_rolling_skew(data, params)
        elif feature_name == 'rolling_kurt':
            return self._compute_rolling_kurt(data, params)
        elif feature_name == 'momentum':
            return self._compute_momentum(data, params)
        elif feature_name == 'volatility':
            return self._compute_volatility(data, params)
        elif feature_name == 'trend_strength':
            return self._compute_trend_strength(data, params)
        else:
            raise ValueError(f"不支持的时间序列特征: {feature_name}")

    def _compute_rolling_mean(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算滚动平均"""
        window = params.get('window', 20)
        column = params.get('column', 'close')
        return data[column].rolling(window=window).mean()

    def _compute_rolling_std(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算滚动标准差"""
        window = params.get('window', 20)
        column = params.get('column', 'close')
        return data[column].rolling(window=window).std()

    def _compute_rolling_skew(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算滚动偏度"""
        window = params.get('window', 20)
        column = params.get('column', 'close')
        return data[column].rolling(window=window).skew()

    def _compute_rolling_kurt(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算滚动峰度"""
        window = params.get('window', 20)
        column = params.get('column', 'close')
        return data[column].rolling(window=window).kurt()

    def _compute_momentum(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算动量"""
        period = params.get('period', 10)
        column = params.get('column', 'close')
        return data[column].pct_change(periods=period)

    def _compute_volatility(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算波动率"""
        window = params.get('window', 20)
        column = params.get('column', 'close')
        returns = data[column].pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # 年化波动率

    def _compute_trend_strength(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算趋势强度"""
        window = params.get('window', 20)
        column = params.get('column', 'close')

        # 使用线性回归计算趋势强度
        trend_strength = pd.Series(index=data.index, dtype=float)

        for i in range(window, len(data)):
            window_data = data[column].iloc[i - window:i + 1]
            x = np.arange(len(window_data))
            slope, _, r_value, _, _ = stats.linregress(x, window_data)
            trend_strength.iloc[i] = r_value ** 2  # R²值作为趋势强度

        return trend_strength

    def _get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """获取特征元数据"""
        metadata = {
            'name': feature_name,
            'type': 'time_series',
            'description': f'时间序列特征: {feature_name}',
            'category': 'technical'
        }

        if feature_name in ['rolling_mean', 'rolling_std', 'rolling_skew', 'rolling_kurt']:
            metadata['parameters'] = ['window', 'column']
        elif feature_name in ['momentum', 'volatility']:
            metadata['parameters'] = ['period', 'column']
        elif feature_name == 'trend_strength':
            metadata['parameters'] = ['window', 'column']

        return metadata

    def _get_available_features(self) -> List[str]:
        """获取可用特征列表"""
        return [
            'rolling_mean', 'rolling_std', 'rolling_skew', 'rolling_kurt',
            'momentum', 'volatility', 'trend_strength'
        ]


class StatisticalFeatureProcessor(BaseFeatureProcessor):

    """统计特征处理器"""

    def __init__(self, config: ProcessorConfig):

        super().__init__(config)

    def _compute_feature(self, data: pd.DataFrame, feature_name: str, params: Dict[str, Any]) -> pd.Series:
        """计算统计特征"""
        if feature_name == 'zscore':
            return self._compute_zscore(data, params)
        elif feature_name == 'percentile':
            return self._compute_percentile(data, params)
        elif feature_name == 'iqr':
            return self._compute_iqr(data, params)
        elif feature_name == 'outlier_score':
            return self._compute_outlier_score(data, params)
        elif feature_name == 'distribution_score':
            return self._compute_distribution_score(data, params)
        else:
            raise ValueError(f"不支持的统计特征: {feature_name}")

    def _compute_zscore(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算Z - score"""
        column = params.get('column', 'close')
        window = params.get('window', 20)

        rolling_mean = data[column].rolling(window=window).mean()
        rolling_std = data[column].rolling(window=window).std()

        return (data[column] - rolling_mean) / rolling_std

    def _compute_percentile(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算百分位数"""
        column = params.get('column', 'close')
        window = params.get('window', 20)
        percentile = params.get('percentile', 75)

        return data[column].rolling(window=window).quantile(percentile / 100)

    def _compute_iqr(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算四分位距"""
        column = params.get('column', 'close')
        window = params.get('window', 20)

        q75 = data[column].rolling(window=window).quantile(0.75)
        q25 = data[column].rolling(window=window).quantile(0.25)

        return q75 - q25

    def _compute_outlier_score(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算异常值分数"""
        column = params.get('column', 'close')
        window = params.get('window', 20)
        threshold = params.get('threshold', 3)

        zscore = self._compute_zscore(data, {'column': column, 'window': window})
        outlier_score = np.where(np.abs(zscore) > threshold, np.abs(zscore), 0)

        return pd.Series(outlier_score, index=data.index)

    def _compute_distribution_score(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算分布分数"""
        column = params.get('column', 'close')
        window = params.get('window', 20)

        # 计算偏度和峰度
        skew = data[column].rolling(window=window).skew()
        kurt = data[column].rolling(window=window).kurt()

        # 组合分布分数
        distribution_score = np.abs(skew) + np.abs(kurt - 3)  # 3是正态分布的峰度

        return pd.Series(distribution_score, index=data.index)

    def _get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """获取特征元数据"""
        metadata = {
            'name': feature_name,
            'type': 'statistical',
            'description': f'统计特征: {feature_name}',
            'category': 'analytical'
        }

        if feature_name == 'zscore':
            metadata['parameters'] = ['window', 'column']
        elif feature_name == 'percentile':
            metadata['parameters'] = ['window', 'column', 'percentile']
        elif feature_name == 'iqr':
            metadata['parameters'] = ['window', 'column']
        elif feature_name == 'outlier_score':
            metadata['parameters'] = ['window', 'column', 'threshold']
        elif feature_name == 'distribution_score':
            metadata['parameters'] = ['window', 'column']

        return metadata

    def _get_available_features(self) -> List[str]:
        """获取可用特征列表"""
        return [
            'zscore', 'percentile', 'iqr', 'outlier_score', 'distribution_score'
        ]


class MLFeatureProcessor(BaseFeatureProcessor):

    """机器学习特征处理器"""

    def __init__(self, config: ProcessorConfig):

        super().__init__(config)
        self.scalers = {}
        self.selectors = {}
        self.pca_models = {}

    def _compute_feature(self, data: pd.DataFrame, feature_name: str, params: Dict[str, Any]) -> pd.Series:
        """计算机器学习特征"""
        if feature_name == 'pca_component':
            return self._compute_pca_component(data, params)
        elif feature_name == 'feature_importance':
            return self._compute_feature_importance(data, params)
        elif feature_name == 'clustering_score':
            return self._compute_clustering_score(data, params)
        elif feature_name == 'anomaly_score':
            return self._compute_anomaly_score(data, params)
        else:
            raise ValueError(f"不支持的机器学习特征: {feature_name}")

    def _compute_pca_component(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算PCA主成分"""
        columns = params.get('columns', ['close', 'volume'])
        component = params.get('component', 0)
        window = params.get('window', 50)

        if len(data) < window:
            return pd.Series(np.nan, index=data.index)

        # 选择数值列
        numeric_data = data[columns].select_dtypes(include=[np.number])
        if numeric_data.empty:
            return pd.Series(np.nan, index=data.index)

        # 标准化数据
        scaler_key = f"scaler_{'_'.join(columns)}"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()

        # 滚动PCA
        pca_scores = pd.Series(index=data.index, dtype=float)

        for i in range(window, len(data)):
            window_data = numeric_data.iloc[i - window:i]
            if len(window_data) < 2:
                continue

            try:
                # 标准化
                scaled_data = self.scalers[scaler_key].fit_transform(window_data)

                # PCA
                pca_key = f"pca_{'_'.join(columns)}_{window}"
                if pca_key not in self.pca_models:
                    self.pca_models[pca_key] = PCA(n_components=min(len(columns), 2))

                pca_result = self.pca_models[pca_key].fit_transform(scaled_data)

                # 取最后一个时间点的指定主成分
                if component < pca_result.shape[1]:
                    pca_scores.iloc[i] = pca_result[-1, component]

            except Exception as e:
                self.logger.warning(f"PCA计算失败: {e}")
                continue

        return pca_scores

    def _compute_feature_importance(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算特征重要性"""
        target_column = params.get('target_column', 'close')
        feature_columns = params.get('feature_columns', ['volume', 'high', 'low'])
        window = params.get('window', 30)

        if len(data) < window:
            return pd.Series(np.nan, index=data.index)

        # 准备特征和目标变量
        feature_data = data[feature_columns].select_dtypes(include=[np.number])
        target_data = data[target_column]

        if feature_data.empty or target_data.isna().all():
            return pd.Series(np.nan, index=data.index)

        # 滚动特征重要性
        importance_scores = pd.Series(index=data.index, dtype=float)

        for i in range(window, len(data)):
            try:
                X = feature_data.iloc[i - window:i]
                y = target_data.iloc[i - window:i]

                # 移除缺失值
                valid_mask = ~(X.isna().any(axis=1) | y.isna())
                if valid_mask.sum() < 5:  # 至少需要5个有效样本
                    continue

                X_clean = X[valid_mask]
                y_clean = y[valid_mask]

                # 计算互信息
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)

                # 取平均重要性
                importance_scores.iloc[i] = np.mean(mi_scores)

            except Exception as e:
                self.logger.warning(f"特征重要性计算失败: {e}")
                continue

        return importance_scores

    def _compute_clustering_score(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算聚类分数"""
        columns = params.get('columns', ['close', 'volume'])
        window = params.get('window', 20)
        n_clusters = params.get('n_clusters', 3)

        if len(data) < window:
            return pd.Series(np.nan, index=data.index)

        # 选择数值列
        numeric_data = data[columns].select_dtypes(include=[np.number])
        if numeric_data.empty:
            return pd.Series(np.nan, index=data.index)

        # 滚动聚类
        clustering_scores = pd.Series(index=data.index, dtype=float)

        for i in range(window, len(data)):
            try:
                window_data = numeric_data.iloc[i - window:i]
                if len(window_data) < n_clusters:
                    continue

                # 标准化
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(window_data)

                # 简单的距离聚类（K - means的简化版本）
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)

                # 计算聚类质量（轮廓系数）
                from sklearn.metrics import silhouette_score
                if len(np.unique(clusters)) > 1:
                    silhouette_avg = silhouette_score(scaled_data, clusters)
                    clustering_scores.iloc[i] = silhouette_avg

            except Exception as e:
                self.logger.warning(f"聚类计算失败: {e}")
                continue

        return clustering_scores

    def _compute_anomaly_score(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算异常分数"""
        columns = params.get('columns', ['close', 'volume'])
        window = params.get('window', 30)
        threshold = params.get('threshold', 2.0)

        if len(data) < window:
            return pd.Series(np.nan, index=data.index)

        # 选择数值列
        numeric_data = data[columns].select_dtypes(include=[np.number])
        if numeric_data.empty:
            return pd.Series(np.nan, index=data.index)

        # 滚动异常检测
        anomaly_scores = pd.Series(index=data.index, dtype=float)

        for i in range(window, len(data)):
            try:
                window_data = numeric_data.iloc[i - window:i]

                # 计算马氏距离
                mean = window_data.mean()
                cov = window_data.cov()

                if cov.shape[0] > 0 and not np.isnan(cov).any():
                    # 计算当前点到分布中心的距离
                    current_point = numeric_data.iloc[i] - mean
                    try:
                        inv_cov = np.linalg.inv(cov.values)
                        mahalanobis_dist = np.sqrt(current_point.dot(inv_cov).dot(current_point))
                        anomaly_scores.iloc[i] = mahalanobis_dist
                    except np.linalg.LinAlgError:
                        # 如果协方差矩阵不可逆，使用欧几里得距离
                        euclidean_dist = np.sqrt(np.sum(current_point ** 2))
                        anomaly_scores.iloc[i] = euclidean_dist

            except Exception as e:
                self.logger.warning(f"异常检测计算失败: {e}")
                continue

        return anomaly_scores

    def _get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """获取特征元数据"""
        metadata = {
            'name': feature_name,
            'type': 'machine_learning',
            'description': f'机器学习特征: {feature_name}',
            'category': 'advanced'
        }

        if feature_name == 'pca_component':
            metadata['parameters'] = ['columns', 'component', 'window']
        elif feature_name == 'feature_importance':
            metadata['parameters'] = ['target_column', 'feature_columns', 'window']
        elif feature_name == 'clustering_score':
            metadata['parameters'] = ['columns', 'window', 'n_clusters']
        elif feature_name == 'anomaly_score':
            metadata['parameters'] = ['columns', 'window', 'threshold']

        return metadata

    def _get_available_features(self) -> List[str]:
        """获取可用特征列表"""
        return [
            'pca_component', 'feature_importance', 'clustering_score', 'anomaly_score'
        ]


# 导出主要类
__all__ = [
    'TimeSeriesFeatureProcessor',
    'StatisticalFeatureProcessor',
    'MLFeatureProcessor'
]
