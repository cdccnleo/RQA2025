import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征稳定性自动检测

实现特征稳定性检测和时间一致性分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class FeatureStabilityAnalyzer:

    """特征稳定性分析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {
            'stability_threshold': 0.8,
            'drift_threshold': 0.1,
            'time_window_size': 30,
            'min_samples': 100,
            'random_state': 42
        }
        self.scaler = StandardScaler()
        self.stability_scores = {}
        self.drift_indicators = {}

    def analyze_feature_stability(self,


                                  features: pd.DataFrame,
                                  time_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """
        分析特征稳定性

        Args:
            features: 特征数据
            time_index: 时间索引（可选）

        Returns:
            特征稳定性分析结果
        """
        logger.info(f"开始特征稳定性分析，特征数量: {len(features.columns)}")

        # 数据预处理
        features_processed = self._preprocess_features(features)

        # 多种稳定性分析方法
        results = {
            'statistical_stability': self._analyze_statistical_stability(features_processed),
            'distribution_stability': self._analyze_distribution_stability(features_processed),
            'temporal_stability': self._analyze_temporal_stability(features_processed, time_index),
            'correlation_stability': self._analyze_correlation_stability(features_processed),
            'drift_detection': self._detect_feature_drift(features_processed, time_index)
        }

        # 综合稳定性评分
        combined_stability = self._combine_stability_scores(results)

        # 生成分析报告
        analysis_report = self._generate_stability_report(results, combined_stability)

        self.stability_scores = combined_stability

        logger.info(f"特征稳定性分析完成，生成了 {len(combined_stability)} 个稳定性评分")

        return {
            'analysis_results': results,
            'combined_stability': combined_stability,
            'analysis_report': analysis_report
        }

    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """预处理特征数据"""
        # 处理缺失值
        features_processed = features.copy()
        features_processed = features_processed.fillna(features_processed.mean())

        # 标准化
        features_scaled = self.scaler.fit_transform(features_processed)
        features_processed = pd.DataFrame(features_scaled,
                                          columns=features_processed.columns,
                                          index=features_processed.index)

        return features_processed

    def _analyze_statistical_stability(self, features: pd.DataFrame) -> Dict[str, float]:
        """分析统计稳定性"""
        stability_scores = {}

        for col in features.columns:
            # 计算基本统计量
            mean_val = features[col].mean()
            std_val = features[col].std()
            skew_val = features[col].skew()
            kurt_val = features[col].kurtosis()

            # 计算变异系数（CV）
            cv = abs(std_val / mean_val) if mean_val != 0 else float('inf')

            # 计算稳定性评分（基于CV的倒数）
            if cv != float('inf') and cv > 0:
                stability_score = 1 / (1 + cv)
            else:
                stability_score = 0.0

            stability_scores[col] = min(1.0, max(0.0, stability_score))

        return stability_scores

    def _analyze_distribution_stability(self, features: pd.DataFrame) -> Dict[str, float]:
        """分析分布稳定性"""
        stability_scores = {}

        for col in features.columns:
            # 计算分位数
            quantiles = features[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

            # 计算分位数间距的稳定性
            q1_q3_range = quantiles[0.75] - quantiles[0.25]
            q10_q90_range = quantiles[0.9] - quantiles[0.1]

            # 计算分布偏度
            skewness = abs(features[col].skew())

            # 计算分布峰度
            kurtosis = abs(features[col].kurtosis())

            # 综合稳定性评分
            range_stability = 1 / (1 + abs(q1_q3_range - q10_q90_range))
            skew_stability = 1 / (1 + skewness)
            kurt_stability = 1 / (1 + kurtosis)

            # 加权平均
            stability_score = (range_stability * 0.4 + skew_stability * 0.3 + kurt_stability * 0.3)
            stability_scores[col] = min(1.0, max(0.0, stability_score))

        return stability_scores

    def _analyze_temporal_stability(self, features: pd.DataFrame, time_index: Optional[pd.DatetimeIndex]) -> Dict[str, float]:
        """分析时间稳定性"""
        stability_scores = {}

        if time_index is None or len(time_index) < self.config['min_samples']:
            # 如果没有时间索引或数据不足，返回默认值
            return {col: 0.5 for col in features.columns}

        # 设置时间索引
        features_with_time = features.copy()
        features_with_time.index = time_index

        for col in features.columns:
            try:
                # 按时间窗口计算统计量
                window_size = self.config['time_window_size']
                rolling_mean = features_with_time[col].rolling(
                    window=window_size, min_periods=1).mean()
                rolling_std = features_with_time[col].rolling(
                    window=window_size, min_periods=1).std()

                # 计算时间序列的稳定性
                mean_stability = 1 - (rolling_mean.std() / abs(rolling_mean.mean())
                                      if rolling_mean.mean() != 0 else 0)
                std_stability = 1 - (rolling_std.std() / rolling_std.mean()
                                     if rolling_std.mean() != 0 else 0)

                # 计算趋势稳定性
                trend = np.polyfit(
                    range(len(features_with_time[col])), features_with_time[col], 1)[0]
                trend_stability = 1 / (1 + abs(trend))

                # 综合时间稳定性评分
                temporal_stability = (mean_stability * 0.4 + std_stability *
                                      0.4 + trend_stability * 0.2)
                stability_scores[col] = min(1.0, max(0.0, temporal_stability))

            except Exception as e:
                logger.warning(f"计算特征 {col} 的时间稳定性失败: {e}")
                stability_scores[col] = 0.5

        return stability_scores

    def _analyze_correlation_stability(self, features: pd.DataFrame) -> Dict[str, float]:
        """分析相关性稳定性"""
        stability_scores = {}

        if len(features.columns) < 2:
            return {col: 1.0 for col in features.columns}

        # 计算相关性矩阵
        correlation_matrix = features.corr()

        for col in features.columns:
            try:
                # 计算该特征与其他特征的相关性稳定性
                correlations = correlation_matrix[col].drop(col)

                # 计算相关性的一致性
                correlation_std = correlations.std()
                correlation_mean = correlations.mean()

                # 相关性稳定性评分
                if correlation_mean != 0:
                    correlation_stability = 1 / (1 + correlation_std / abs(correlation_mean))
                else:
                    correlation_stability = 1 / (1 + correlation_std)

                stability_scores[col] = min(1.0, max(0.0, correlation_stability))

            except Exception as e:
                logger.warning(f"计算特征 {col} 的相关性稳定性失败: {e}")
                stability_scores[col] = 0.5

        return stability_scores

    def _detect_feature_drift(self, features: pd.DataFrame, time_index: Optional[pd.DatetimeIndex]) -> Dict[str, Any]:
        """检测特征漂移"""
        drift_results = {
            'drift_scores': {},
            'drift_indicators': {},
            'drift_severity': {}
        }

        if time_index is None or len(time_index) < self.config['min_samples']:
            return drift_results

        # 设置时间索引
        features_with_time = features.copy()
        features_with_time.index = time_index

        # 将数据分为前后两部分
        mid_point = len(features_with_time) // 2
        first_half = features_with_time.iloc[:mid_point]
        second_half = features_with_time.iloc[mid_point:]

        for col in features.columns:
            try:
                # 计算分布漂移
                first_dist = first_half[col]
                second_dist = second_half[col]

                # KS检验统计量（简化版）
                ks_statistic = self._calculate_ks_statistic(first_dist, second_dist)

                # 均值漂移
                mean_drift = abs(second_dist.mean() - first_dist.mean()) / (first_dist.std() + 1e-8)

                # 方差漂移
                var_drift = abs(second_dist.var() - first_dist.var()) / (first_dist.var() + 1e-8)

                # 综合漂移评分
                drift_score = (ks_statistic + mean_drift + var_drift) / 3

                # 漂移严重程度
                if drift_score < self.config['drift_threshold']:
                    severity = 'low'
                elif drift_score < self.config['drift_threshold'] * 2:
                    severity = 'medium'
                else:
                    severity = 'high'

                drift_results['drift_scores'][col] = drift_score
                drift_results['drift_indicators'][col] = {
                    'ks_statistic': ks_statistic,
                    'mean_drift': mean_drift,
                    'var_drift': var_drift
                }
                drift_results['drift_severity'][col] = severity

            except Exception as e:
                logger.warning(f"检测特征 {col} 的漂移失败: {e}")
                drift_results['drift_scores'][col] = 0.0
                drift_results['drift_indicators'][col] = {
                    'ks_statistic': 0.0, 'mean_drift': 0.0, 'var_drift': 0.0}
                drift_results['drift_severity'][col] = 'unknown'

        return drift_results

    def _calculate_ks_statistic(self, dist1: pd.Series, dist2: pd.Series) -> float:
        """计算KS统计量（简化版）"""
        try:
            # 计算经验分布函数

            def ecdf(data):

                sorted_data = np.sort(data)
                n = len(sorted_data)
                return lambda x: np.searchsorted(sorted_data, x, side='right') / n

            ecdf1 = ecdf(dist1)
            ecdf2 = ecdf(dist2)

            # 计算KS统计量
            combined_data = np.concatenate([dist1, dist2])
            ks_stat = np.max(np.abs(ecdf1(combined_data) - ecdf2(combined_data)))

            return ks_stat
        except Exception:
            return 0.0

    def _combine_stability_scores(self, individual_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """综合多种稳定性评分"""
        features = list(individual_scores['statistical_stability'].keys())
        combined_scores = {}

        for feature in features:
            scores = []
            weights = {
                'statistical_stability': 0.25,
                'distribution_stability': 0.25,
                'temporal_stability': 0.3,
                'correlation_stability': 0.2
            }

            weighted_score = 0.0
            total_weight = 0.0

            for method, weight in weights.items():
                if method in individual_scores and feature in individual_scores[method]:
                    score = individual_scores[method][feature]
                    weighted_score += score * weight
                    total_weight += weight

            combined_scores[feature] = weighted_score / total_weight if total_weight > 0 else 0.0

        return combined_scores

    def _generate_stability_report(self, results: Dict[str, Any], combined_stability: Dict[str, float]) -> Dict[str, Any]:
        """生成稳定性分析报告"""
        drift_results = results['drift_detection']

        # 计算稳定性统计
        stability_values = list(combined_stability.values())
        high_stability_count = len([s for s in stability_values if s > 0.8])
        medium_stability_count = len([s for s in stability_values if 0.5 <= s <= 0.8])
        low_stability_count = len([s for s in stability_values if s < 0.5])

        # 计算漂移统计
        drift_scores = list(drift_results['drift_scores'].values())
        high_drift_count = len([d for d in drift_scores if d > self.config['drift_threshold'] * 2])
        medium_drift_count = len(
            [d for d in drift_scores if self.config['drift_threshold'] <= d <= self.config['drift_threshold'] * 2])
        low_drift_count = len([d for d in drift_scores if d < self.config['drift_threshold']])

        report = {
            'summary': {
                'total_features': len(combined_stability),
                'high_stability_features': high_stability_count,
                'medium_stability_features': medium_stability_count,
                'low_stability_features': low_stability_count,
                'high_drift_features': high_drift_count,
                'medium_drift_features': medium_drift_count,
                'low_drift_features': low_drift_count
            },
            'recommendations': [],
            'stability_ranking': sorted(combined_stability.items(), key=lambda x: x[1], reverse=True),
            'drift_ranking': sorted(drift_results['drift_scores'].items(), key=lambda x: x[1], reverse=True)
        }

        # 生成建议
        if low_stability_count > len(combined_stability) * 0.3:
            report['recommendations'].append("检测到较多低稳定性特征，建议进行特征工程或数据质量改进")

        if high_drift_count > 0:
            report['recommendations'].append(f"检测到 {high_drift_count} 个高漂移特征，建议监控这些特征的数据质量")

        if high_stability_count < len(combined_stability) * 0.2:
            report['recommendations'].append("高稳定性特征较少，建议增加更多稳定的特征")

        return report

    def get_stability_recommendations(self, threshold: float = 0.7) -> Dict[str, List[str]]:
        """获取稳定性建议"""
        if not self.stability_scores:
            return {'stable': [], 'unstable': [], 'monitor': []}

        recommendations = {
            'stable': [],
            'unstable': [],
            'monitor': []
        }

        for feature, score in self.stability_scores.items():
            if score > threshold:
                recommendations['stable'].append(feature)
            elif score < threshold * 0.7:
                recommendations['unstable'].append(feature)
            else:
                recommendations['monitor'].append(feature)

        return recommendations

    def export_stability_report(self, filepath: str):
        """导出稳定性报告"""
        if not self.stability_scores:
            logger.warning("没有稳定性评分数据，无法导出报告")
            return

        report_data = {
            'stability_scores': self.stability_scores,
            'stability_ranking': sorted(self.stability_scores.items(), key=lambda x: x[1], reverse=True),
            'recommendations': self.get_stability_recommendations()
        }

        # 保存为JSON格式
        import json
        with open(filepath, 'w', encoding='utf - 8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"特征稳定性报告已导出到: {filepath}")
