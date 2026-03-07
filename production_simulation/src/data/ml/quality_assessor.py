from typing import Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings


@dataclass
class QualityAssessmentConfig:

    """数据质量评估配置"""
    anomaly_detection_enabled: bool = True
    completeness_threshold: float = 0.95
    consistency_threshold: float = 0.9
    outlier_detection_enabled: bool = True
    clustering_enabled: bool = False
    max_features_for_ml: int = 10


class MLQualityAssessor:

    """
    机器学习驱动的数据质量评估器
    """

    def __init__(self, config: QualityAssessmentConfig = None):

        self.config = config or QualityAssessmentConfig()
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self._is_fitted = False

    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        评估数据质量
        """
        if data.empty:
            return {
                'overall_score': 0.0,
                'completeness': 0.0,
                'consistency': 0.0,
                'anomaly_score': 0.0,
                'outlier_ratio': 0.0,
                'recommendations': ['数据为空，无法评估质量']
            }

        results = {}

        # 基础质量指标
        results['completeness'] = self._assess_completeness(data)
        results['consistency'] = self._assess_consistency(data)

        # 机器学习驱动的异常检测
        if self.config.anomaly_detection_enabled:
            results['anomaly_score'] = self._detect_anomalies(data)

        # 异常值检测
        if self.config.outlier_detection_enabled:
            results['outlier_ratio'] = self._detect_outliers(data)

        # 聚类分析（可选）
        if self.config.clustering_enabled:
            results['clustering_score'] = self._assess_clustering(data)

        # 计算综合评分
        results['overall_score'] = self._calculate_overall_score(results)

        # 生成建议
        results['recommendations'] = self._generate_recommendations(results, data)

        return results

    def _assess_completeness(self, data: pd.DataFrame) -> float:
        """评估数据完整性"""
        if data.empty:
            return 0.0

        # 计算非空值比例
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        return min(completeness, 1.0)

    def _assess_consistency(self, data: pd.DataFrame) -> float:
        """评估数据一致性"""
        if data.empty:
            return 0.0

        consistency_scores = []

        # 检查数据类型一致性
        for col in data.columns:
            if data[col].dtype in ['object', 'string']:
                # 字符串列的一致性（非空且格式一致）
                non_null = data[col].dropna()
                if len(non_null) > 0:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)
            else:
                # 数值列的一致性（检查是否有异常值）
                non_null = data[col].dropna()
                if len(non_null) > 0:
                    # 使用IQR方法检测异常值
                    Q1 = non_null.quantile(0.25)
                    Q3 = non_null.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((non_null < (Q1 - 1.5 * IQR)) | (non_null > (Q3 + 1.5 * IQR))).sum()
                    consistency_ratio = 1 - (outliers / len(non_null))
                    consistency_scores.append(max(consistency_ratio, 0.0))
                else:
                    consistency_scores.append(0.0)

        return np.mean(consistency_scores) if consistency_scores else 0.0

    def _detect_anomalies(self, data: pd.DataFrame) -> float:
        """使用机器学习检测异常"""
        try:
            # 选择数值列进行异常检测
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return 0.0

            # 限制特征数量
            if len(numeric_cols) > self.config.max_features_for_ml:
                numeric_cols = numeric_cols[:self.config.max_features_for_ml]

            # 准备数据
            X = data[numeric_cols].fillna(data[numeric_cols].median())

            if len(X) < 10:  # 数据太少，无法进行有效检测
                return 0.5

            # 标准化
            X_scaled = self.scaler.fit_transform(X)

            # 训练异常检测模型
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.anomaly_detector.fit(X_scaled)

            # 预测异常
            predictions = self.anomaly_detector.predict(X_scaled)
            anomaly_ratio = (predictions == -1).mean()

            self._is_fitted = True

            # 返回异常检测质量分数（异常比例越低，质量越高）
            return max(1.0 - anomaly_ratio, 0.0)

        except Exception as e:
            warnings.warn(f"异常检测失败: {e}")
            return 0.5

    def _detect_outliers(self, data: pd.DataFrame) -> float:
        """检测异常值比例"""
        outlier_ratios = []

        for col in data.select_dtypes(include=[np.number]).columns:
            non_null = data[col].dropna()
            if len(non_null) > 0:
                Q1 = non_null.quantile(0.25)
                Q3 = non_null.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((non_null < (Q1 - 1.5 * IQR)) | (non_null > (Q3 + 1.5 * IQR))).sum()
                outlier_ratio = outliers / len(non_null)
                outlier_ratios.append(outlier_ratio)

        return np.mean(outlier_ratios) if outlier_ratios else 0.0

    def _assess_clustering(self, data: pd.DataFrame) -> float:
        """评估数据聚类质量"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2 or len(data) < 10:
                return 0.5

            # 限制特征数量
            if len(numeric_cols) > self.config.max_features_for_ml:
                numeric_cols = numeric_cols[:self.config.max_features_for_ml]

            X = data[numeric_cols].fillna(data[numeric_cols].median())
            X_scaled = StandardScaler().fit_transform(X)

            # 使用K - means进行聚类
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(3, len(X)), random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            # 计算轮廓系数
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                return max(score, 0.0)
            else:
                return 0.5

        except Exception as e:
            warnings.warn(f"聚类评估失败: {e}")
            return 0.5

    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """计算综合质量评分"""
        scores = []
        weights = []

        # 完整性权重
        if 'completeness' in results:
            scores.append(results['completeness'])
            weights.append(0.3)

        # 一致性权重
        if 'consistency' in results:
            scores.append(results['consistency'])
            weights.append(0.3)

        # 异常检测权重
        if 'anomaly_score' in results:
            scores.append(results['anomaly_score'])
            weights.append(0.2)

        # 异常值检测权重
        if 'outlier_ratio' in results:
            # 异常值比例越低越好
            outlier_score = max(1.0 - results['outlier_ratio'], 0.0)
            scores.append(outlier_score)
            weights.append(0.2)

        # 聚类质量权重（可选）
        if 'clustering_score' in results:
            scores.append(results['clustering_score'])
            weights.append(0.1)

        if not scores:
            return 0.0

        # 加权平均
        total_weight = sum(weights)
        if total_weight == 0:
            return np.mean(scores)

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return min(weighted_score, 1.0)

    def _generate_recommendations(self, results: Dict[str, Any], data: pd.DataFrame) -> List[str]:
        """生成质量改进建议"""
        recommendations = []

        # 完整性建议
        if results.get('completeness', 1.0) < self.config.completeness_threshold:
            missing_ratio = 1 - results['completeness']
            recommendations.append(f"数据完整性不足 ({results['completeness']:.2%})，建议补充缺失数据")

        # 一致性建议
        if results.get('consistency', 1.0) < self.config.consistency_threshold:
            recommendations.append(f"数据一致性不足 ({results['consistency']:.2%})，建议检查数据格式和异常值")

        # 异常检测建议
        if 'anomaly_score' in results and results['anomaly_score'] < 0.8:
            recommendations.append("检测到数据异常模式，建议进一步分析异常数据")

        # 异常值建议
        if 'outlier_ratio' in results and results['outlier_ratio'] > 0.1:
            recommendations.append(f"异常值比例较高 ({results['outlier_ratio']:.2%})，建议检查数据来源和处理流程")

        # 数据量建议
        if len(data) < 100:
            recommendations.append("数据量较少，可能影响机器学习模型的准确性")

        if not recommendations:
            recommendations.append("数据质量良好，无需特殊处理")

        return recommendations

    def get_quality_summary(self, data: pd.DataFrame) -> str:
        """获取质量评估摘要"""
        results = self.assess_data_quality(data)

        summary = f"""
            数据质量评估摘要:
- 综合评分: {results['overall_score']:.2%}
- 完整性: {results.get('completeness', 0):.2%}
- 一致性: {results.get('consistency', 0):.2%}
- 异常检测: {results.get('anomaly_score', 0):.2%}
- 异常值比例: {results.get('outlier_ratio', 0):.2%}

建议:
"""
        for rec in results['recommendations']:
            summary += f"- {rec}\n"

        return summary
