from ..core.config import FeatureRegistrationConfig
import logging
"""特征质量评估器"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
# 使用统一的sklearn导入工具
from ..utils.sklearn_imports import (
    mutual_info_regression, mutual_info_classif,
    RandomForestRegressor, RandomForestClassifier,
    StandardScaler
)
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:

    """质量评估指标"""
    importance_score: float = 0.0
    correlation_score: float = 0.0
    stability_score: float = 0.0
    information_score: float = 0.0
    redundancy_score: float = 0.0
    overall_score: float = 0.0


@dataclass
class AssessmentConfig:

    """评估配置"""
    target_column: str = 'target'
    min_importance: float = 0.01
    max_correlation: float = 0.95
    min_stability: float = 0.7
    use_mutual_info: bool = True
    use_random_forest: bool = True
    n_estimators: int = 100
    random_state: int = 42
    max_sample_size: int = 2000


class FeatureQualityAssessor:

    """特征质量评估器"""

    def __init__(self, config: AssessmentConfig = None):

        self.config = config or AssessmentConfig()
        self.scaler = StandardScaler()
        self.quality_metrics = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def assess_feature_quality(


        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_configs: Optional[List[FeatureRegistrationConfig]] = None
    ) -> Dict[str, QualityMetrics]:
        """
        评估特征质量

        Args:
            features: 特征DataFrame
            target: 目标变量
            feature_configs: 特征配置列表

        Returns:
            特征质量评估结果
        """
        if features.empty:
            raise ValueError("features 数据为空，无法评估质量")
        if target.empty:
            raise ValueError("target 数据为空，无法评估质量")
        if len(features) != len(target):
            raise ValueError("features 与 target 的长度不匹配")

        logger.info("开始特征质量评估...")

        # 数据预处理
        features_clean = self._preprocess_features(features)
        model_features, model_target = self._downsample_for_model(features_clean, target)

        # 检查是否有有效特征
        if features_clean.empty:
            raise ValueError("预处理后没有可用特征")

        # 计算各项质量指标
        importance_scores = self._calculate_importance(model_features, model_target)
        correlation_scores = self._calculate_correlation(features_clean, target)
        stability_scores = self._calculate_stability(features_clean)
        information_scores = self._calculate_information_content(model_features, model_target)
        redundancy_scores = self._calculate_redundancy(features_clean)

        # 汇总质量指标
        results = {}
        for feature in features_clean.columns:
            metrics = QualityMetrics(
                importance_score=importance_scores.get(feature, 0.0),
                correlation_score=correlation_scores.get(feature, 0.0),
                stability_score=stability_scores.get(feature, 0.0),
                information_score=information_scores.get(feature, 0.0),
                redundancy_score=redundancy_scores.get(feature, 0.0)
            )

            # 计算综合评分
            metrics.overall_score = self._calculate_overall_score(metrics)
            results[feature] = metrics

        self.quality_metrics = results
        logger.info(f"特征质量评估完成，共评估{len(results)}个特征")

        report = {
            'feature_scores': results,
            'recommendations': self._generate_recommendations(),
            'summary': self._build_summary(results),
        }

        return report

    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """预处理特征数据"""
        if features.empty:
            return features

        # 移除常量特征
        constant_features = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"发现{len(constant_features)}个常量特征: {constant_features}")
            features = features.drop(columns=constant_features)

        # 处理缺失值
        features = features.ffill().bfill()

        # 移除无穷值
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()

        return features

    def _calculate_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """计算特征重要性"""
        importance_scores = {}

        try:
            # 确保数据不为空
            if features.empty or target.empty:
                return importance_scores

            # 确保特征和目标变量长度一致
            if len(features) != len(target):
                min_len = min(len(features), len(target))
                features = features.iloc[:min_len]
                target = target.iloc[:min_len]

            # 随机森林重要性
            if self.config.use_random_forest and len(features.columns) > 0:
                if target.dtype in ['object', 'category'] or target.nunique() < 10:
                    # 分类问题
                    rf = RandomForestClassifier(
                        n_estimators=self.config.n_estimators,
                        random_state=self.config.random_state
                    )
                    rf.fit(features, target)
                    importances = rf.feature_importances_
                else:
                    # 回归问题
                    rf = RandomForestRegressor(
                        n_estimators=self.config.n_estimators,
                        random_state=self.config.random_state
                    )
                    rf.fit(features, target)
                    importances = rf.feature_importances_

                for i, feature in enumerate(features.columns):
                    importance_scores[feature] = importances[i]

            # 互信息重要性
            if self.config.use_mutual_info and len(features.columns) > 0:
                if target.dtype in ['object', 'category'] or target.nunique() < 10:
                    mi_scores = mutual_info_classif(
                        features, target, random_state=self.config.random_state)
                else:
                    mi_scores = mutual_info_regression(
                        features, target, random_state=self.config.random_state)

                for i, feature in enumerate(features.columns):
                    if feature not in importance_scores:
                        importance_scores[feature] = 0.0
                    importance_scores[feature] = (importance_scores[feature] + mi_scores[i]) / 2

        except Exception as e:
            logger.error(f"计算特征重要性失败: {e}")

        return importance_scores

    def _calculate_correlation(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """计算特征与目标变量的相关性"""
        correlation_scores = {}

        try:
            if features.empty or target.empty:
                return correlation_scores

            # 确保长度一致
            if len(features) != len(target):
                min_len = min(len(features), len(target))
                features = features.iloc[:min_len]
                target = target.iloc[:min_len]

            for feature in features.columns:
                try:
                    corr = abs(features[feature].corr(target))
                    correlation_scores[feature] = corr if not np.isnan(corr) else 0.0
                except Exception:
                    correlation_scores[feature] = 0.0
        except Exception as e:
            logger.error(f"计算相关性失败: {e}")

        return correlation_scores

    def _calculate_stability(self, features: pd.DataFrame) -> Dict[str, float]:
        """计算特征稳定性"""
        stability_scores = {}

        try:
            for feature in features.columns:
                try:
                    # 计算特征值的变异系数
                    values = features[feature].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        if mean_val != 0:
                            cv = std_val / abs(mean_val)
                            # 稳定性评分：变异系数越小，稳定性越高
                            stability_scores[feature] = max(0, 1 - cv)
                        else:
                            stability_scores[feature] = 1.0
                    else:
                        stability_scores[feature] = 0.0
                except Exception:
                    stability_scores[feature] = 0.0
        except Exception as e:
            logger.error(f"计算特征稳定性失败: {e}")

        return stability_scores

    def _calculate_information_content(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """计算特征信息含量"""
        information_scores = {}

        try:
            for feature in features.columns:
                try:
                    # 计算特征的信息熵
                    values = features[feature].dropna()
                    if len(values) > 0:
                        # 简化的信息含量计算
                        unique_ratio = values.nunique() / len(values)
                        information_scores[feature] = unique_ratio
                    else:
                        information_scores[feature] = 0.0
                except Exception:
                    information_scores[feature] = 0.0
        except Exception as e:
            logger.error(f"计算信息含量失败: {e}")

        return information_scores

    def _calculate_redundancy(self, features: pd.DataFrame) -> Dict[str, float]:
        """计算特征冗余度"""
        redundancy_scores = {}

        try:
            if len(features.columns) < 2:
                # 如果只有一个特征，冗余度为0
                for feature in features.columns:
                    redundancy_scores[feature] = 0.0
                return redundancy_scores

            # 计算特征间的相关性矩阵
            corr_matrix = features.corr().abs()

            for feature in features.columns:
                try:
                    # 计算与其他特征的平均相关性
                    other_features = [col for col in features.columns if col != feature]
                    if other_features:
                        avg_corr = corr_matrix.loc[feature, other_features].mean()
                        # 冗余度评分：相关性越高，冗余度越高
                        redundancy_scores[feature] = avg_corr
                    else:
                        redundancy_scores[feature] = 0.0
                except Exception:
                    redundancy_scores[feature] = 0.0
        except Exception as e:
            logger.error(f"计算特征冗余度失败: {e}")

        return redundancy_scores

    def _identify_redundant_features(self, features: pd.DataFrame, threshold: float = 0.9) -> List[str]:
        """识别冗余特征"""
        if features.empty or len(features.columns) < 2:
            return []

        corr_matrix = features.corr().abs().fillna(0)
        redundant = set()

        for i, col_i in enumerate(corr_matrix.columns):
            for col_j in corr_matrix.columns[i + 1:]:
                if corr_matrix.loc[col_i, col_j] >= threshold:
                    redundant.add(col_j if features[col_j].std() <= features[col_i].std() else col_i)

        return list(redundant)

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """计算综合质量评分"""
        # 权重配置
        weights = {
            'importance': 0.3,
            'correlation': 0.25,
            'stability': 0.2,
            'information': 0.15,
            'redundancy': 0.1
        }

        # 计算加权平均分
        overall_score = (
            metrics.importance_score * weights['importance']
            + metrics.correlation_score * weights['correlation']
            + metrics.stability_score * weights['stability']
            + metrics.information_score * weights['information']
            + (1 - metrics.redundancy_score) * weights['redundancy']  # 冗余度越低越好
        )

        return min(1.0, max(0.0, overall_score))

    def get_quality_report(self) -> Dict[str, Any]:
        """生成质量评估报告"""
        if not self.quality_metrics:
            return {}

        # 统计信息
        total_features = len(self.quality_metrics)
        high_quality_features = sum(
            1 for m in self.quality_metrics.values() if m.overall_score >= 0.8)
        medium_quality_features = sum(
            1 for m in self.quality_metrics.values() if 0.6 <= m.overall_score < 0.8)
        low_quality_features = sum(
            1 for m in self.quality_metrics.values() if m.overall_score < 0.6)

        # 特征排名
        sorted_features = sorted(
            self.quality_metrics.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )

        report = {
            'summary': {
                'total_features': total_features,
                'high_quality_count': high_quality_features,
                'medium_quality_count': medium_quality_features,
                'low_quality_count': low_quality_features,
                'avg_overall_score': np.mean([m.overall_score for m in self.quality_metrics.values()])
            },
            'top_features': sorted_features[:10],
            'bottom_features': sorted_features[-10:],
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self, quality_report: Optional[Dict[str, Any]] = None) -> List[str]:
        """生成改进建议"""
        recommendations: List[str] = []

        metrics_source = None
        if quality_report:
            metrics_source = quality_report.get('feature_scores')
        if metrics_source is None:
            metrics_source = self.quality_metrics

        if not metrics_source:
            return recommendations

        # 支持输入为 dict 或 QualityMetrics 实例
        normalized_metrics = {}
        for name, metrics in metrics_source.items():
            if isinstance(metrics, QualityMetrics):
                normalized_metrics[name] = metrics
            elif isinstance(metrics, dict):
                normalized_metrics[name] = QualityMetrics(**metrics)
            else:
                continue

        low_quality = [
            (name, metrics) for name, metrics in normalized_metrics.items()
            if metrics.overall_score < 0.6
        ]

        for name, metrics in low_quality:
            if metrics.importance_score < self.config.min_importance:
                recommendations.append(f"特征 '{name}' 重要性较低，建议移除或重新设计")

            if metrics.correlation_score < 0.1:
                recommendations.append(f"特征 '{name}' 与目标变量相关性较低，建议检查特征设计")

            if metrics.stability_score < self.config.min_stability:
                recommendations.append(f"特征 '{name}' 稳定性较差，建议增加数据预处理")

            if metrics.redundancy_score > 0.8:
                recommendations.append(f"特征 '{name}' 冗余度较高，建议移除或合并相关特征")

        return recommendations

    def _build_summary(self, results: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        if not results:
            return {}
        overall_scores = [metrics.overall_score for metrics in results.values()]
        return {
            'total_features': len(results),
            'avg_overall_score': float(np.mean(overall_scores)),
            'high_quality_count': int(sum(score >= 0.8 for score in overall_scores)),
            'medium_quality_count': int(sum(0.6 <= score < 0.8 for score in overall_scores)),
            'low_quality_count': int(sum(score < 0.6 for score in overall_scores)),
        }

    def filter_features(


        self,
        features: pd.DataFrame,
        min_score: float = 0.6,
        max_features: Optional[int] = None
    ) -> pd.DataFrame:
        """根据质量评分过滤特征"""
        if not self.quality_metrics:
            logger.warning("未进行质量评估，无法过滤特征")
            return features

        # 筛选高质量特征
        selected_features = [
            name for name, metrics in self.quality_metrics.items()
            if metrics.overall_score >= min_score
        ]

        # 限制特征数量
        if max_features and len(selected_features) > max_features:
            # 按质量评分排序，选择前N个
            sorted_features = sorted(
                self.quality_metrics.items(),
                key=lambda x: x[1].overall_score,
                reverse=True
            )
            selected_features = [name for name, _ in sorted_features[:max_features]]

        # 确保选择的特征存在于DataFrame中
        available_features = [f for f in selected_features if f in features.columns]

        if available_features:
            filtered_features = features[available_features]
            logger.info(f"特征过滤完成，从{len(features.columns)}个特征中选择{len(available_features)}个")
        else:
            filtered_features = features
            logger.warning("没有特征通过质量过滤，返回原始特征")

        return filtered_features

    def _downsample_for_model(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """为重型模型计算进行下采样，提升性能"""
        max_samples = getattr(self.config, "max_sample_size", None)
        if not max_samples or len(features) <= max_samples:
            return features, target

        sampled_index = features.sample(
            n=max_samples,
            random_state=self.config.random_state
        ).index
        logger.debug(
            "特征样本数 %d 超过阈值 %d，已下采样用于重要性计算",
            len(features),
            max_samples,
        )
        return features.loc[sampled_index], target.loc[sampled_index]

    # -------------- 测试辅助方法 --------------
    def _calculate_importance_scores(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        return self._calculate_importance(features, target)

    def _calculate_correlation_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()
        return features.corr()

    def _calculate_stability_scores(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, float]:
        return self._calculate_stability(features)

    def _calculate_information_scores(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, float]:
        return self._calculate_information_content(features, target if target is not None else pd.Series([0] * len(features)))
