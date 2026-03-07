#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征质量评估系统

提供全面的特征质量评估功能，包括：
- 特征完整性评估
- 特征有效性评估
- 特征稳定性评估
- 特征相关性评估
- 特征质量报告生成
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FeatureQualityMetrics:
    """特征质量指标"""
    feature_name: str
    completeness: float = 0.0  # 完整性 (0-1)
    validity: float = 0.0      # 有效性 (0-1)
    stability: float = 0.0     # 稳定性 (0-1)
    uniqueness: float = 0.0    # 唯一性 (0-1)
    correlation: float = 0.0   # 相关性 (0-1)
    overall_score: float = 0.0 # 综合评分 (0-1)
    quality_level: str = "unknown"  # 质量等级: 优秀/良好/一般/较差
    details: Dict[str, Any] = field(default_factory=dict)


class FeatureQualityAssessor:
    """
    特征质量评估器
    
    对特征数据进行全面的质量评估
    """
    
    def __init__(self):
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "fair": 0.5,
            "poor": 0.0
        }
    
    def assess_feature_quality(
        self,
        feature_data: pd.Series,
        target: Optional[pd.Series] = None,
        feature_name: str = ""
    ) -> FeatureQualityMetrics:
        """
        评估单个特征的质量
        
        Args:
            feature_data: 特征数据
            target: 目标变量（可选）
            feature_name: 特征名称
            
        Returns:
            特征质量指标
        """
        metrics = FeatureQualityMetrics(feature_name=feature_name)
        
        try:
            # 1. 完整性评估
            metrics.completeness = self._assess_completeness(feature_data)
            
            # 2. 有效性评估
            metrics.validity = self._assess_validity(feature_data)
            
            # 3. 稳定性评估
            metrics.stability = self._assess_stability(feature_data)
            
            # 4. 唯一性评估
            metrics.uniqueness = self._assess_uniqueness(feature_data)
            
            # 5. 相关性评估（如果有目标变量）
            if target is not None:
                metrics.correlation = self._assess_correlation(feature_data, target)
            
            # 6. 计算综合评分
            weights = {
                'completeness': 0.25,
                'validity': 0.25,
                'stability': 0.2,
                'uniqueness': 0.15,
                'correlation': 0.15
            }
            
            metrics.overall_score = (
                metrics.completeness * weights['completeness'] +
                metrics.validity * weights['validity'] +
                metrics.stability * weights['stability'] +
                metrics.uniqueness * weights['uniqueness'] +
                metrics.correlation * weights['correlation']
            )
            
            # 7. 确定质量等级
            metrics.quality_level = self._get_quality_level(metrics.overall_score)
            
            # 8. 记录详细信息
            metrics.details = {
                'data_type': str(feature_data.dtype),
                'data_count': len(feature_data),
                'missing_count': feature_data.isnull().sum(),
                'unique_count': feature_data.nunique(),
                'mean': float(feature_data.mean()) if pd.api.types.is_numeric_dtype(feature_data) else None,
                'std': float(feature_data.std()) if pd.api.types.is_numeric_dtype(feature_data) else None,
                'min': float(feature_data.min()) if pd.api.types.is_numeric_dtype(feature_data) else None,
                'max': float(feature_data.max()) if pd.api.types.is_numeric_dtype(feature_data) else None
            }
            
        except Exception as e:
            logger.error(f"评估特征 {feature_name} 质量失败: {e}")
            metrics.quality_level = "error"
            metrics.details['error'] = str(e)
        
        return metrics
    
    def assess_features_quality(
        self,
        features_df: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        评估多个特征的质量
        
        Args:
            features_df: 特征数据框
            target: 目标变量（可选）
            
        Returns:
            质量评估结果
        """
        results = {
            'feature_metrics': [],
            'quality_distribution': {
                '优秀': 0,
                '良好': 0,
                '一般': 0,
                '较差': 0
            },
            'overall_score': 0.0,
            'assessment_time': datetime.now().isoformat()
        }
        
        try:
            total_score = 0.0
            
            # 检查是否只有原始数据列（没有计算出的特征列）
            original_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            has_computed_features = any(col not in original_columns for col in features_df.columns)
            
            for column in features_df.columns:
                # 跳过非特征列（但如果只有原始数据列，则评估原始数据列）
                if column in original_columns and has_computed_features:
                    continue
                
                # 评估单个特征
                metrics = self.assess_feature_quality(
                    features_df[column],
                    target,
                    column
                )
                
                results['feature_metrics'].append({
                    'feature_name': metrics.feature_name,
                    'completeness': metrics.completeness,
                    'validity': metrics.validity,
                    'stability': metrics.stability,
                    'uniqueness': metrics.uniqueness,
                    'correlation': metrics.correlation,
                    'overall_score': metrics.overall_score,
                    'quality_level': metrics.quality_level,
                    'details': metrics.details
                })
                
                # 统计质量分布
                results['quality_distribution'][metrics.quality_level] = \
                    results['quality_distribution'].get(metrics.quality_level, 0) + 1
                
                total_score += metrics.overall_score
            
            # 计算整体评分
            if results['feature_metrics']:
                results['overall_score'] = total_score / len(results['feature_metrics'])
            
            logger.info(f"特征质量评估完成: {len(results['feature_metrics'])} 个特征, "
                       f"整体评分: {results['overall_score']:.2f}")
            
        except Exception as e:
            logger.error(f"评估特征质量失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _assess_completeness(self, data: pd.Series) -> float:
        """评估数据完整性"""
        if len(data) == 0:
            return 0.0
        
        missing_ratio = data.isnull().sum() / len(data)
        return 1.0 - missing_ratio
    
    def _assess_validity(self, data: pd.Series) -> float:
        """评估数据有效性"""
        if len(data) == 0:
            return 0.0
        
        # 检查异常值
        if pd.api.types.is_numeric_dtype(data):
            # 使用IQR方法检测异常值
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_ratio = outliers / len(data)
            
            # 检查无穷值
            inf_count = np.isinf(data).sum()
            inf_ratio = inf_count / len(data)
            
            return 1.0 - outlier_ratio - inf_ratio
        else:
            # 非数值类型，检查空值
            non_empty_ratio = (data.notna() & (data != '')).sum() / len(data)
            return non_empty_ratio
    
    def _assess_stability(self, data: pd.Series) -> float:
        """评估数据稳定性"""
        if len(data) < 2:
            return 1.0
        
        if pd.api.types.is_numeric_dtype(data):
            # 计算变异系数
            mean = data.mean()
            std = data.std()
            
            if mean == 0:
                return 1.0 if std == 0 else 0.5
            
            cv = std / abs(mean)
            # 变异系数越小越稳定
            stability = max(0.0, 1.0 - cv)
            return stability
        else:
            # 非数值类型，检查类别分布稳定性
            value_counts = data.value_counts()
            if len(value_counts) == 0:
                return 1.0
            
            # 计算熵
            probs = value_counts / len(data)
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(len(value_counts))
            
            if max_entropy == 0:
                return 1.0
            
            # 归一化熵，熵越小越稳定
            normalized_entropy = entropy / max_entropy
            return 1.0 - normalized_entropy
    
    def _assess_uniqueness(self, data: pd.Series) -> float:
        """评估数据唯一性"""
        if len(data) == 0:
            return 0.0
        
        unique_ratio = data.nunique() / len(data)
        
        # 唯一性不是越高越好，适度即可
        # 理想情况下，唯一性在0.3-0.8之间
        if unique_ratio < 0.1:  # 常量或几乎常量
            return 0.3
        elif unique_ratio > 0.95:  # 几乎唯一（可能是ID列）
            return 0.5
        else:
            return 0.8 + (0.2 * (1 - abs(unique_ratio - 0.5) * 2))
    
    def _assess_correlation(self, feature: pd.Series, target: pd.Series) -> float:
        """评估特征与目标的相关性"""
        try:
            # 确保数据对齐
            aligned_feature, aligned_target = feature.align(target, join='inner')
            
            if len(aligned_feature) < 2:
                return 0.0
            
            if pd.api.types.is_numeric_dtype(aligned_feature) and \
               pd.api.types.is_numeric_dtype(aligned_target):
                # 计算皮尔逊相关系数
                correlation = aligned_feature.corr(aligned_target)
                return abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                # 非数值类型，返回中等评分
                return 0.5
        except Exception as e:
            logger.debug(f"计算相关性失败: {e}")
            return 0.0
    
    def _get_quality_level(self, score: float) -> str:
        """根据评分确定质量等级"""
        if score >= self.quality_thresholds['excellent']:
            return "优秀"
        elif score >= self.quality_thresholds['good']:
            return "良好"
        elif score >= self.quality_thresholds['fair']:
            return "一般"
        else:
            return "较差"
    
    def generate_quality_report(
        self,
        assessment_results: Dict[str, Any],
        task_id: str = ""
    ) -> Dict[str, Any]:
        """
        生成质量评估报告
        
        Args:
            assessment_results: 评估结果
            task_id: 任务ID
            
        Returns:
            质量报告
        """
        report = {
            'task_id': task_id,
            'report_time': datetime.now().isoformat(),
            'summary': {
                'total_features': len(assessment_results.get('feature_metrics', [])),
                'overall_score': assessment_results.get('overall_score', 0.0),
                'quality_distribution': assessment_results.get('quality_distribution', {})
            },
            'feature_details': assessment_results.get('feature_metrics', []),
            'recommendations': self._generate_recommendations(assessment_results)
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        distribution = results.get('quality_distribution', {})
        metrics = results.get('feature_metrics', [])
        
        # 基于质量分布生成建议
        if distribution.get('较差', 0) > 0:
            poor_count = distribution['较差']
            recommendations.append(f"有 {poor_count} 个特征质量较差，建议检查数据质量或重新计算")
        
        if distribution.get('一般', 0) > len(metrics) * 0.3:
            recommendations.append("较多特征质量一般，建议优化特征计算逻辑")
        
        # 检查低完整性特征
        low_completeness = [m for m in metrics if m.get('completeness', 1.0) < 0.9]
        if low_completeness:
            recommendations.append(f"{len(low_completeness)} 个特征存在缺失值，建议进行数据清洗")
        
        # 检查低稳定性特征
        low_stability = [m for m in metrics if m.get('stability', 1.0) < 0.5]
        if low_stability:
            recommendations.append(f"{len(low_stability)} 个特征稳定性较差，建议检查数据源")
        
        if not recommendations:
            recommendations.append("特征质量良好，继续保持")
        
        return recommendations


# 全局评估器实例
_assessor: Optional[FeatureQualityAssessor] = None


def get_feature_quality_assessor() -> FeatureQualityAssessor:
    """
    获取全局特征质量评估器实例
    
    Returns:
        特征质量评估器实例
    """
    global _assessor
    if _assessor is None:
        _assessor = FeatureQualityAssessor()
    return _assessor
