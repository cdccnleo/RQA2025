"""数据质量监控模块

提供数据质量监控功能，包括数据完整性、准确性、一致性检查
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum


class QualityLevel(Enum):
    """质量等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityMetrics:
    """质量指标数据类"""
    completeness: float  # 完整性
    accuracy: float      # 准确性
    consistency: float   # 一致性
    timeliness: float    # 及时性
    validity: float      # 有效性
    overall_score: float # 总体评分
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'timeliness': self.timeliness,
            'validity': self.validity,
            'overall_score': self.overall_score
        }


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化监控器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.thresholds = self.config.get('thresholds', {
            'completeness': 0.95,
            'accuracy': 0.90,
            'consistency': 0.85,
            'timeliness': 0.80,
            'validity': 0.90
        })
        self.quality_history = []
    
    def check_completeness(self, data: pd.DataFrame) -> float:
        """检查数据完整性
        
        Args:
            data: 待检查的数据
            
        Returns:
            完整性得分 (0-1)
        """
        if data.empty:
            return 0.0
        
        # 计算非空值比例
        total_cells = data.size
        non_null_cells = data.count().sum()
        completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        return completeness
    
    def check_accuracy(self, data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None) -> float:
        """检查数据准确性
        
        Args:
            data: 待检查的数据
            reference_data: 参考数据（可选）
            
        Returns:
            准确性得分 (0-1)
        """
        if data.empty:
            return 0.0
        
        # 如果没有参考数据，使用基本规则检查
        if reference_data is None:
            # 检查数值列的合理性
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            accuracy_scores = []
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) == 0:
                    continue
                
                # 检查异常值（使用3倍标准差）
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    outliers = col_data[abs(col_data - mean_val) > 3 * std_val]
                    accuracy_score = 1 - (len(outliers) / len(col_data))
                    accuracy_scores.append(accuracy_score)
            
            return np.mean(accuracy_scores) if accuracy_scores else 1.0
        else:
            # 与参考数据比较
            common_cols = set(data.columns) & set(reference_data.columns)
            if not common_cols:
                return 0.0
            
            accuracy_scores = []
            for col in common_cols:
                if col in data.columns and col in reference_data.columns:
                    # 计算相关系数作为准确性指标
                    corr = data[col].corr(reference_data[col])
                    if pd.notna(corr):
                        accuracy_scores.append(abs(corr))
            
            return np.mean(accuracy_scores) if accuracy_scores else 0.0
    
    def check_consistency(self, data: pd.DataFrame) -> float:
        """检查数据一致性
        
        Args:
            data: 待检查的数据
            
        Returns:
            一致性得分 (0-1)
        """
        if data.empty:
            return 0.0
        
        consistency_scores = []
        
        # 检查数据类型一致性
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            # 检查数据类型是否一致
            dtype_consistency = 1.0
            if data[col].dtype == 'object':
                # 对于对象类型，检查是否可以转换为数值
                try:
                    pd.to_numeric(col_data)
                    dtype_consistency = 1.0
                except:
                    dtype_consistency = 0.8  # 对象类型但无法转换为数值
            
            consistency_scores.append(dtype_consistency)
        
        # 检查时间序列的单调性（如果有时间列）
        time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in time_cols:
            if col in data.columns:
                time_data = pd.to_datetime(data[col], errors='coerce').dropna()
                if len(time_data) > 1:
                    # 检查时间是否单调递增
                    is_monotonic = time_data.is_monotonic_increasing
                    consistency_scores.append(1.0 if is_monotonic else 0.5)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def check_timeliness(self, data: pd.DataFrame, expected_frequency: str = '1D') -> float:
        """检查数据及时性
        
        Args:
            data: 待检查的数据
            expected_frequency: 期望的数据频率
            
        Returns:
            及时性得分 (0-1)
        """
        if data.empty:
            return 0.0
        
        # 查找时间列
        time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if not time_cols:
            return 1.0  # 没有时间列，无法检查及时性
        
        time_col = time_cols[0]
        try:
            time_data = pd.to_datetime(data[time_col], errors='coerce').dropna()
            if len(time_data) < 2:
                return 0.0
            
            # 计算时间间隔
            time_diff = time_data.diff().dropna()
            if len(time_diff) == 0:
                return 0.0
            
            # 计算期望间隔（秒）
            if expected_frequency == '1D':
                expected_interval = pd.Timedelta(days=1).total_seconds()
            elif expected_frequency == '1H':
                expected_interval = pd.Timedelta(hours=1).total_seconds()
            else:
                expected_interval = pd.Timedelta(days=1).total_seconds()
            
            # 计算实际间隔与期望间隔的偏差
            actual_intervals = time_diff.dt.total_seconds()
            deviations = abs(actual_intervals - expected_interval) / expected_interval
            timeliness_score = 1 - np.mean(deviations)
            
            return max(0.0, min(1.0, timeliness_score))
        except:
            return 0.5  # 时间解析失败
    
    def check_validity(self, data: pd.DataFrame) -> float:
        """检查数据有效性
        
        Args:
            data: 待检查的数据
            
        Returns:
            有效性得分 (0-1)
        """
        if data.empty:
            return 0.0
        
        validity_scores = []
        
        for col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            # 检查数值列的有效性
            if pd.api.types.is_numeric_dtype(data[col]):
                # 检查是否有无穷大或NaN
                invalid_count = np.isinf(col_data).sum() + np.isnan(col_data).sum()
                validity_score = 1 - (invalid_count / len(col_data)) if len(col_data) > 0 else 0.0
                validity_scores.append(validity_score)
            else:
                # 对于非数值列，检查是否有空字符串
                empty_strings = (col_data == '').sum()
                validity_score = 1 - (empty_strings / len(col_data)) if len(col_data) > 0 else 0.0
                validity_scores.append(validity_score)
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """计算总体质量评分
        
        Args:
            metrics: 质量指标
            
        Returns:
            总体评分 (0-1)
        """
        weights = self.config.get('weights', {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.15
        })
        
        overall_score = (
            metrics.completeness * weights['completeness'] +
            metrics.accuracy * weights['accuracy'] +
            metrics.consistency * weights['consistency'] +
            metrics.timeliness * weights['timeliness'] +
            metrics.validity * weights['validity']
        )
        
        return overall_score
    
    def assess_quality(self, data: pd.DataFrame, **kwargs) -> QualityMetrics:
        """评估数据质量
        
        Args:
            data: 待评估的数据
            **kwargs: 其他参数
            
        Returns:
            质量指标
        """
        completeness = self.check_completeness(data)
        accuracy = self.check_accuracy(data, kwargs.get('reference_data'))
        consistency = self.check_consistency(data)
        timeliness = self.check_timeliness(data, kwargs.get('expected_frequency', '1D'))
        validity = self.check_validity(data)
        
        metrics = QualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            overall_score=0.0  # 临时值
        )
        
        # 计算总体评分
        metrics.overall_score = self.calculate_overall_score(metrics)
        
        # 记录质量历史
        self.quality_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.to_dict()
        })
        
        return metrics
    
    def get_quality_level(self, metrics: QualityMetrics) -> QualityLevel:
        """获取质量等级
        
        Args:
            metrics: 质量指标
            
        Returns:
            质量等级
        """
        score = metrics.overall_score
        
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.FAIR
        elif score >= 0.50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def get_quality_history(self, days: int = 30) -> List[Dict]:
        """获取质量历史记录
        
        Args:
            days: 天数限制
            
        Returns:
            质量历史记录列表
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            record for record in self.quality_history
            if record['timestamp'] >= cutoff_date
        ]
    
    def generate_report(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """生成质量报告
        
        Args:
            data: 待评估的数据
            **kwargs: 其他参数
            
        Returns:
            质量报告字典
        """
        metrics = self.assess_quality(data, **kwargs)
        quality_level = self.get_quality_level(metrics)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'metrics': metrics.to_dict(),
            'quality_level': quality_level.value,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """生成改进建议
        
        Args:
            metrics: 质量指标
            
        Returns:
            建议列表
        """
        recommendations = []
        
        if metrics.completeness < self.thresholds['completeness']:
            recommendations.append("数据完整性不足，建议检查数据源和采集过程")
        
        if metrics.accuracy < self.thresholds['accuracy']:
            recommendations.append("数据准确性需要改进，建议增加数据验证规则")
        
        if metrics.consistency < self.thresholds['consistency']:
            recommendations.append("数据一致性存在问题，建议统一数据格式和标准")
        
        if metrics.timeliness < self.thresholds['timeliness']:
            recommendations.append("数据及时性不足，建议优化数据采集频率")
        
        if metrics.validity < self.thresholds['validity']:
            recommendations.append("数据有效性需要提升，建议加强数据清洗")
        
        if not recommendations:
            recommendations.append("数据质量良好，继续保持")
        
        return recommendations 