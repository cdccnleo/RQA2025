"""
特征质量趋势分析
用于追踪特征质量随时间的变化
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """趋势方向"""
    IMPROVING = "improving"    # 改善
    STABLE = "stable"          # 稳定
    DECLINING = "declining"    # 下降
    UNKNOWN = "unknown"        # 未知


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    feature_id: int
    feature_name: str
    trend: TrendDirection
    trend_strength: float  # 趋势强度 (0-1)
    current_score: float
    previous_score: float
    change_percent: float
    volatility: float      # 波动性
    anomalies: List[Dict]  # 异常点
    statistics: Dict       # 统计信息
    analysis_period_days: int
    analyzed_at: datetime


class QualityTrendAnalyzer:
    """质量趋势分析器"""
    
    # 趋势阈值配置
    TREND_THRESHOLDS = {
        'improving': 0.02,    # 改善阈值 (2%)
        'declining': -0.02,   # 下降阈值 (-2%)
        'stable_range': 0.02  # 稳定范围 (±2%)
    }
    
    # 异常检测阈值
    ANOMALY_THRESHOLDS = {
        'z_score': 2.0,       # Z-score阈值
        'change_percent': 0.1  # 变化百分比阈值 (10%)
    }
    
    def __init__(self):
        self._min_data_points = 3  # 最小数据点数量
    
    def analyze_trends(
        self,
        quality_history: List[Dict],
        feature_name: str = "Unknown"
    ) -> Optional[TrendAnalysis]:
        """
        分析质量趋势
        
        Args:
            quality_history: 质量历史记录列表
                [{"recorded_at": datetime, "quality_score": float}, ...]
            feature_name: 特征名称
        
        Returns:
            趋势分析结果
        """
        if len(quality_history) < self._min_data_points:
            logger.warning(f"数据点不足 ({len(quality_history)} < {self._min_data_points})，无法分析趋势")
            return None
        
        # 按时间排序
        sorted_history = sorted(quality_history, key=lambda x: x['recorded_at'])
        
        # 提取分数列表
        scores = [record['quality_score'] for record in sorted_history]
        timestamps = [record['recorded_at'] for record in sorted_history]
        
        # 计算趋势
        trend, trend_strength = self._calculate_trend(scores)
        
        # 检测异常
        anomalies = self._detect_anomalies(scores, timestamps)
        
        # 计算统计信息
        statistics = self._calculate_statistics(scores)
        
        # 计算波动性
        volatility = self._calculate_volatility(scores)
        
        # 计算变化百分比
        current_score = scores[-1]
        previous_score = scores[0]
        change_percent = (current_score - previous_score) / previous_score if previous_score > 0 else 0
        
        # 计算分析周期
        analysis_period_days = (timestamps[-1] - timestamps[0]).days if len(timestamps) >= 2 else 0
        
        analysis = TrendAnalysis(
            feature_id=sorted_history[-1].get('feature_id', 0),
            feature_name=feature_name,
            trend=trend,
            trend_strength=trend_strength,
            current_score=current_score,
            previous_score=previous_score,
            change_percent=change_percent,
            volatility=volatility,
            anomalies=anomalies,
            statistics=statistics,
            analysis_period_days=analysis_period_days,
            analyzed_at=datetime.now()
        )
        
        logger.info(f"特征 {feature_name} 趋势分析完成: {trend.value}, "
                   f"强度: {trend_strength:.2f}, 变化: {change_percent:.1%}")
        
        return analysis
    
    def _calculate_trend(self, scores: List[float]) -> Tuple[TrendDirection, float]:
        """
        计算趋势方向和强度
        
        使用线性回归计算趋势
        """
        if len(scores) < 2:
            return TrendDirection.UNKNOWN, 0.0
        
        # 使用简单线性回归
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        # 归一化斜率到 [-1, 1] 范围
        max_possible_slope = 1.0 / len(scores)  # 假设最大可能变化是1.0
        normalized_slope = np.clip(slope / max_possible_slope, -1, 1)
        
        # 确定趋势方向
        if normalized_slope > self.TREND_THRESHOLDS['improving']:
            trend = TrendDirection.IMPROVING
        elif normalized_slope < self.TREND_THRESHOLDS['declining']:
            trend = TrendDirection.DECLINING
        else:
            trend = TrendDirection.STABLE
        
        # 计算趋势强度 (0-1)
        trend_strength = abs(normalized_slope)
        
        return trend, trend_strength
    
    def _detect_anomalies(
        self,
        scores: List[float],
        timestamps: List[datetime]
    ) -> List[Dict]:
        """
        检测异常点
        
        使用Z-score方法和变化百分比方法
        """
        anomalies = []
        
        if len(scores) < 3:
            return anomalies
        
        # 计算均值和标准差
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return anomalies
        
        # Z-score方法
        for i, (score, timestamp) in enumerate(zip(scores, timestamps)):
            z_score = (score - mean) / std
            
            if abs(z_score) > self.ANOMALY_THRESHOLDS['z_score']:
                anomalies.append({
                    'index': i,
                    'timestamp': timestamp.isoformat(),
                    'score': score,
                    'z_score': z_score,
                    'type': 'z_score_anomaly',
                    'severity': 'high' if abs(z_score) > 3 else 'medium'
                })
        
        # 变化百分比方法
        for i in range(1, len(scores)):
            change_percent = abs(scores[i] - scores[i-1]) / scores[i-1] if scores[i-1] > 0 else 0
            
            if change_percent > self.ANOMALY_THRESHOLDS['change_percent']:
                # 检查是否已经被Z-score方法检测到
                already_detected = any(a['index'] == i for a in anomalies)
                
                if not already_detected:
                    anomalies.append({
                        'index': i,
                        'timestamp': timestamps[i].isoformat(),
                        'score': scores[i],
                        'previous_score': scores[i-1],
                        'change_percent': change_percent,
                        'type': 'change_anomaly',
                        'severity': 'high' if change_percent > 0.2 else 'medium'
                    })
        
        return sorted(anomalies, key=lambda x: x['timestamp'])
    
    def _calculate_statistics(self, scores: List[float]) -> Dict:
        """计算统计信息"""
        if not scores:
            return {}
        
        return {
            'count': len(scores),
            'mean': round(np.mean(scores), 3),
            'median': round(np.median(scores), 3),
            'std': round(np.std(scores), 3),
            'min': round(min(scores), 3),
            'max': round(max(scores), 3),
            'range': round(max(scores) - min(scores), 3)
        }
    
    def _calculate_volatility(self, scores: List[float]) -> float:
        """
        计算波动性
        
        使用变异系数 (Coefficient of Variation)
        """
        if len(scores) < 2:
            return 0.0
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        return round(min(cv, 1.0), 3)  # 限制在0-1范围内
    
    def generate_trend_report(
        self,
        analyses: List[TrendAnalysis]
    ) -> Dict:
        """
        生成趋势报告
        
        Args:
            analyses: 多个特征的趋势分析结果
        
        Returns:
            趋势报告
        """
        if not analyses:
            return {
                'summary': '无数据',
                'total_features': 0,
                'trend_distribution': {},
                'recommendations': []
            }
        
        # 统计趋势分布
        trend_counts = {}
        for analysis in analyses:
            trend_value = analysis.trend.value
            trend_counts[trend_value] = trend_counts.get(trend_value, 0) + 1
        
        # 识别需要关注的特征
        concerning_features = [
            {
                'feature_id': a.feature_id,
                'feature_name': a.feature_name,
                'trend': a.trend.value,
                'current_score': a.current_score,
                'reason': self._get_concern_reason(a)
            }
            for a in analyses
            if a.trend == TrendDirection.DECLINING or a.current_score < 0.7
        ]
        
        # 生成建议
        recommendations = self._generate_recommendations(analyses)
        
        return {
            'summary': f"分析了 {len(analyses)} 个特征的质量趋势",
            'total_features': len(analyses),
            'trend_distribution': trend_counts,
            'concerning_features': concerning_features,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_concern_reason(self, analysis: TrendAnalysis) -> str:
        """获取关注原因"""
        reasons = []
        
        if analysis.trend == TrendDirection.DECLINING:
            reasons.append(f"质量下降趋势 ({analysis.change_percent:.1%})")
        
        if analysis.current_score < 0.5:
            reasons.append("质量评分严重偏低")
        elif analysis.current_score < 0.7:
            reasons.append("质量评分偏低")
        
        if analysis.volatility > 0.3:
            reasons.append("质量波动较大")
        
        if len(analysis.anomalies) > 0:
            reasons.append(f"发现 {len(analysis.anomalies)} 个异常点")
        
        return "; ".join(reasons) if reasons else "一般关注"
    
    def _generate_recommendations(self, analyses: List[TrendAnalysis]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 统计各类问题
        declining_count = sum(1 for a in analyses if a.trend == TrendDirection.DECLINING)
        low_quality_count = sum(1 for a in analyses if a.current_score < 0.7)
        high_volatility_count = sum(1 for a in analyses if a.volatility > 0.3)
        
        # 生成建议
        if declining_count > 0:
            recommendations.append(
                f"有 {declining_count} 个特征质量呈下降趋势，建议检查数据源和计算逻辑"
            )
        
        if low_quality_count > 0:
            recommendations.append(
                f"有 {low_quality_count} 个特征质量评分偏低，建议优化特征工程流程"
            )
        
        if high_volatility_count > 0:
            recommendations.append(
                f"有 {high_volatility_count} 个特征质量波动较大，建议检查数据稳定性"
            )
        
        if not recommendations:
            recommendations.append("整体质量状况良好，继续保持")
        
        return recommendations


# 全局趋势分析器实例
_trend_analyzer: Optional[QualityTrendAnalyzer] = None


def get_trend_analyzer() -> QualityTrendAnalyzer:
    """获取全局趋势分析器实例"""
    global _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = QualityTrendAnalyzer()
    return _trend_analyzer


def analyze_quality_trends(
    quality_history: List[Dict],
    feature_name: str = "Unknown"
) -> Optional[TrendAnalysis]:
    """
    分析质量趋势的便捷函数
    
    Args:
        quality_history: 质量历史记录列表
        feature_name: 特征名称
    
    Returns:
        趋势分析结果
    """
    analyzer = get_trend_analyzer()
    return analyzer.analyze_trends(quality_history, feature_name)
