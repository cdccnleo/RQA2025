#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量趋势分析和预测模块

提供质量评分的历史趋势分析和未来预测：
- 质量评分趋势分析
- 质量预测和预警
- 质量异常检测
- 质量改进建议
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class QualityTrendAnalyzer:
    """
    质量趋势分析器
    
    分析质量评分的历史趋势并进行预测
    """
    
    def __init__(self):
        self.min_data_points = 3  # 最小数据点数
        self.prediction_days = 7  # 预测天数
        self.anomaly_threshold = 2.0  # 异常检测阈值（标准差倍数）
    
    def analyze_trend(
        self,
        quality_history: List[Dict[str, Any]],
        feature_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析质量趋势
        
        Args:
            quality_history: 质量历史数据
            feature_name: 特定特征名称（可选）
            
        Returns:
            趋势分析结果
        """
        try:
            if len(quality_history) < self.min_data_points:
                return {
                    'success': False,
                    'message': f'数据点不足，需要至少{self.min_data_points}个数据点',
                    'data_points': len(quality_history)
                }
            
            # 提取评分数据
            scores = [h.get('overall_score', 0) for h in quality_history]
            timestamps = [h.get('timestamp', 0) for h in quality_history]
            
            # 基本统计
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            # 趋势方向分析
            trend_direction = self._analyze_trend_direction(scores)
            
            # 变化率分析
            change_rate = self._calculate_change_rate(scores)
            
            # 波动性分析
            volatility = self._calculate_volatility(scores)
            
            # 异常检测
            anomalies = self._detect_anomalies(scores, timestamps)
            
            # 预测
            prediction = self._predict_future(scores)
            
            result = {
                'success': True,
                'feature_name': feature_name,
                'analysis_time': datetime.now().isoformat(),
                'data_points': len(quality_history),
                'statistics': {
                    'mean': float(mean_score),
                    'std': float(std_score),
                    'min': float(min_score),
                    'max': float(max_score),
                    'current': float(scores[-1]) if scores else 0
                },
                'trend': {
                    'direction': trend_direction,
                    'change_rate': float(change_rate),
                    'volatility': float(volatility),
                    'stability': 'stable' if volatility < 0.1 else 'volatile'
                },
                'anomalies': anomalies,
                'prediction': prediction,
                'alerts': self._generate_alerts(scores, trend_direction, anomalies)
            }
            
            logger.info(f"质量趋势分析完成: {feature_name or '整体'}, "
                       f"趋势: {trend_direction}, 数据点: {len(quality_history)}")
            
            return result
            
        except Exception as e:
            logger.error(f"质量趋势分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_trend_direction(self, scores: List[float]) -> str:
        """分析趋势方向"""
        if len(scores) < 2:
            return 'stable'
        
        # 使用线性回归分析趋势
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        # 根据斜率判断趋势
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_change_rate(self, scores: List[float]) -> float:
        """计算变化率"""
        if len(scores) < 2:
            return 0.0
        
        # 计算最近的变化率
        recent_scores = scores[-5:] if len(scores) >= 5 else scores
        if len(recent_scores) < 2:
            return 0.0
        
        first_score = recent_scores[0]
        last_score = recent_scores[-1]
        
        if first_score == 0:
            return 0.0
        
        change_rate = (last_score - first_score) / first_score
        return change_rate
    
    def _calculate_volatility(self, scores: List[float]) -> float:
        """计算波动性"""
        if len(scores) < 2:
            return 0.0
        
        # 计算变异系数
        mean = np.mean(scores)
        std = np.std(scores)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        return cv
    
    def _detect_anomalies(
        self,
        scores: List[float],
        timestamps: List[float]
    ) -> List[Dict[str, Any]]:
        """检测异常值"""
        anomalies = []
        
        if len(scores) < 3:
            return anomalies
        
        mean = np.mean(scores)
        std = np.std(scores)
        
        if std == 0:
            return anomalies
        
        for i, (score, timestamp) in enumerate(zip(scores, timestamps)):
            z_score = (score - mean) / std
            
            if abs(z_score) > self.anomaly_threshold:
                anomalies.append({
                    'index': i,
                    'timestamp': timestamp,
                    'score': float(score),
                    'z_score': float(z_score),
                    'type': 'high' if z_score > 0 else 'low'
                })
        
        return anomalies
    
    def _predict_future(self, scores: List[float]) -> Dict[str, Any]:
        """预测未来质量评分"""
        try:
            if len(scores) < 3:
                return {
                    'success': False,
                    'message': '数据点不足，无法预测'
                }
            
            # 简单线性预测
            x = np.arange(len(scores))
            y = np.array(scores)
            
            # 线性回归
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            
            # 预测未来7天
            future_predictions = []
            for i in range(1, self.prediction_days + 1):
                future_x = len(scores) + i
                predicted_score = slope * future_x + intercept
                # 限制在合理范围内
                predicted_score = max(0.0, min(1.0, predicted_score))
                
                future_predictions.append({
                    'day': i,
                    'predicted_score': float(predicted_score),
                    'confidence': max(0.0, 1.0 - i * 0.1)  # 置信度随时间递减
                })
            
            # 计算预测趋势
            if len(future_predictions) >= 2:
                first_pred = future_predictions[0]['predicted_score']
                last_pred = future_predictions[-1]['predicted_score']
                
                if last_pred > first_pred + 0.05:
                    predicted_trend = 'improving'
                elif last_pred < first_pred - 0.05:
                    predicted_trend = 'declining'
                else:
                    predicted_trend = 'stable'
            else:
                predicted_trend = 'unknown'
            
            return {
                'success': True,
                'predictions': future_predictions,
                'predicted_trend': predicted_trend,
                'slope': float(slope),
                'method': 'linear_regression'
            }
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_alerts(
        self,
        scores: List[float],
        trend_direction: str,
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """生成预警信息"""
        alerts = []
        
        if not scores:
            return alerts
        
        current_score = scores[-1]
        
        # 质量下降预警
        if trend_direction == 'declining':
            alerts.append({
                'level': 'warning',
                'type': 'quality_declining',
                'message': '质量评分呈下降趋势，建议检查特征计算逻辑',
                'current_score': float(current_score)
            })
        
        # 低质量预警
        if current_score < 0.5:
            alerts.append({
                'level': 'critical',
                'type': 'low_quality',
                'message': f'当前质量评分较低 ({current_score:.2f})，建议优化特征',
                'current_score': float(current_score)
            })
        
        # 异常预警
        if anomalies:
            alerts.append({
                'level': 'warning',
                'type': 'anomaly_detected',
                'message': f'检测到{len(anomalies)}个异常值，建议检查数据',
                'anomaly_count': len(anomalies)
            })
        
        # 高波动性预警
        if len(scores) >= 5:
            recent_scores = scores[-5:]
            volatility = self._calculate_volatility(recent_scores)
            if volatility > 0.2:
                alerts.append({
                    'level': 'info',
                    'type': 'high_volatility',
                    'message': '质量评分波动较大，建议检查数据源稳定性',
                    'volatility': float(volatility)
                })
        
        return alerts
    
    def generate_improvement_suggestions(
        self,
        trend_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        生成改进建议
        
        Args:
            trend_analysis: 趋势分析结果
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        if not trend_analysis.get('success'):
            return suggestions
        
        trend = trend_analysis.get('trend', {})
        stats = trend_analysis.get('statistics', {})
        alerts = trend_analysis.get('alerts', [])
        
        # 基于趋势的建议
        if trend.get('direction') == 'declining':
            suggestions.append(
                "质量评分呈下降趋势，建议：\n"
                "1. 检查最近的数据源是否有变化\n"
                "2. 重新评估特征计算逻辑\n"
                "3. 增加数据清洗步骤"
            )
        
        # 基于波动性的建议
        if trend.get('stability') == 'volatile':
            suggestions.append(
                "质量评分波动较大，建议：\n"
                "1. 检查数据采集的稳定性\n"
                "2. 增加数据验证步骤\n"
                "3. 考虑使用更稳定的特征"
            )
        
        # 基于当前评分的建议
        current_score = stats.get('current', 0)
        if current_score < 0.5:
            suggestions.append(
                f"当前质量评分较低 ({current_score:.2f})，建议：\n"
                "1. 检查特征缺失值情况\n"
                "2. 优化特征计算参数\n"
                "3. 考虑更换数据源"
            )
        elif current_score < 0.7:
            suggestions.append(
                f"质量评分有提升空间 ({current_score:.2f})，建议：\n"
                "1. 优化特征选择策略\n"
                "2. 增加特征多样性\n"
                "3. 定期监控质量指标"
            )
        
        # 基于预警的建议
        for alert in alerts:
            if alert['type'] == 'anomaly_detected':
                suggestions.append(
                    f"检测到{alert.get('anomaly_count', 0)}个异常值，建议：\n"
                    "1. 检查异常时间点的数据\n"
                    "2. 增加异常值处理逻辑\n"
                    "3. 设置数据质量阈值"
                )
        
        if not suggestions:
            suggestions.append(
                "质量评分表现良好，建议继续保持当前的数据处理流程，"
                "并定期监控质量指标。"
            )
        
        return suggestions


# 全局分析器实例
_analyzer: Optional[QualityTrendAnalyzer] = None


def get_quality_trend_analyzer() -> QualityTrendAnalyzer:
    """
    获取全局质量趋势分析器实例
    
    Returns:
        质量趋势分析器实例
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = QualityTrendAnalyzer()
    return _analyzer
