"""
健康趋势分析器

负责分析健康数据的趋势和变化。
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from ...core.shared_interfaces import ILogger, StandardLogger


class TrendAnalyzer:
    """健康趋势分析器"""
    
    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
    
    def analyze_health_trends(self, health_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析健康趋势"""
        if len(health_history) < 2:
            return {'trend': 'insufficient_data', 'direction': 'unknown', 'confidence': 0.0}

        scores = self._extract_scores(health_history)
        trend_info = self._calculate_trend_info(scores)
        return self._build_trend_result(trend_info, scores)

    def get_time_range(self, health_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取时间范围"""
        if not health_history:
            return {'start': None, 'end': None, 'duration_hours': 0}

        timestamps = self._extract_timestamps(health_history)
        if not timestamps:
            return {'start': None, 'end': None, 'duration_hours': 0}

        return self._calculate_time_range(timestamps)

    def calculate_average_score(self, health_history: List[Dict[str, Any]]) -> float:
        """计算平均健康评分"""
        scores = self._extract_scores(health_history)
        return sum(scores) / len(scores) if scores else 0.0

    def analyze_key_metrics_trends(self, health_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析关键指标趋势"""
        # 这里可以扩展为更详细的指标趋势分析
        return {
            'performance_trend': 'stable',
            'alert_trend': 'stable',
            'test_trend': 'stable',
            'details': {}
        }

    def _extract_scores(self, health_history: List[Dict[str, Any]]) -> List[float]:
        """提取评分数据"""
        return [h.get('overall_score', 0) for h in health_history]

    def _extract_timestamps(self, health_history: List[Dict[str, Any]]) -> List[float]:
        """提取时间戳数据"""
        return [h.get('evaluation_timestamp') for h in health_history if h.get('evaluation_timestamp')]

    def _calculate_trend_info(self, scores: List[float]) -> Dict[str, float]:
        """计算趋势信息"""
        if not scores:
            return {'recent_avg': 0, 'older_avg': 0, 'change': 0}
        
        recent_avg = sum(scores[-5:]) / min(5, len(scores))
        older_avg = sum(scores[:-5]) / max(1, len(scores) - 5) if len(scores) > 5 else recent_avg
        
        return {
            'recent_avg': recent_avg,
            'older_avg': older_avg,
            'change': recent_avg - older_avg
        }

    def _build_trend_result(self, trend_info: Dict[str, float], scores: List[float]) -> Dict[str, Any]:
        """构建趋势结果"""
        recent_avg = trend_info['recent_avg']
        older_avg = trend_info['older_avg']
        change = trend_info['change']

        if abs(change) < 0.1:
            trend = 'stable'
            direction = 'stable'
            confidence = 0.8
        elif change > 0:
            trend = 'improving'
            direction = 'up'
            confidence = min(1.0, change * 2)
        else:
            trend = 'degrading'
            direction = 'down'
            confidence = min(1.0, abs(change) * 2)

        return {
            'trend': trend,
            'direction': direction,
            'confidence': confidence,
            'recent_average': recent_avg,
            'older_average': older_avg,
            'change': change
        }

    def _calculate_time_range(self, timestamps: List[float]) -> Dict[str, Any]:
        """计算时间范围"""
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration_hours = (end_time - start_time) / 3600

        return {
            'start': datetime.fromtimestamp(start_time).isoformat(),
            'end': datetime.fromtimestamp(end_time).isoformat(),
            'duration_hours': duration_hours
        }
