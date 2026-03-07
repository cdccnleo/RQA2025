
from ..core.common_mixins import ConfigComponentMixin
from typing import Dict, Any, List
"""性能预测功能"""


class PerformancePredictor(ConfigComponentMixin):
    """性能预测器"""

    def __init__(self, prediction_window: int = 10):
        """初始化性能预测器"""
        super().__init__()
        self._init_component_attributes(enable_threading=True, enable_data=True)
        self.prediction_window = prediction_window
        self._historical_data: Dict[str, List[float]] = {}

    def add_historical_data(self, metric_name: str, value: float):
        """添加历史数据"""
        if metric_name not in self._historical_data:
            self._historical_data[metric_name] = []

        self._historical_data[metric_name].append(value)

        # 保持合理的历史数据量
        if len(self._historical_data[metric_name]) > 1000:
            self._historical_data[metric_name] = self._historical_data[metric_name][-500:]

    def predict_next_value(self, metric_name: str) -> Dict[str, Any]:
        """预测下一个值"""
        if metric_name not in self._historical_data:
            return {"prediction": None, "confidence": 0.0, "method": "insufficient_data"}

        values = self._historical_data[metric_name]
        if len(values) < 5:
            return {"prediction": None, "confidence": 0.0, "method": "insufficient_data"}

        # 使用简单移动平均进行预测
        window_size = min(10, len(values))
        recent_values = values[-window_size:]
        prediction = sum(recent_values) / len(recent_values)

        # 计算置信度（基于数据的稳定性）
        if len(recent_values) >= 3:
            mean = sum(recent_values) / len(recent_values)
            variance = sum((x - mean) ** 2 for x in recent_values) / len(recent_values)
            std_dev = variance ** 0.5
            cv = std_dev / mean if mean > 0 else 0  # 变异系数
            confidence = max(0, 1 - cv)  # 稳定性越好，置信度越高
        else:
            confidence = 0.5

        return {
            "prediction": prediction,
            "confidence": confidence,
            "method": "moving_average",
            "window_size": window_size,
            "historical_points": len(values)
        }

    def predict_trend(self, metric_name: str) -> Dict[str, Any]:
        """预测趋势"""
        if metric_name not in self._historical_data:
            return {"trend": "unknown", "confidence": 0.0}

        values = self._historical_data[metric_name]
        if len(values) < 10:
            return {"trend": "insufficient_data", "confidence": 0.0}

        # 计算最近趋势
        recent = values[-20:]  # 最近20个点
        if len(recent) < 5:
            return {"trend": "insufficient_data", "confidence": 0.0}

        # 计算斜率
        x = list(range(len(recent)))
        slope = sum((xi - sum(x)/len(x)) * (yi - sum(recent)/len(recent))
                    for xi, yi in zip(x, recent)) / sum((xi - sum(x)/len(x)) ** 2 for xi in x)

        if abs(slope) < 0.001:
            trend = "stable"
            confidence = 0.8
        elif slope > 0:
            trend = "increasing"
            confidence = min(1.0, abs(slope) * 10)  # 斜率越大，置信度越高
        else:
            trend = "decreasing"
            confidence = min(1.0, abs(slope) * 10)

        return {
            "trend": trend,
            "confidence": confidence,
            "slope": slope,
            "data_points": len(recent)
        }




