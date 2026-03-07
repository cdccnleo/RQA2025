
from ..core.common_mixins import ConfigComponentMixin
from typing import Dict, Any, List
"""趋势分析功能"""


class TrendAnalyzer(ConfigComponentMixin):
    """趋势分析器"""

    def __init__(self, window_size: int = 50):
        """初始化趋势分析器"""
        super().__init__()
        self._init_component_attributes(enable_threading=True, enable_data=True)
        self.window_size = window_size
        self._data_series: Dict[str, List[float]] = {}

    def add_data_point(self, metric_name: str, value: float):
        """添加数据点"""
        if metric_name not in self._data_series:
            self._data_series[metric_name] = []

        self._data_series[metric_name].append(value)

        # 保持窗口大小
        if len(self._data_series[metric_name]) > self.window_size:
            self._data_series[metric_name].pop(0)

    def analyze_trend(self, metric_name: str) -> Dict[str, Any]:
        """分析趋势"""
        if metric_name not in self._data_series:
            return {"trend": "insufficient_data", "slope": 0.0, "confidence": 0.0}

        values = self._data_series[metric_name]
        if len(values) < 10:  # 需要至少10个数据点
            return {"trend": "insufficient_data", "slope": 0.0, "confidence": 0.0}

        # 计算线性回归
        n = len(values)
        x = list(range(n))

        # 计算斜率和截距
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # 计算R²值作为置信度
        y_mean = sum_y / n
        ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, values))
        ss_tot = sum((yi - y_mean) ** 2 for yi in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 确定趋势方向
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "trend": trend,
            "slope": slope,
            "confidence": r_squared,
            "intercept": intercept,
            "data_points": n
        }




