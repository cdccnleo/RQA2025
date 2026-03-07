
from ..core.common_mixins import ConfigComponentMixin
from typing import Dict, Any, List
"""异常检测功能"""


class AnomalyDetector(ConfigComponentMixin):
    """异常检测器"""

    def __init__(self, window_size: int = 20, threshold: float = 2.5):
        """初始化异常检测器"""
        super().__init__()
        self._init_component_attributes(enable_threading=True, enable_data=True)
        self.window_size = window_size
        self.threshold = threshold
        self._data_windows: Dict[str, List[float]] = {}
        self._baselines: Dict[str, float] = {}
        self._std_devs: Dict[str, float] = {}

    def update_baseline(self, metric_name: str, values: List[float]):
        """更新基线"""
        if len(values) >= self.window_size:
            window = values[-self.window_size:]
            self._baselines[metric_name] = sum(window) / len(window)
            variance = sum((x - self._baselines[metric_name]) ** 2 for x in window) / len(window)
            self._std_devs[metric_name] = variance ** 0.5

    def detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """检测异常"""
        if metric_name not in self._data_windows:
            self._data_windows[metric_name] = []

        self._data_windows[metric_name].append(value)

        # 保持窗口大小
        if len(self._data_windows[metric_name]) > self.window_size:
            self._data_windows[metric_name].pop(0)

        # 更新基线
        self.update_baseline(metric_name, self._data_windows[metric_name])

        # 检测异常
        if metric_name in self._baselines and metric_name in self._std_devs:
            baseline = self._baselines[metric_name]
            std_dev = self._std_devs[metric_name]

            if std_dev > 0:
                z_score = abs(value - baseline) / std_dev
                is_anomaly = z_score > self.threshold

                return {
                    "is_anomaly": is_anomaly,
                    "z_score": z_score,
                    "baseline": baseline,
                    "std_dev": std_dev,
                    "threshold": self.threshold
                }

        return {
            "is_anomaly": False,
            "z_score": 0.0,
            "baseline": value if len(self._data_windows[metric_name]) == 1 else 0.0,
            "std_dev": 0.0,
            "threshold": self.threshold
        }




