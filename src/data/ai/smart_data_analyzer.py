"""
轻量可测的数据分析工具
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SummaryStats:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float


class SmartDataAnalyzer:
    """
    最小实现，提供确定性的基本统计、简单趋势与异常点检测
    - 不依赖外部数值库，便于稳定单测
    """

    def summarize(self, series: List[float]) -> SummaryStats:
        if not isinstance(series, list):
            raise TypeError("series must be a list")
        n = len(series)
        if n == 0:
            return SummaryStats(0, 0.0, 0.0, 0.0, 0.0)
        s = sum(series)
        mean = s / n
        # 使用总体标准差，提升对极端点的敏感性
        if n > 1:
            var = sum((x - mean) ** 2 for x in series) / n
            std = var ** 0.5
        else:
            std = 0.0
        return SummaryStats(n, mean, std, min(series), max(series))

    def detect_outliers(self, series: List[float], z_threshold: float = 3.0) -> List[int]:
        """
        基于 z-score 的简单异常检测；返回异常点索引
        """
        stats = self.summarize(series)
        if stats.count == 0 or stats.std == 0.0:
            return []
        outliers = []
        for i, x in enumerate(series):
            z = abs((x - stats.mean) / stats.std)
            if z >= z_threshold:
                outliers.append(i)
        return outliers

    def compute_trend(self, series: List[float]) -> str:
        """
        极简趋势判定：
        - 均值右半段 > 左半段 => up
        - 均值右半段 < 左半段 => down
        - 其他 => flat
        """
        n = len(series)
        if n < 2:
            return "flat"
        mid = n // 2
        left_mean = sum(series[:mid]) / max(1, mid)
        right_mean = sum(series[mid:]) / max(1, n - mid)
        eps = 1e-9
        if right_mean - left_mean > eps:
            return "up"
        if left_mean - right_mean > eps:
            return "down"
        return "flat"

    def analyze(self, series: List[float], z_threshold: float = 3.0) -> Dict[str, Optional[object]]:
        """
        组合分析：统计、趋势、异常
        """
        stats = self.summarize(series)
        return {
            "stats": stats,
            "trend": self.compute_trend(series),
            "outliers": self.detect_outliers(series, z_threshold=z_threshold),
        }


__all__ = ["SmartDataAnalyzer", "SummaryStats"]


