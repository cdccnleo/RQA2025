import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """漂移类型枚举"""
    DATA_DRIFT = auto()      # 数据漂移
    CONCEPT_DRIFT = auto()   # 概念漂移
    MODEL_DECAY = auto()     # 模型衰减

class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = auto()       # 信息
    WARNING = auto()    # 警告
    CRITICAL = auto()   # 严重

@dataclass
class ModelPerformance:
    """模型性能数据结构"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    log_loss: float

@dataclass
class DriftDetectionResult:
    """漂移检测结果"""
    drift_type: DriftType
    drift_score: float
    p_value: float
    is_drifted: bool
    timestamp: datetime

class ModelMonitor:
    """模型监控器"""

    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.drift_history = []
        self.performance_history = []

    def check_performance(self, y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None) -> ModelPerformance:
        """检查模型性能"""
        acc = accuracy_score(y_true, y_pred)
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0
        log_loss = self._calculate_log_loss(y_true, y_prob) if y_prob is not None else 0

        perf = ModelPerformance(
            accuracy=acc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            log_loss=log_loss
        )
        self.performance_history.append(perf)
        return perf

    def detect_drift(self, current_data: pd.DataFrame,
                   features: List[str]) -> DriftDetectionResult:
        """检测数据漂移"""
        # KS检验检测分布变化
        drift_scores = {}
        p_values = {}

        for feature in features:
            ref = self.reference_data[feature]
            curr = current_data[feature]
            stat, p = stats.ks_2samp(ref, curr)
            drift_scores[feature] = stat
            p_values[feature] = p

        # 综合漂移分数
        avg_drift_score = np.mean(list(drift_scores.values()))
        min_p_value = np.min(list(p_values.values()))
        is_drifted = min_p_value < 0.05  # 显著性水平5%

        # 确定漂移类型
        drift_type = self._determine_drift_type(avg_drift_score, is_drifted)

        result = DriftDetectionResult(
            drift_type=drift_type,
            drift_score=avg_drift_score,
            p_value=min_p_value,
            is_drifted=is_drifted,
            timestamp=datetime.now()
        )
        self.drift_history.append(result)
        return result

    def check_stability(self, window_size: int = 30) -> Dict[str, float]:
        """检查模型稳定性"""
        if len(self.performance_history) < window_size:
            return {}

        # 计算滑动窗口指标
        metrics = ['accuracy', 'f1_score', 'roc_auc']
        stability = {}

        for metric in metrics:
            values = [getattr(p, metric) for p in self.performance_history[-window_size:]]
            mean = np.mean(values)
            std = np.std(values)
            stability[f'{metric}_mean'] = mean
            stability[f'{metric}_std'] = std
            stability[f'{metric}_cv'] = std / mean if mean > 0 else 0

        return stability

    def generate_alert(self, result: DriftDetectionResult,
                      perf: ModelPerformance) -> Optional[AlertLevel]:
        """生成预警"""
        if result.is_drifted and result.drift_score > 0.3:
            return AlertLevel.CRITICAL
        elif result.is_drifted:
            return AlertLevel.WARNING
        elif perf.f1_score < 0.5:
            return AlertLevel.WARNING
        elif perf.roc_auc < 0.7:
            return AlertLevel.INFO
        return None

    def plot_performance_trend(self) -> plt.Figure:
        """绘制性能趋势图"""
        fig, ax = plt.subplots(figsize=(12,6))
        metrics = ['accuracy', 'f1_score', 'roc_auc']

        for metric in metrics:
            values = [getattr(p, metric) for p in self.performance_history]
            ax.plot(values, label=metric)

        ax.set_title("Model Performance Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)
        return fig

    def plot_drift_history(self) -> plt.Figure:
        """绘制漂移历史图"""
        fig, ax = plt.subplots(figsize=(12,6))
        scores = [d.drift_score for d in self.drift_history]
        ax.plot(scores, 'o-', label='Drift Score')
        ax.axhline(y=0.3, color='r', linestyle='--', label='Threshold')
        ax.set_title("Data Drift History")
        ax.set_xlabel("Check Point")
        ax.set_ylabel("Drift Score")
        ax.legend()
        ax.grid(True)
        return fig

    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算精确率"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算召回率"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def _calculate_log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """计算对数损失"""
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    def _determine_drift_type(self, drift_score: float, is_drifted: bool) -> DriftType:
        """确定漂移类型"""
        if not is_drifted:
            return None
        if drift_score > 0.5:
            return DriftType.CONCEPT_DRIFT
        elif drift_score > 0.3:
            return DriftType.MODEL_DECAY
        else:
            return DriftType.DATA_DRIFT

class MonitoringDashboard:
    """监控看板"""

    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor

    def show_performance(self):
        """显示性能看板"""
        fig = self.monitor.plot_performance_trend()
        plt.show()

    def show_drift(self):
        """显示漂移看板"""
        fig = self.monitor.plot_drift_history()
        plt.show()

    def show_alerts(self):
        """显示预警看板"""
        alerts = []
        for i, (drift, perf) in enumerate(zip(
            self.monitor.drift_history[-10:],
            self.monitor.performance_history[-10:]
        )):
            alert = self.monitor.generate_alert(drift, perf)
            if alert:
                alerts.append({
                    'id': i,
                    'type': drift.drift_type.name if drift else 'PERFORMANCE',
                    'level': alert.name,
                    'time': drift.timestamp if drift else datetime.now()
                })

        if alerts:
            print(pd.DataFrame(alerts))
        else:
            print("No recent alerts")
