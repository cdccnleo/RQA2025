import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging
from sklearn.metrics import accuracy_score
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """集成方法枚举"""
    AVERAGE = auto()      # 简单平均
    WEIGHTED = auto()     # 动态加权
    STACKING = auto()     # 堆叠集成
    VOTING = auto()       # 投票集成

class WeightUpdateRule(Enum):
    """权重更新规则枚举"""
    PERFORMANCE = auto()  # 基于性能
    DIVERSITY = auto()    # 基于多样性
    ADAPTIVE = auto()     # 自适应混合

@dataclass
class EnsembleResult:
    """集成结果数据结构"""
    final_prediction: np.ndarray
    model_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    uncertainty: Optional[np.ndarray] = None

class BaseEnsemble(ABC):
    """集成学习基类"""

    @abstractmethod
    def predict(self, model_predictions: Dict[str, np.ndarray]) -> EnsembleResult:
        """执行模型集成预测"""
        pass

class WeightedEnsemble(BaseEnsemble):
    """动态加权集成"""

    def __init__(self,
                initial_weights: Optional[Dict[str, float]] = None,
                update_rule: WeightUpdateRule = WeightUpdateRule.PERFORMANCE,
                decay_factor: float = 0.9):
        self.weights = initial_weights or {}
        self.update_rule = update_rule
        self.decay_factor = decay_factor
        self.performance_history = {}
        self.window_size = 10
        self.history = {}

    def predict(self, model_predictions: Dict[str, np.ndarray],
               true_labels: Optional[np.ndarray] = None) -> EnsembleResult:
        # 初始化权重
        if not self.weights:
            self.weights = {name: 1/len(model_predictions) for name in model_predictions}
            self.performance_history = {name: deque(maxlen=self.window_size) for name in model_predictions}

        # 计算加权预测
        weighted_sum = np.zeros_like(next(iter(model_predictions.values())))
        for name, pred in model_predictions.items():
            weighted_sum += pred * self.weights[name]

        # 更新权重
        if true_labels is not None:
            self._update_weights(model_predictions, true_labels)

        # 计算不确定性
        uncertainties = self._calculate_uncertainty(model_predictions)

        return EnsembleResult(
            final_prediction=weighted_sum,
            model_weights=self.weights.copy(),
            performance_metrics=self._calculate_metrics(weighted_sum, true_labels),
            uncertainty=uncertainties
        )

    def _update_weights(self, model_predictions: Dict[str, np.ndarray],
                       true_labels: np.ndarray):
        """更新模型权重"""
        # 计算各模型当前性能
        current_perf = {}
        for name, pred in model_predictions.items():
            score = accuracy_score(true_labels, np.round(pred))
            self.performance_history[name].append(score)
            current_perf[name] = np.mean(self.performance_history[name]) if self.performance_history[name] else 0.5

        # 根据更新规则调整权重
        if self.update_rule == WeightUpdateRule.PERFORMANCE:
            total = sum(current_perf.values())
            self.weights = {name: perf/total for name, perf in current_perf.items()}
        elif self.update_rule == WeightUpdateRule.DIVERSITY:
            # 基于多样性的权重更新
            pass
        else:
            # 自适应混合更新
            pass

        # 应用衰减因子
        self.weights = {name: w*self.decay_factor for name, w in self.weights.items()}

    def _calculate_uncertainty(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """计算预测不确定性"""
        pred_matrix = np.array(list(predictions.values()))
        return np.std(pred_matrix, axis=0)

    def _calculate_metrics(self, prediction: np.ndarray,
                          true_labels: Optional[np.ndarray]) -> Dict[str, float]:
        """计算集成模型性能指标"""
        if true_labels is None:
            return {}

        return {
            'accuracy': accuracy_score(true_labels, np.round(prediction)),
            'confidence': np.mean(prediction * (1 - prediction))
        }

class EnsembleMonitor:
    """集成模型监控"""

    def __init__(self, models: List[str]):
        self.model_performance = {name: [] for name in models}
        self.ensemble_performance = []
        self.correlation_matrix = pd.DataFrame(index=models, columns=models)

    def update(self, predictions: Dict[str, np.ndarray],
              true_labels: np.ndarray,
              ensemble_pred: np.ndarray):
        """更新监控数据"""
        # 更新各模型性能
        for name, pred in predictions.items():
            acc = accuracy_score(true_labels, np.round(pred))
            self.model_performance[name].append(acc)

        # 更新集成性能
        self.ensemble_performance.append(
            accuracy_score(true_labels, np.round(ensemble_pred))
        )

        # 更新相关性矩阵
        self._update_correlation(predictions)

    def _update_correlation(self, predictions: Dict[str, np.ndarray]):
        """更新模型预测相关性"""
        pred_df = pd.DataFrame(predictions)
        self.correlation_matrix = pred_df.corr()

    def plot_performance_trend(self) -> plt.Figure:
        """绘制性能趋势图"""
        plt.figure(figsize=(12,6))
        for name, perf in self.model_performance.items():
            plt.plot(perf, label=name)
        plt.plot(self.ensemble_performance, 'k--', label='Ensemble')
        plt.title("Model Performance Trend")
        plt.xlabel("Update Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        return plt.gcf()

    def plot_correlation_heatmap(self) -> plt.Figure:
        """绘制相关性热力图"""
        plt.figure(figsize=(10,8))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm',
                   vmin=-1, vmax=1, center=0)
        plt.title("Model Prediction Correlation")
        return plt.gcf()

class EnsembleVisualizer:
    """集成可视化工具"""

    @staticmethod
    def plot_weight_distribution(weights: Dict[str, float]) -> plt.Figure:
        """绘制权重分布图"""
        plt.figure(figsize=(10,6))
        pd.Series(weights).sort_values().plot.barh()
        plt.title("Model Weight Distribution")
        plt.xlabel("Weight")
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_uncertainty(prediction: np.ndarray,
                        uncertainty: np.ndarray) -> plt.Figure:
        """绘制不确定性分析图"""
        plt.figure(figsize=(12,6))
        plt.plot(prediction, label='Prediction')
        plt.fill_between(range(len(prediction)),
                        prediction - uncertainty,
                        prediction + uncertainty,
                        alpha=0.2, label='Uncertainty')
        plt.title("Prediction with Uncertainty")
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()
        return plt.gcf()
