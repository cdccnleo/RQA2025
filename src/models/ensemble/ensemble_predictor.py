import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
import warnings
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """集成方法枚举"""
    AVERAGE = auto()      # 简单平均
    STACKING = auto()     # 堆叠泛化
    BMA = auto()          # 贝叶斯模型平均
    DYNAMIC = auto()      # 动态加权

class UncertaintyMethod(Enum):
    """不确定性量化方法"""
    STD = auto()         # 标准差
    ENTROPY = auto()     # 预测熵
    CONFIDENCE = auto()  # 置信区间

@dataclass
class ModelPrediction:
    """模型预测数据结构"""
    model_name: str
    prediction: np.ndarray
    confidence: Optional[np.ndarray] = None
    meta: Optional[Dict] = None

@dataclass
class EnsembleResult:
    """集成结果数据结构"""
    prediction: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    weights: Optional[Dict[str, float]] = None
    plot_data: Optional[Dict] = None

class BaseEnsemble(ABC):
    """集成模型基类"""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练集成模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> EnsembleResult:
        """生成集成预测"""
        pass

class AverageEnsemble(BaseEnsemble):
    """简单平均集成"""

    def __init__(self, models: Dict[str, object]):
        self.models = models

    def fit(self, X, y):
        # 简单平均不需要训练
        pass

    def predict(self, X) -> EnsembleResult:
        predictions = []
        weights = {name: 1/len(self.models) for name in self.models}

        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)

        avg_pred = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return EnsembleResult(
            prediction=avg_pred,
            uncertainty=std,
            weights=weights
        )

class StackingEnsemble(BaseEnsemble):
    """堆叠集成"""

    def __init__(self,
                base_models: Dict[str, object],
                meta_model: object = None):
        self.base_models = base_models
        self.meta_model = meta_model or LinearRegression()

    def fit(self, X, y):
        # 生成基模型预测
        base_preds = []
        for name, model in self.base_models.items():
            pred = model.predict(X)
            base_preds.append(pred)

        # 训练元模型
        X_meta = np.column_stack(base_preds)
        self.meta_model.fit(X_meta, y)

    def predict(self, X) -> EnsembleResult:
        base_preds = []
        for name, model in self.base_models.items():
            pred = model.predict(X)
            base_preds.append(pred)

        X_meta = np.column_stack(base_preds)
        final_pred = self.meta_model.predict(X_meta)

        return EnsembleResult(
            prediction=final_pred,
            weights={name: coef for name, coef in
                   zip(self.base_models.keys(), self.meta_model.coef_)}
        )

class BayesianModelAveraging(BaseEnsemble):
    """贝叶斯模型平均"""

    def __init__(self, models: Dict[str, object]):
        self.models = models
        self.model_weights = None

    def fit(self, X, y):
        # 计算模型权重(基于样本外误差)
        errors = {}
        for name, model in self.models.items():
            pred = model.predict(X)
            error = np.mean((pred - y) ** 2)
            errors[name] = error

        # 转换为权重(误差越小权重越大)
        min_error = min(errors.values())
        weights = {name: min_error/max(error, 1e-6)
                  for name, error in errors.items()}
        total = sum(weights.values())
        self.model_weights = {name: w/total for name, w in weights.items()}

    def predict(self, X) -> EnsembleResult:
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X) * self.model_weights[name]
            predictions.append(pred)

        final_pred = np.sum(predictions, axis=0)

        return EnsembleResult(
            prediction=final_pred,
            weights=self.model_weights
        )

class DynamicWeightedEnsemble(BaseEnsemble):
    """动态加权集成"""

    def __init__(self,
                models: Dict[str, object],
                lookback: int = 21):
        self.models = models
        self.lookback = lookback
        self.recent_performance = None

    def fit(self, X, y):
        # 计算近期表现
        self._update_performance(X, y)

    def _update_performance(self, X, y):
        """更新模型近期表现"""
        performances = {}
        for name, model in self.models.items():
            pred = model.predict(X)
            error = np.mean((pred - y) ** 2)
            performances[name] = 1 / (error + 1e-6)

        self.recent_performance = performances

    def predict(self, X) -> EnsembleResult:
        if self.recent_performance is None:
            raise ValueError("Model not fitted")

        # 计算动态权重
        total = sum(self.recent_performance.values())
        weights = {name: perf/total
                  for name, perf in self.recent_performance.items()}

        # 加权预测
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X) * weights[name]
            predictions.append(pred)

        final_pred = np.sum(predictions, axis=0)

        return EnsembleResult(
            prediction=final_pred,
            weights=weights
        )

class EnsembleVisualizer:
    """集成可视化工具"""

    @staticmethod
    def plot_weights(weights: Dict[str, float]):
        """绘制模型权重图"""
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=list(weights.values()),
            y=list(weights.keys()),
            orient='h'
        )
        plt.title("Model Weights")
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_uncertainty(prediction: np.ndarray,
                        uncertainty: np.ndarray):
        """绘制不确定性带图"""
        plt.figure(figsize=(12, 6))
        plt.plot(prediction, label='Prediction')
        plt.fill_between(
            range(len(prediction)),
            prediction - uncertainty,
            prediction + uncertainty,
            alpha=0.2,
            label='Uncertainty'
        )
        plt.legend()
        plt.title("Prediction with Uncertainty")
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_contributions(predictions: Dict[str, np.ndarray],
                         final_pred: np.ndarray):
        """绘制模型贡献图"""
        plt.figure(figsize=(12, 6))
        for name, pred in predictions.items():
            plt.plot(pred, alpha=0.5, label=name)
        plt.plot(final_pred, 'k-', linewidth=2, label='Ensemble')
        plt.legend()
        plt.title("Model Contributions")
        plt.tight_layout()
        return plt.gcf()
