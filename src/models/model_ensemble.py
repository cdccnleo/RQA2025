import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WeightMethod(Enum):
    """权重分配方法"""
    EQUAL = "equal"              # 等权重
    OPTIMAL = "optimal"          # 最优组合
    DYNAMIC = "dynamic"          # 动态调整
    RISK_PARITY = "risk_parity"  # 风险平价

@dataclass
class ModelPrediction:
    """单个模型预测结果"""
    model_name: str
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

class ModelEnsemble:
    """多模型集成预测"""

    def __init__(self, weight_method: WeightMethod = WeightMethod.OPTIMAL):
        self.weight_method = weight_method
        self.models = {}  # 存储注册的模型
        self.weights = {}  # 模型权重
        self.meta_model = None  # 二级学习器

    def add_model(self, model_name: str, model: object):
        """注册预测模型"""
        self.models[model_name] = model
        self.weights[model_name] = 1.0 / len(self.models) if self.models else 1.0

    def update_weights(self, new_weights: Dict[str, float]):
        """更新模型权重"""
        for name, weight in new_weights.items():
            if name in self.weights:
                self.weights[name] = weight

    def _equal_weights(self) -> Dict[str, float]:
        """等权重分配"""
        return {name: 1.0/len(self.models) for name in self.models}

    def _optimal_weights(self, predictions: Dict[str, ModelPrediction],
                        actual: np.ndarray) -> Dict[str, float]:
        """最优组合权重"""
        # 将预测结果转换为DataFrame
        pred_df = pd.DataFrame({
            name: pred.predictions for name, pred in predictions.items()
        })

        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = []

        for train_idx, val_idx in tscv.split(pred_df):
            # 训练元模型
            self.meta_model = LinearRegression()
            self.meta_model.fit(pred_df.iloc[train_idx], actual[train_idx])

            # 在验证集上生成元特征
            val_pred = self.meta_model.predict(pred_df.iloc[val_idx])
            meta_features.append(val_pred)

        # 合并交叉验证结果
        full_meta = np.concatenate(meta_features)

        # 训练最终元模型
        self.meta_model = LinearRegression()
        self.meta_model.fit(pred_df, actual)

        # 返回模型系数作为权重
        return dict(zip(predictions.keys(), self.meta_model.coef_))

    def _dynamic_weights(self, predictions: Dict[str, ModelPrediction],
                       window: int = 30) -> Dict[str, float]:
        """动态权重分配"""
        # 基于近期表现调整权重
        performances = {}
        for name, pred in predictions.items():
            # 使用置信度或预测误差作为性能指标
            if pred.confidence is not None:
                performances[name] = np.mean(pred.confidence[-window:])
            else:
                performances[name] = 1.0  # 默认权重

        total = sum(performances.values())
        return {name: perf/total for name, perf in performances.items()}

    def _risk_parity_weights(self, predictions: Dict[str, ModelPrediction]) -> Dict[str, float]:
        """风险平价权重"""
        risks = {}
        for name, pred in predictions.items():
            # 计算每个模型预测的风险(波动率)
            risks[name] = np.std(pred.predictions)

        inv_risks = {name: 1.0/risk for name, risk in risks.items()}
        total = sum(inv_risks.values())
        return {name: risk/total for name, risk in inv_risks.items()}

    def combine_predictions(self, predictions: Dict[str, ModelPrediction],
                          actual: Optional[np.ndarray] = None) -> np.ndarray:
        """
        组合多个模型的预测结果

        Args:
            predictions: 各模型的预测结果
            actual: 实际值(用于有监督组合)

        Returns:
            组合后的预测结果
        """
        if not predictions:
            raise ValueError("No predictions provided")

        # 根据选择的方法计算权重
        if self.weight_method == WeightMethod.EQUAL:
            weights = self._equal_weights()
        elif self.weight_method == WeightMethod.OPTIMAL and actual is not None:
            weights = self._optimal_weights(predictions, actual)
        elif self.weight_method == WeightMethod.DYNAMIC:
            weights = self._dynamic_weights(predictions)
        elif self.weight_method == WeightMethod.RISK_PARITY:
            weights = self._risk_parity_weights(predictions)
        else:
            weights = self._equal_weights()

        self.update_weights(weights)

        # 加权组合预测
        combined = np.zeros_like(next(iter(predictions.values())).predictions)
        total_weight = 0.0

        for name, pred in predictions.items():
            weight = weights.get(name, 0.0)
            combined += weight * pred.predictions
            total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        return combined

    def evaluate_models(self, predictions: Dict[str, ModelPrediction],
                       actual: np.ndarray) -> pd.DataFrame:
        """
        评估各模型表现

        Args:
            predictions: 各模型的预测结果
            actual: 实际值

        Returns:
            包含各模型评估指标的DataFrame
        """
        metrics = []
        for name, pred in predictions.items():
            error = pred.predictions - actual
            mse = np.mean(error**2)
            corr = np.corrcoef(pred.predictions, actual)[0,1]

            metrics.append({
                'Model': name,
                'MSE': mse,
                'Correlation': corr,
                'StdDev': np.std(pred.predictions),
                'CurrentWeight': self.weights.get(name, 0.0)
            })

        return pd.DataFrame(metrics)

    def rolling_combination(self,
                          predictions: Dict[str, ModelPrediction],
                          actual: np.ndarray,
                          window: int = 30) -> np.ndarray:
        """
        滚动窗口组合预测

        Args:
            predictions: 各模型的预测结果
            actual: 实际值(用于动态调整)
            window: 滚动窗口大小

        Returns:
            滚动组合后的预测结果
        """
        if len(actual) < window:
            raise ValueError("Window size larger than available data")

        combined = np.zeros_like(actual)

        for i in range(window, len(actual)):
            # 获取窗口内数据
            pred_window = {}
            for name, pred in predictions.items():
                pred_window[name] = ModelPrediction(
                    model_name=name,
                    predictions=pred.predictions[i-window:i],
                    confidence=pred.confidence[i-window:i] if pred.confidence else None
                )

            # 计算窗口内最优权重
            weights = self._optimal_weights(pred_window, actual[i-window:i])

            # 应用权重到当前预测点
            current_pred = 0.0
            total_weight = 0.0
            for name, pred in predictions.items():
                weight = weights.get(name, 0.0)
                current_pred += weight * pred.predictions[i]
                total_weight += weight

            if total_weight > 0:
                combined[i] = current_pred / total_weight

        return combined

class RiskAwareEnsemble(ModelEnsemble):
    """带风险控制的模型组合"""

    def __init__(self, risk_target: float = 0.1, max_leverage: float = 2.0):
        super().__init__(weight_method=WeightMethod.RISK_PARITY)
        self.risk_target = risk_target  # 目标风险水平
        self.max_leverage = max_leverage  # 最大杠杆

    def combine_predictions(self, predictions: Dict[str, ModelPrediction],
                          actual: Optional[np.ndarray] = None) -> np.ndarray:
        """
        带风险控制的组合预测

        Args:
            predictions: 各模型的预测结果
            actual: 实际值(用于有监督组合)

        Returns:
            风险调整后的组合预测
        """
        # 首先获取基础组合预测
        combined = super().combine_predictions(predictions, actual)

        # 计算组合风险
        pred_std = np.std(combined)
        scaling_factor = min(self.risk_target / (pred_std + 1e-6), self.max_leverage)

        # 应用风险调整
        return combined * scaling_factor

    def calculate_portfolio_risk(self, predictions: Dict[str, ModelPrediction]) -> float:
        """计算组合风险"""
        # 获取各模型预测
        preds = {name: pred.predictions for name, pred in predictions.items()}
        pred_df = pd.DataFrame(preds)

        # 计算协方差矩阵
        cov_matrix = pred_df.cov()

        # 计算组合风险
        weights = np.array(list(self.weights.values()))
        portfolio_var = weights.T @ cov_matrix @ weights
        return np.sqrt(portfolio_var)

class OnlineModelEnsemble(ModelEnsemble):
    """在线学习模型组合"""

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(weight_method=WeightMethod.DYNAMIC)
        self.learning_rate = learning_rate
        self.online_weights = None

    def update_online(self, predictions: Dict[str, ModelPrediction],
                     actual: float) -> Dict[str, float]:
        """
        在线更新权重

        Args:
            predictions: 当前各模型的预测
            actual: 实际观测值

        Returns:
            更新后的权重
        """
        if self.online_weights is None:
            self.online_weights = self._equal_weights()

        # 计算各模型当前误差
        errors = {}
        for name, pred in predictions.items():
            last_pred = pred.predictions[-1] if len(pred.predictions) > 0 else 0
            errors[name] = (last_pred - actual)**2

        # 归一化误差
        total_error = sum(errors.values())
        if total_error > 0:
            normalized_errors = {name: err/total_error for name, err in errors.items()}
        else:
            normalized_errors = {name: 1.0/len(errors) for name in errors}

        # 更新权重
        new_weights = {}
        for name in self.online_weights:
            new_weights[name] = self.online_weights[name] * \
                               (1 - self.learning_rate * normalized_errors.get(name, 0.5))

        # 归一化权重
        total_weight = sum(new_weights.values())
        self.online_weights = {name: w/total_weight for name, w in new_weights.items()}

        return self.online_weights
