import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """组合优化方法枚举"""
    MEAN_VARIANCE = auto()  # 均值-方差优化
    RISK_PARITY = auto()    # 风险平价
    MIN_VARIANCE = auto()   # 最小方差
    MAX_DIVERSITY = auto()  # 最大多样性
    ADAPTIVE = auto()       # 自适应混合

@dataclass
class ModelPerformance:
    """模型表现跟踪"""
    returns: pd.Series
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

class EnsembleOptimizer:
    """模型组合优化器"""

    def __init__(self,
                 method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                 risk_aversion: float = 0.5,
                 rebalance_freq: str = 'W'):
        """
        初始化优化器

        Args:
            method: 优化方法
            risk_aversion: 风险厌恶系数(0-1)
            rebalance_freq: 再平衡频率(D/W/M)
        """
        self.method = method
        self.risk_aversion = risk_aversion
        self.rebalance_freq = rebalance_freq
        self.cov_estimator = LedoitWolf()
        self.weights_history = []
        self.performance_history = []

    def calculate_performance(self,
                            predictions: Dict[str, pd.DataFrame]) -> Dict[str, ModelPerformance]:
        """
        计算各模型表现指标

        Args:
            predictions: 各模型预测结果 {model_name: DataFrame}

        Returns:
            各模型表现指标字典
        """
        performance = {}
        for name, pred in predictions.items():
            returns = pred['return']
            vol = returns.std()
            sharpe = returns.mean() / vol if vol > 0 else 0
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.expanding().max()
            drawdown = (cum_returns - peak) / peak

            performance[name] = ModelPerformance(
                returns=returns,
                volatility=vol,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown.min()
            )
        return performance

    def optimize_weights(self,
                       predictions: Dict[str, pd.DataFrame],
                       current_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        优化模型权重

        Args:
            predictions: 各模型预测结果 {model_name: DataFrame}
            current_weights: 当前权重(用于自适应调整)

        Returns:
            优化后的权重字典
        """
        # 准备收益数据
        returns = pd.DataFrame({name: pred['return'] for name, pred in predictions.items()})
        mean_returns = returns.mean()
        cov_matrix = self.cov_estimator.fit(returns).covariance_
        n = len(mean_returns)

        # 初始权重(等权)
        if current_weights is None:
            init_weights = np.ones(n) / n
        else:
            init_weights = np.array([current_weights[name] for name in returns.columns])

        # 根据方法选择优化目标
        if self.method == OptimizationMethod.MEAN_VARIANCE:
            def objective(w):
                port_return = np.dot(w, mean_returns)
                port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                return - (port_return - self.risk_aversion * port_vol ** 2)

        elif self.method == OptimizationMethod.RISK_PARITY:
            def objective(w):
                risk_contrib = w * np.dot(cov_matrix, w) / np.dot(w, np.dot(cov_matrix, w))
                return np.sum((risk_contrib - 1/n) ** 2)

        elif self.method == OptimizationMethod.MIN_VARIANCE:
            def objective(w):
                return np.dot(w, np.dot(cov_matrix, w))

        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

        # 约束条件
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]  # 不允许做空

        # 优化求解
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return dict(zip(returns.columns, init_weights))

        optimal_weights = result.x
        optimal_weights = optimal_weights / optimal_weights.sum()  # 确保和为1

        # 记录历史
        self.weights_history.append(dict(zip(returns.columns, optimal_weights)))

        return dict(zip(returns.columns, optimal_weights))

    def adaptive_update(self,
                      new_predictions: Dict[str, pd.DataFrame],
                      current_weights: Dict[str, float],
                      learning_rate: float = 0.1) -> Dict[str, float]:
        """
        自适应权重更新

        Args:
            new_predictions: 新预测结果
            current_weights: 当前权重
            learning_rate: 学习率

        Returns:
            更新后的权重
        """
        # 计算新数据下的最优权重
        new_optimal = self.optimize_weights(new_predictions)

        # 线性混合
        updated_weights = {
            name: (1 - learning_rate) * current_weights.get(name, 0) +
                  learning_rate * new_optimal.get(name, 0)
            for name in set(current_weights) | set(new_optimal)
        }

        # 归一化
        total = sum(updated_weights.values())
        updated_weights = {name: w/total for name, w in updated_weights.items()}

        return updated_weights

    def analyze_performance(self,
                          predictions: Dict[str, pd.DataFrame],
                          weights: Dict[str, float]) -> Dict[str, float]:
        """
        分析组合表现

        Args:
            predictions: 各模型预测
            weights: 权重分配

        Returns:
            组合表现指标
        """
        returns = pd.DataFrame({name: pred['return'] for name, pred in predictions.items()})
        weighted_returns = returns.dot(np.array(list(weights.values())))

        # 计算组合指标
        port_return = weighted_returns.mean()
        port_vol = weighted_returns.std()
        sharpe = port_return / port_vol if port_vol > 0 else 0
        cum_returns = (1 + weighted_returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak

        # 保存历史
        self.performance_history.append({
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe,
            'max_drawdown': drawdown.min()
        })

        return {
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe,
            'max_drawdown': drawdown.min()
        }

    def calculate_correlation(self, predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算模型间相关性矩阵

        Args:
            predictions: 各模型预测

        Returns:
            相关性矩阵DataFrame
        """
        returns = pd.DataFrame({name: pred['return'] for name, pred in predictions.items()})
        return returns.corr()

class RiskBudgetOptimizer:
    """风险预算优化器"""

    def __init__(self, risk_budget: Dict[str, float] = None):
        """
        初始化优化器

        Args:
            risk_budget: 各模型风险预算(默认等风险贡献)
        """
        self.risk_budget = risk_budget

    def optimize(self,
                predictions: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        风险预算优化

        Args:
            predictions: 各模型预测

        Returns:
            优化权重
        """
        returns = pd.DataFrame({name: pred['return'] for name, pred in predictions.items()})
        cov_matrix = LedoitWolf().fit(returns).covariance_
        n = len(returns.columns)

        # 默认等风险预算
        if self.risk_budget is None:
            budget = np.ones(n) / n
        else:
            budget = np.array([self.risk_budget.get(name, 0) for name in returns.columns])
            budget = budget / budget.sum()

        # 优化目标: 风险贡献与预算的差异
        def objective(w):
            risk_contrib = w * np.dot(cov_matrix, w) / np.dot(w, np.dot(cov_matrix, w))
            return np.sum((risk_contrib - budget) ** 2)

        # 约束条件
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]  # 不允许做空

        # 初始猜测(等权)
        init_weights = np.ones(n) / n

        # 优化求解
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"Risk budget optimization failed: {result.message}")
            return dict(zip(returns.columns, init_weights))

        optimal_weights = result.x
        optimal_weights = optimal_weights / optimal_weights.sum()  # 确保和为1

        return dict(zip(returns.columns, optimal_weights))
