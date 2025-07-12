import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class StrategyStatus(Enum):
    """策略状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"

@dataclass
class StrategyAllocation:
    """策略分配配置"""
    strategy_id: str
    weight: float
    capital: float
    status: StrategyStatus = StrategyStatus.ACTIVE

class PortfolioOptimizer:
    """组合优化器"""

    def __init__(self, strategies: List[StrategyAllocation]):
        self.strategies = strategies
        self._validate_weights()

    def _validate_weights(self):
        """验证权重配置"""
        total = sum(s.weight for s in self.strategies)
        if not np.isclose(total, 1.0):
            raise ValueError("策略权重总和必须为1")

    def optimize_weights(self, returns: pd.DataFrame,
                        risk_free_rate: float = 0.02) -> Dict[str, float]:
        """优化策略权重"""
        cov_matrix = returns.cov()
        mean_returns = returns.mean()

        # 定义夏普率优化目标函数
        def negative_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - risk_free_rate) / port_vol

        # 约束条件
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.strategies)))

        # 初始权重
        init_weights = np.array([s.weight for s in self.strategies])

        # 优化
        opt_result = minimize(negative_sharpe, init_weights,
                             method='SLSQP', bounds=bounds,
                             constraints=constraints)

        # 更新策略权重
        optimized_weights = opt_result.x
        for i, strategy in enumerate(self.strategies):
            strategy.weight = optimized_weights[i]

        return {s.strategy_id: s.weight for s in self.strategies}

class CapitalAllocator:
    """动态资金分配器"""

    def __init__(self, total_capital: float = 1e6,
                max_strategy_capital: float = 0.3):
        self.total_capital = total_capital
        self.max_strategy_capital = max_strategy_capital
        self.allocated_capital = 0.0

    def allocate_capital(self, strategies: List[StrategyAllocation]) -> Dict[str, float]:
        """分配资金"""
        # 计算总权重
        total_weight = sum(s.weight for s in strategies)

        # 分配资金
        allocations = {}
        for strategy in strategies:
            capital = min(
                self.total_capital * strategy.weight / total_weight,
                self.max_strategy_capital * self.total_capital
            )
            strategy.capital = capital
            allocations[strategy.strategy_id] = capital
            self.allocated_capital += capital

        return allocations

    def rebalance(self, new_total: float):
        """重新平衡资金"""
        self.total_capital = new_total
        self.allocated_capital = 0.0

class PerformanceAttribution:
    """绩效归因系统"""

    def __init__(self):
        self.performance_data = {}

    def add_performance(self, strategy_id: str,
                      returns: pd.Series,
                      benchmark: pd.Series):
        """添加策略绩效"""
        stats = self._calculate_attribution(returns, benchmark)
        self.performance_data[strategy_id] = stats
        return stats

    def _calculate_attribution(self, returns: pd.Series,
                            benchmark: pd.Series) -> Dict[str, float]:
        """计算归因指标"""
        excess = returns - benchmark
        return {
            'alpha': excess.mean(),
            'beta': returns.cov(benchmark) / benchmark.var(),
            'information_ratio': excess.mean() / excess.std(),
            'active_share': (returns - benchmark).abs().mean(),
            'tracking_error': excess.std()
        }

    def get_strategy_attribution(self, strategy_id: str) -> Optional[Dict]:
        """获取策略归因"""
        return self.performance_data.get(strategy_id)

class PortfolioRiskManager:
    """组合风险管理系统"""

    def __init__(self, max_drawdown: float = 0.2,
                max_strategy_risk: float = 0.15,
                correlation_threshold: float = 0.7):
        self.max_drawdown = max_drawdown
        self.max_strategy_risk = max_strategy_risk
        self.correlation_threshold = correlation_threshold
        self.risk_metrics = {}

    def check_portfolio_risk(self, returns: pd.DataFrame) -> bool:
        """检查组合风险"""
        # 计算组合回撤
        port_returns = returns.mean(axis=1)
        cum_returns = (1 + port_returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (peak - cum_returns) / peak

        # 检查策略相关性
        corr_matrix = returns.corr()
        high_corr = (corr_matrix.abs() > self.correlation_threshold).sum().sum()

        return (drawdown.max() <= self.max_drawdown and
                high_corr <= len(returns.columns))

    def check_strategy_risk(self, strategy_id: str,
                         returns: pd.Series) -> bool:
        """检查策略风险"""
        volatility = returns.std()
        return volatility <= self.max_strategy_risk

    def update_risk_metrics(self, strategy_id: str,
                          metrics: Dict[str, float]):
        """更新风险指标"""
        self.risk_metrics[strategy_id] = metrics

class StrategyPortfolio:
    """策略组合管理系统"""

    def __init__(self, total_capital: float = 1e6):
        self.strategies: Dict[str, StrategyAllocation] = {}
        self.optimizer = PortfolioOptimizer([])
        self.allocator = CapitalAllocator(total_capital)
        self.attribution = PerformanceAttribution()
        self.risk_manager = PortfolioRiskManager()

    def add_strategy(self, strategy_id: str,
                   initial_weight: float = 0.0):
        """添加策略"""
        if strategy_id in self.strategies:
            raise ValueError("策略已存在")

        self.strategies[strategy_id] = StrategyAllocation(
            strategy_id=strategy_id,
            weight=initial_weight,
            capital=0.0
        )
        self._update_optimizer()

    def _update_optimizer(self):
        """更新优化器"""
        self.optimizer = PortfolioOptimizer(list(self.strategies.values()))

    def allocate_capital(self) -> Dict[str, float]:
        """分配资金"""
        return self.allocator.allocate_capital(list(self.strategies.values()))

    def optimize_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """优化策略权重"""
        return self.optimizer.optimize_weights(returns)

    def update_performance(self, strategy_id: str,
                        returns: pd.Series,
                        benchmark: pd.Series):
        """更新绩效"""
        return self.attribution.add_performance(strategy_id, returns, benchmark)

    def check_risk(self, returns: pd.DataFrame) -> bool:
        """检查组合风险"""
        return self.risk_manager.check_portfolio_risk(returns)
