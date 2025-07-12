import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class StrategyWeight:
    """策略权重配置"""
    name: str
    weight: float
    strategy: object

class PortfolioOptimizer:
    """组合策略优化器"""

    def __init__(self, strategies: List[StrategyWeight]):
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

        return {s.name: s.weight for s in self.strategies}

class PositionManager:
    """动态仓位管理器"""

    def __init__(self, max_position: float = 0.2,
                risk_per_trade: float = 0.01):
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade
        self.current_positions = {}

    def calculate_position_size(self, signal_strength: float,
                             volatility: float,
                             account_size: float) -> float:
        """计算仓位大小"""
        # 基于信号强度和波动率调整
        adjusted_risk = self.risk_per_trade * signal_strength / volatility
        position_size = min(adjusted_risk * account_size,
                           self.max_position * account_size)
        return position_size

    def update_positions(self, symbol: str, size: float):
        """更新持仓"""
        self.current_positions[symbol] = size

    def get_total_exposure(self) -> float:
        """获取总风险敞口"""
        return sum(self.current_positions.values())

class RiskController:
    """风险控制系统"""

    def __init__(self, max_drawdown: float = 0.2,
                volatility_threshold: float = 0.3):
        self.max_drawdown = max_drawdown
        self.volatility_threshold = volatility_threshold
        self.performance_stats = {}

    def check_drawdown(self, current_value: float,
                      peak_value: float) -> bool:
        """检查最大回撤"""
        drawdown = (peak_value - current_value) / peak_value
        return drawdown <= self.max_drawdown

    def check_volatility(self, returns: pd.Series) -> bool:
        """检查波动率"""
        volatility = returns.std()
        return volatility <= self.volatility_threshold

    def evaluate_strategy_risk(self, strategy_name: str,
                             returns: pd.Series) -> Dict[str, float]:
        """评估策略风险"""
        stats = {
            'volatility': returns.std(),
            'max_drawdown': (returns.cummax() - returns).max(),
            'sharpe_ratio': returns.mean() / returns.std()
        }
        self.performance_stats[strategy_name] = stats
        return stats

class SignalProcessor:
    """信号处理系统"""

    def __init__(self, confirmation_period: int = 3,
                threshold: float = 0.5):
        self.confirmation_period = confirmation_period
        self.threshold = threshold
        self.signal_history = {}

    def filter_signals(self, signals: pd.Series) -> pd.Series:
        """过滤信号"""
        # 确认信号持续性
        confirmed_signals = signals.rolling(
            self.confirmation_period).mean()
        # 应用阈值过滤
        filtered = confirmed_signals.abs() >= self.threshold
        return signals.where(filtered, 0)

    def smooth_signals(self, signals: pd.Series) -> pd.Series:
        """平滑信号"""
        return signals.ewm(span=5).mean()

class BacktestAnalyzer:
    """回测分析增强版"""

    def __init__(self, initial_capital: float = 1e6):
        self.initial_capital = initial_capital
        self.metrics = {}

    def calculate_performance(self, returns: pd.Series) -> Dict[str, float]:
        """计算绩效指标"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (peak - cumulative) / peak

        stats = {
            'total_return': cumulative.iloc[-1] - 1,
            'annualized_return': (1 + returns.mean())**252 - 1,
            'max_drawdown': drawdown.max(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'win_rate': (returns > 0).mean(),
            'profit_factor': returns[returns > 0].sum() / -returns[returns < 0].sum()
        }
        return stats

    def generate_report(self, results: Dict[str, pd.DataFrame]) -> Dict:
        """生成回测报告"""
        report = {}
        for name, df in results.items():
            report[name] = self.calculate_performance(df['returns'])
        return report
