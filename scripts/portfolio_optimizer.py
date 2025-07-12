import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import cvxpy as cp
from scipy.optimize import minimize
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PortfolioOptimizer:
    """组合优化引擎"""

    def __init__(self, strategies: List[str], risk_budget: Dict[str, float]):
        self.strategies = strategies
        self.risk_budget = risk_budget
        self.cov_matrix = None
        self.returns = None

    def load_performance_data(self, data: pd.DataFrame):
        """加载策略历史表现数据"""
        self.returns = data[[f'ret_{s}' for s in self.strategies]]
        self.cov_matrix = self.returns.cov()
        logger.info(f"Loaded performance data for {len(self.strategies)} strategies")

    def risk_parity_optimization(self) -> Dict[str, float]:
        """风险平价优化"""
        n = len(self.strategies)
        x = cp.Variable(n)
        risk = cp.quad_form(x, self.cov_matrix.values)

        # 风险预算约束
        constraints = [
            cp.sum(x) == 1,
            x >= 0
        ]

        # 优化目标
        objective = cp.Minimize(
            cp.sum_squares(
                cp.multiply(x, cp.sqrt(cp.diag(self.cov_matrix.values))) -
                np.array(list(self.risk_budget.values()))
            )
        )

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = {s: x.value[i] for i, s in enumerate(self.strategies)}
        logger.info(f"Risk parity weights: {weights}")
        return weights

    def mean_variance_optimization(self, target_return: float) -> Dict[str, float]:
        """均值-方差优化"""
        n = len(self.strategies)
        x = cp.Variable(n)
        expected_returns = self.returns.mean().values

        # 优化目标
        risk = cp.quad_form(x, self.cov_matrix.values)
        ret = expected_returns @ x

        constraints = [
            ret >= target_return,
            cp.sum(x) == 1,
            x >= 0
        ]

        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve()

        weights = {s: x.value[i] for i, s in enumerate(self.strategies)}
        logger.info(f"Mean-variance weights: {weights}")
        return weights

    def black_litterman_optimization(self, views: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Black-Litterman优化"""
        # 先验分布 (市场均衡收益)
        tau = 0.05
        pi = self.returns.mean().values  # 市场隐含收益

        # 构建观点矩阵
        P = np.zeros((len(views), len(self.strategies)))
        Q = np.zeros(len(views))
        omega = np.zeros(len(views))

        for i, (s, (q, conf)) in enumerate(views.items()):
            idx = self.strategies.index(s)
            P[i, idx] = 1
            Q[i] = q
            omega[i] = conf

        # 后验收益计算
        cov_inv = np.linalg.inv(self.cov_matrix.values)
        omega_diag = np.diag(omega)

        mu = np.linalg.inv(
            np.linalg.inv(tau * self.cov_matrix.values) +
            P.T @ np.linalg.inv(omega_diag) @ P
        ) @ (
            np.linalg.inv(tau * self.cov_matrix.values) @ pi +
            P.T @ np.linalg.inv(omega_diag) @ Q
        )

        # 均值-方差优化
        weights = self.mean_variance_optimization(mu.mean())
        return weights

    def calculate_marginal_risk_contribution(self, weights: Dict[str, float]) -> Dict[str, float]:
        """计算边际风险贡献"""
        w = np.array([weights[s] for s in self.strategies])
        sigma = np.sqrt(w.T @ self.cov_matrix.values @ w)
        mrc = (self.cov_matrix.values @ w) / sigma

        return {s: mrc[i] for i, s in enumerate(self.strategies)}

    def calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """计算分散化比率"""
        w = np.array([weights[s] for s in self.strategies])
        weighted_vol = np.sqrt(np.sum([(weights[s] * np.sqrt(self.cov_matrix.loc[s, s]))**2
                                      for s in self.strategies]))
        portfolio_vol = np.sqrt(w.T @ self.cov_matrix.values @ w)

        return weighted_vol / portfolio_vol

class SmartRebalancer:
    """智能调仓引擎"""

    def __init__(self, optimizer: PortfolioOptimizer):
        self.optimizer = optimizer
        self.current_weights = None
        self.target_weights = None

    def set_current_weights(self, weights: Dict[str, float]):
        """设置当前持仓权重"""
        self.current_weights = weights

    def set_target_weights(self, weights: Dict[str, float]):
        """设置目标权重"""
        self.target_weights = weights

    def calculate_trades(self, portfolio_value: float) -> Dict[str, float]:
        """计算调仓交易量"""
        if not self.current_weights or not self.target_weights:
            raise ValueError("Weights not set")

        trades = {}
        for s in self.optimizer.strategies:
            trades[s] = (self.target_weights[s] - self.current_weights[s]) * portfolio_value

        logger.info(f"Calculated trades: {trades}")
        return trades

    def optimize_execution(self, trades: Dict[str, float],
                          liquidity: Dict[str, float],
                          cost_params: Dict[str, float]) -> Dict[str, Dict]:
        """优化交易执行"""
        execution_plan = {}

        for s, amount in trades.items():
            # 计算市场冲击成本
            impact_cost = self._calculate_impact_cost(amount, liquidity[s], cost_params)

            # 生成执行计划
            execution_plan[s] = {
                'target_amount': amount,
                'estimated_cost': impact_cost,
                'slices': self._generate_slices(amount, liquidity[s])
            }

        logger.info(f"Execution plan: {execution_plan}")
        return execution_plan

    def _calculate_impact_cost(self, amount: float, daily_volume: float,
                              params: Dict[str, float]) -> float:
        """计算市场冲击成本"""
        participation_rate = abs(amount) / daily_volume
        cost = (params['fixed_cost'] +
                params['linear_cost'] * participation_rate +
                params['quadratic_cost'] * participation_rate**2)
        return cost * np.sign(amount)

    def _generate_slices(self, amount: float, daily_volume: float) -> List[Dict]:
        """生成分笔交易计划"""
        max_participation = 0.1  # 单笔最大参与率
        max_slice = daily_volume * max_participation
        n_slices = int(np.ceil(abs(amount) / max_slice))

        slices = []
        remaining = amount

        for _ in range(n_slices):
            slice_amount = np.sign(remaining) * min(abs(remaining), max_slice)
            slices.append({
                'amount': slice_amount,
                'participation': abs(slice_amount) / daily_volume
            })
            remaining -= slice_amount

        return slices

    def adaptive_rebalance(self, market_state: Dict[str, Dict]) -> Dict[str, float]:
        """自适应调仓"""
        # 根据市场状态调整目标权重
        adjusted_weights = {}
        for s in self.optimizer.strategies:
            state = market_state[s]

            # 波动率调整
            vol_adjustment = np.sqrt(0.1 / state['volatility']) if state['volatility'] > 0 else 1

            # 流动性调整
            liq_adjustment = min(1, state['liquidity'] / 1e6)

            adjusted_weights[s] = self.target_weights[s] * vol_adjustment * liq_adjustment

        # 归一化
        total = sum(adjusted_weights.values())
        adjusted_weights = {s: w/total for s, w in adjusted_weights.items()}

        logger.info(f"Adaptive weights: {adjusted_weights}")
        return adjusted_weights

def main():
    """组合优化主流程"""
    # 策略和风险预算配置
    strategies = ['LSTM', 'RandomForest', 'NeuralNet', 'MeanReversion']
    risk_budget = {
        'LSTM': 0.3,
        'RandomForest': 0.2,
        'NeuralNet': 0.3,
        'MeanReversion': 0.2
    }

    # 初始化优化器
    optimizer = PortfolioOptimizer(strategies, risk_budget)

    # 加载历史表现数据
    data = pd.read_csv('data/strategy_returns.csv', index_col=0)
    optimizer.load_performance_data(data)

    # 运行优化
    risk_parity_weights = optimizer.risk_parity_optimization()
    mv_weights = optimizer.mean_variance_optimization(target_return=0.15)

    # 计算风险贡献
    mrc = optimizer.calculate_marginal_risk_contribution(risk_parity_weights)
    div_ratio = optimizer.calculate_diversification_ratio(risk_parity_weights)

    logger.info(f"Marginal Risk Contribution: {mrc}")
    logger.info(f"Diversification Ratio: {div_ratio:.2f}")

    # 智能调仓
    rebalancer = SmartRebalancer(optimizer)
    rebalancer.set_current_weights({
        'LSTM': 0.25,
        'RandomForest': 0.25,
        'NeuralNet': 0.3,
        'MeanReversion': 0.2
    })
    rebalancer.set_target_weights(risk_parity_weights)

    # 计算调仓交易
    trades = rebalancer.calculate_trades(portfolio_value=1e6)

    # 优化执行
    liquidity = {
        'LSTM': 5e6,
        'RandomForest': 8e6,
        'NeuralNet': 3e6,
        'MeanReversion': 1e7
    }
    cost_params = {
        'fixed_cost': 0.0005,
        'linear_cost': 0.001,
        'quadratic_cost': 0.01
    }
    execution_plan = rebalancer.optimize_execution(trades, liquidity, cost_params)

    # 自适应调仓
    market_state = {
        'LSTM': {'volatility': 0.12, 'liquidity': 4e6},
        'RandomForest': {'volatility': 0.08, 'liquidity': 7e6},
        'NeuralNet': {'volatility': 0.15, 'liquidity': 2.5e6},
        'MeanReversion': {'volatility': 0.1, 'liquidity': 9e6}
    }
    adaptive_weights = rebalancer.adaptive_rebalance(market_state)

if __name__ == "__main__":
    main()
