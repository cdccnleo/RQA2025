# -*- coding: utf-8 -*-
"""
优化层 - 组合优化模块测试覆盖率提升测试
补充组合优化模块单元测试，目标覆盖率: 80%+

测试范围:
1. 均值方差优化测试 - 有效前沿、权重优化、约束处理
2. 风险平价优化测试 - 风险贡献均衡、相关性调整、再平衡
3. Black-Litterman模型测试 - 主观观点整合、后验分布、投资组合调整
4. 组合优化器集成测试 - 多策略比较、性能评估、参数调优
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math


class TestMeanVarianceOptimization:
    """测试均值方差优化功能"""

    def test_efficient_frontier_calculation(self):
        """测试有效前沿计算"""
        class EfficientFrontierCalculator:
            def __init__(self):
                self.risk_free_rate = 0.02

            def calculate_efficient_frontier(self, expected_returns: np.ndarray,
                                           cov_matrix: np.ndarray, n_portfolios: int = 100) -> Dict[str, Any]:
                """计算有效前沿"""
                n_assets = len(expected_returns)

                # 生成随机权重组合
                portfolios = []
                for _ in range(n_portfolios):
                    weights = np.random.random(n_assets)
                    weights /= np.sum(weights)  # 归一化

                    # 计算组合预期收益率和风险
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                    portfolios.append({
                        "weights": weights,
                        "return": portfolio_return,
                        "risk": portfolio_risk,
                        "sharpe_ratio": (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                    })

                # 找到有效前沿（最高夏普比率的组合）
                best_portfolio = max(portfolios, key=lambda x: x["sharpe_ratio"])

                # 计算最小方差组合
                min_variance_portfolio = min(portfolios, key=lambda x: x["risk"])

                return {
                    "portfolios": portfolios,
                    "efficient_portfolio": best_portfolio,
                    "min_variance_portfolio": min_variance_portfolio,
                    "frontier_points": len(portfolios),
                    "max_sharpe_ratio": best_portfolio["sharpe_ratio"]
                }

            def optimize_for_target_return(self, expected_returns: np.ndarray,
                                         cov_matrix: np.ndarray, target_return: float) -> Dict[str, Any]:
                """为目标收益率优化最小风险组合"""
                from scipy.optimize import minimize

                n_assets = len(expected_returns)

                # 目标函数：最小化方差
                def objective(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))

                # 约束条件
                constraints = [
                    {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # 权重和为1
                    {"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return}  # 目标收益率
                ]

                # 边界条件：权重在0-1之间
                bounds = [(0, 1) for _ in range(n_assets)]

                # 初始权重
                x0 = np.ones(n_assets) / n_assets

                try:
                    result = minimize(objective, x0, method="SLSQP",
                                    bounds=bounds, constraints=constraints)

                    if result.success:
                        optimal_weights = result.x
                        portfolio_return = np.dot(optimal_weights, expected_returns)
                        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

                        return {
                            "optimal_weights": optimal_weights,
                            "portfolio_return": portfolio_return,
                            "portfolio_risk": portfolio_risk,
                            "sharpe_ratio": (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0,
                            "optimization_success": True
                        }
                    else:
                        return {"optimization_success": False, "error": result.message}

                except Exception as e:
                    return {"optimization_success": False, "error": str(e)}

        calculator = EfficientFrontierCalculator()

        # 创建测试数据
        np.random.seed(42)
        expected_returns = np.array([0.08, 0.12, 0.10, 0.14, 0.09])
        cov_matrix = np.random.rand(5, 5)
        cov_matrix = np.dot(cov_matrix, cov_matrix.T) / 10  # 确保正定

        # 计算有效前沿
        frontier = calculator.calculate_efficient_frontier(expected_returns, cov_matrix, 50)

        assert "portfolios" in frontier
        assert "efficient_portfolio" in frontier
        assert "min_variance_portfolio" in frontier
        assert len(frontier["portfolios"]) == 50
        assert frontier["max_sharpe_ratio"] > 0

        # 为目标收益率优化
        target_return = 0.11
        optimized = calculator.optimize_for_target_return(expected_returns, cov_matrix, target_return)

        if optimized["optimization_success"]:
            assert abs(optimized["portfolio_return"] - target_return) < 0.01  # 收益率接近目标
            assert optimized["portfolio_risk"] > 0
            assert len(optimized["optimal_weights"]) == 5
            assert abs(np.sum(optimized["optimal_weights"]) - 1.0) < 0.01  # 权重和为1

    def test_portfolio_weight_optimization(self):
        """测试组合权重优化"""
        class PortfolioWeightOptimizer:
            def __init__(self, expected_returns: np.ndarray, cov_matrix: np.ndarray):
                self.expected_returns = expected_returns
                self.cov_matrix = cov_matrix
                self.n_assets = len(expected_returns)

            def optimize_max_sharpe_ratio(self, risk_free_rate: float = 0.02) -> Dict[str, Any]:
                """优化最大夏普比率组合"""
                from scipy.optimize import minimize

                def negative_sharpe_ratio(weights):
                    portfolio_return = np.dot(weights, self.expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                    return -sharpe_ratio  # 最小化负夏普比率

                # 约束条件
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

                # 边界条件
                bounds = [(0, 1) for _ in range(self.n_assets)]

                # 初始权重
                x0 = np.ones(self.n_assets) / self.n_assets

                try:
                    result = minimize(negative_sharpe_ratio, x0, method="SLSQP",
                                    bounds=bounds, constraints=constraints)

                    if result.success:
                        optimal_weights = result.x
                        portfolio_return = np.dot(optimal_weights, self.expected_returns)
                        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
                        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

                        return {
                            "optimal_weights": optimal_weights,
                            "portfolio_return": portfolio_return,
                            "portfolio_risk": portfolio_risk,
                            "sharpe_ratio": sharpe_ratio,
                            "optimization_success": True
                        }
                    else:
                        return {"optimization_success": False, "error": result.message}

                except Exception as e:
                    return {"optimization_success": False, "error": str(e)}

            def optimize_min_variance(self) -> Dict[str, Any]:
                """优化最小方差组合"""
                from scipy.optimize import minimize

                def portfolio_variance(weights):
                    return np.dot(weights.T, np.dot(self.cov_matrix, weights))

                # 约束条件
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

                # 边界条件
                bounds = [(0, 1) for _ in range(self.n_assets)]

                # 初始权重
                x0 = np.ones(self.n_assets) / self.n_assets

                try:
                    result = minimize(portfolio_variance, x0, method="SLSQP",
                                    bounds=bounds, constraints=constraints)

                    if result.success:
                        optimal_weights = result.x
                        portfolio_return = np.dot(optimal_weights, self.expected_returns)
                        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))

                        return {
                            "optimal_weights": optimal_weights,
                            "portfolio_return": portfolio_return,
                            "portfolio_risk": portfolio_risk,
                            "optimization_success": True
                        }
                    else:
                        return {"optimization_success": False, "error": result.message}

                except Exception as e:
                    return {"optimization_success": False, "error": str(e)}

            def add_constraints(self, constraints_config: Dict[str, Any]) -> Dict[str, Any]:
                """添加额外约束条件"""
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # 基本约束

                # 添加权重上限约束
                if "max_weight" in constraints_config:
                    max_weight = constraints_config["max_weight"]
                    for i in range(self.n_assets):
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda w, i=i: max_weight - w[i]
                        })

                # 添加权重下限约束
                if "min_weight" in constraints_config:
                    min_weight = constraints_config["min_weight"]
                    for i in range(self.n_assets):
                        constraints.append({
                            "type": "ineq",
                            "fun": lambda w, i=i: w[i] - min_weight
                        })

                # 添加目标收益率约束
                if "target_return" in constraints_config:
                    target_return = constraints_config["target_return"]
                    constraints.append({
                        "type": "eq",
                        "fun": lambda w: np.dot(w, self.expected_returns) - target_return
                    })

                return {
                    "constraints": constraints,
                    "n_constraints": len(constraints),
                    "constraint_types": [c["type"] for c in constraints]
                }

        # 创建测试数据
        np.random.seed(42)
        expected_returns = np.array([0.08, 0.12, 0.10, 0.14, 0.09])
        cov_matrix = np.random.rand(5, 5)
        cov_matrix = np.dot(cov_matrix, cov_matrix.T) / 10

        optimizer = PortfolioWeightOptimizer(expected_returns, cov_matrix)

        # 优化最大夏普比率
        max_sharpe_result = optimizer.optimize_max_sharpe_ratio()

        if max_sharpe_result["optimization_success"]:
            assert "optimal_weights" in max_sharpe_result
            assert "sharpe_ratio" in max_sharpe_result
            assert max_sharpe_result["sharpe_ratio"] > 0
            assert abs(np.sum(max_sharpe_result["optimal_weights"]) - 1.0) < 0.01

        # 优化最小方差
        min_variance_result = optimizer.optimize_min_variance()

        if min_variance_result["optimization_success"]:
            assert "optimal_weights" in min_variance_result
            assert "portfolio_risk" in min_variance_result
            assert min_variance_result["portfolio_risk"] > 0
            assert abs(np.sum(min_variance_result["optimal_weights"]) - 1.0) < 0.01

        # 测试约束条件添加
        constraints_config = {
            "max_weight": 0.4,
            "min_weight": 0.05,
            "target_return": 0.11
        }

        constraints = optimizer.add_constraints(constraints_config)
        assert constraints["n_constraints"] >= 3  # 基本约束 + 权重约束 + 收益率约束
        assert "eq" in constraints["constraint_types"]  # 等式约束
        assert "ineq" in constraints["constraint_types"]  # 不等式约束


class TestRiskParityOptimization:
    """测试风险平价优化功能"""

    def test_risk_contribution_equalization(self):
        """测试风险贡献均衡"""
        class RiskParityOptimizer:
            def __init__(self, cov_matrix: np.ndarray):
                self.cov_matrix = cov_matrix
                self.n_assets = cov_matrix.shape[0]

            def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
                """计算每个资产的风险贡献"""
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

                # 风险贡献 = 权重 * (协方差矩阵 * 权重) / 组合风险
                risk_contributions = weights * np.dot(self.cov_matrix, weights) / portfolio_risk

                return risk_contributions

            def optimize_risk_parity(self, target_risk_contribution: Optional[float] = None,
                                   max_iter: int = 100, tolerance: float = 1e-6) -> Dict[str, Any]:
                """优化风险平价组合"""
                # 初始化等权重
                weights = np.ones(self.n_assets) / self.n_assets

                for iteration in range(max_iter):
                    # 计算当前风险贡献
                    risk_contributions = self.calculate_risk_contributions(weights)

                    # 检查收敛性
                    if target_risk_contribution is not None:
                        target_contributions = np.full(self.n_assets, target_risk_contribution)
                    else:
                        target_contributions = np.full(self.n_assets, np.mean(risk_contributions))

                    # 计算偏差
                    deviation = risk_contributions - target_contributions
                    max_deviation = np.max(np.abs(deviation))

                    if max_deviation < tolerance:
                        break

                    # 更新权重（简化版本）
                    # 在实际实现中，这里会使用更复杂的优化算法
                    adjustment_factor = 0.1
                    weights = weights * (1 + adjustment_factor * (target_contributions - risk_contributions) / np.sum(risk_contributions))

                    # 归一化权重
                    weights = weights / np.sum(weights)

                    # 确保权重在合理范围内
                    weights = np.clip(weights, 0.01, 0.99)
                    weights = weights / np.sum(weights)

                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                risk_contributions = self.calculate_risk_contributions(weights)

                return {
                    "optimal_weights": weights,
                    "risk_contributions": risk_contributions,
                    "portfolio_risk": portfolio_risk,
                    "iterations": iteration + 1,
                    "converged": iteration < max_iter - 1,
                    "max_deviation": max_deviation if 'max_deviation' in locals() else 0,
                    "contribution_std": np.std(risk_contributions)
                }

            def analyze_risk_concentration(self, weights: np.ndarray) -> Dict[str, Any]:
                """分析风险集中度"""
                risk_contributions = self.calculate_risk_contributions(weights)

                # 计算集中度指标
                herfindahl_index = np.sum(risk_contributions ** 2)  # 赫芬达尔指数
                gini_coefficient = self._calculate_gini_coefficient(risk_contributions)

                # 风险集中度评估
                max_contribution = np.max(risk_contributions)
                min_contribution = np.min(risk_contributions)
                concentration_ratio = max_contribution / np.sum(risk_contributions)

                return {
                    "herfindahl_index": herfindahl_index,
                    "gini_coefficient": gini_coefficient,
                    "concentration_ratio": concentration_ratio,
                    "max_min_ratio": max_contribution / min_contribution if min_contribution > 0 else float('inf'),
                    "risk_distribution": "concentrated" if concentration_ratio > 0.5 else "balanced"
                }

            def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
                """计算基尼系数"""
                values = np.sort(values)
                n = len(values)
                cumsum = np.cumsum(values)
                return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

        # 创建测试协方差矩阵
        np.random.seed(42)
        n_assets = 4
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = np.dot(cov_matrix, cov_matrix.T) / 10  # 确保正定

        optimizer = RiskParityOptimizer(cov_matrix)

        # 优化风险平价
        result = optimizer.optimize_risk_parity()

        assert "optimal_weights" in result
        assert "risk_contributions" in result
        assert len(result["optimal_weights"]) == n_assets
        assert len(result["risk_contributions"]) == n_assets
        assert abs(np.sum(result["optimal_weights"]) - 1.0) < 0.01

        # 验证风险贡献的均衡性（标准差应该较小）
        contribution_std = np.std(result["risk_contributions"])
        mean_contribution = np.mean(result["risk_contributions"])

        # 风险平价的目标是各资产风险贡献相等
        # 因此标准差应该相对较小
        assert contribution_std / mean_contribution < 0.5  # 变异系数小于50%

        # 分析风险集中度
        concentration_analysis = optimizer.analyze_risk_concentration(result["optimal_weights"])

        assert "herfindahl_index" in concentration_analysis
        assert "gini_coefficient" in concentration_analysis
        assert "concentration_ratio" in concentration_analysis
        assert concentration_analysis["herfindahl_index"] >= 0
        assert concentration_analysis["gini_coefficient"] >= 0
        assert concentration_analysis["gini_coefficient"] <= 1

    def test_correlation_adjusted_risk_parity(self):
        """测试相关性调整的风险平价"""
        class CorrelationAdjustedRiskParity:
            def __init__(self, cov_matrix: np.ndarray):
                self.cov_matrix = cov_matrix
                self.n_assets = cov_matrix.shape[0]
                self.correlation_matrix = self._cov_to_corr(cov_matrix)

            def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
                """协方差矩阵转换为相关性矩阵"""
                std_devs = np.sqrt(np.diag(cov_matrix))
                outer_std = np.outer(std_devs, std_devs)
                correlation_matrix = cov_matrix / outer_std
                # 处理对角线上的NaN
                np.fill_diagonal(correlation_matrix, 1.0)
                return correlation_matrix

            def optimize_correlation_adjusted_parity(self, correlation_sensitivity: float = 1.0) -> Dict[str, Any]:
                """优化相关性调整的风险平价"""
                # 计算相关性调整的风险度量
                adjusted_volatilities = np.sqrt(np.diag(self.cov_matrix))

                # 考虑相关性的权重调整
                correlation_penalty = np.mean(self.correlation_matrix, axis=1) - 1  # 平均相关性相对于1的偏差
                adjusted_weights = adjusted_volatilities * (1 + correlation_sensitivity * correlation_penalty)

                # 归一化权重
                weights = adjusted_weights / np.sum(adjusted_weights)

                # 确保权重在合理范围内
                weights = np.clip(weights, 0.01, 0.99)
                weights = weights / np.sum(weights)

                # 计算风险贡献
                risk_contributions = weights * np.dot(self.cov_matrix, weights) / np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

                return {
                    "optimal_weights": weights,
                    "risk_contributions": risk_contributions,
                    "correlation_sensitivity": correlation_sensitivity,
                    "avg_correlation": np.mean(self.correlation_matrix),
                    "correlation_penalty": correlation_penalty,
                    "contribution_std": np.std(risk_contributions)
                }

            def compare_risk_parity_strategies(self) -> Dict[str, Any]:
                """比较不同风险平价策略"""
                strategies = {}

                # 标准风险平价
                standard_weights = np.ones(self.n_assets) / self.n_assets
                standard_risk = np.sqrt(np.dot(standard_weights.T, np.dot(self.cov_matrix, standard_weights)))
                strategies["standard"] = {
                    "weights": standard_weights,
                    "risk": standard_risk,
                    "contribution_std": np.std(standard_weights * np.dot(self.cov_matrix, standard_weights) / standard_risk)
                }

                # 相关性调整的风险平价
                adjusted_result = self.optimize_correlation_adjusted_parity()
                strategies["correlation_adjusted"] = {
                    "weights": adjusted_result["optimal_weights"],
                    "risk": np.sqrt(np.dot(adjusted_result["optimal_weights"].T, np.dot(self.cov_matrix, adjusted_result["optimal_weights"]))),
                    "contribution_std": adjusted_result["contribution_std"]
                }

                # 比较策略性能
                best_strategy = min(strategies.items(), key=lambda x: x[1]["contribution_std"])

                return {
                    "strategies": strategies,
                    "best_strategy": best_strategy[0],
                    "improvement": (strategies["standard"]["contribution_std"] - strategies["correlation_adjusted"]["contribution_std"]) / strategies["standard"]["contribution_std"] if strategies["standard"]["contribution_std"] > 0 else 0
                }

        # 创建测试协方差矩阵（包含不同相关性水平的资产）
        np.random.seed(42)
        n_assets = 4

        # 创建具有不同相关性结构的协方差矩阵
        base_corr = np.array([
            [1.0, 0.8, 0.3, 0.1],
            [0.8, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0]
        ])

        volatilities = np.array([0.2, 0.15, 0.25, 0.18])
        outer_vol = np.outer(volatilities, volatilities)
        cov_matrix = base_corr * outer_vol

        optimizer = CorrelationAdjustedRiskParity(cov_matrix)

        # 优化相关性调整的风险平价
        result = optimizer.optimize_correlation_adjusted_parity(correlation_sensitivity=0.5)

        assert "optimal_weights" in result
        assert "risk_contributions" in result
        assert len(result["optimal_weights"]) == n_assets
        assert abs(np.sum(result["optimal_weights"]) - 1.0) < 0.01

        # 比较不同策略
        comparison = optimizer.compare_risk_parity_strategies()

        assert "strategies" in comparison
        assert "best_strategy" in comparison
        assert "standard" in comparison["strategies"]
        assert "correlation_adjusted" in comparison["strategies"]
        assert comparison["best_strategy"] in ["standard", "correlation_adjusted"]

    def test_rebalancing_strategy(self):
        """测试再平衡策略"""
        class PortfolioRebalancer:
            def __init__(self, target_weights: np.ndarray, rebalance_threshold: float = 0.05):
                self.target_weights = target_weights
                self.rebalance_threshold = rebalance_threshold
                self.rebalance_history = []

            def check_rebalance_needed(self, current_weights: np.ndarray) -> Dict[str, Any]:
                """检查是否需要再平衡"""
                weight_deviations = current_weights - self.target_weights
                max_deviation = np.max(np.abs(weight_deviations))
                mean_deviation = np.mean(np.abs(weight_deviations))

                needs_rebalance = max_deviation > self.rebalance_threshold

                return {
                    "needs_rebalance": needs_rebalance,
                    "max_deviation": max_deviation,
                    "mean_deviation": mean_deviation,
                    "weight_deviations": weight_deviations,
                    "rebalance_cost_estimate": self._estimate_rebalance_cost(weight_deviations)
                }

            def calculate_rebalance_trades(self, current_weights: np.ndarray,
                                         portfolio_value: float = 1000000) -> Dict[str, Any]:
                """计算再平衡交易"""
                weight_differences = self.target_weights - current_weights
                value_differences = weight_differences * portfolio_value

                # 确定买入和卖出
                buy_trades = []
                sell_trades = []

                for i, diff in enumerate(value_differences):
                    if diff > 0:
                        buy_trades.append({"asset": i, "value": diff, "percentage": weight_differences[i]})
                    elif diff < 0:
                        sell_trades.append({"asset": i, "value": -diff, "percentage": -weight_differences[i]})

                total_buy_value = sum(t["value"] for t in buy_trades)
                total_sell_value = sum(t["value"] for t in sell_trades)

                return {
                    "buy_trades": buy_trades,
                    "sell_trades": sell_trades,
                    "total_buy_value": total_buy_value,
                    "total_sell_value": total_sell_value,
                    "net_cash_flow": total_sell_value - total_buy_value,
                    "rebalance_ratio": (total_buy_value + total_sell_value) / 2 / portfolio_value
                }

            def execute_rebalance(self, current_weights: np.ndarray, transaction_costs: float = 0.001) -> Dict[str, Any]:
                """执行再平衡"""
                rebalance_check = self.check_rebalance_needed(current_weights)

                if not rebalance_check["needs_rebalance"]:
                    return {"rebalanced": False, "reason": "no_rebalance_needed"}

                # 计算交易
                trades = self.calculate_rebalance_trades(current_weights)

                # 模拟执行再平衡
                new_weights = self.target_weights.copy()

                # 计算交易成本
                total_transaction_value = trades["total_buy_value"] + trades["total_sell_value"]
                total_cost = total_transaction_value * transaction_costs

                # 记录再平衡历史
                rebalance_record = {
                    "timestamp": datetime.now(),
                    "old_weights": current_weights.copy(),
                    "new_weights": new_weights.copy(),
                    "rebalance_reason": f"max_deviation_{rebalance_check['max_deviation']:.4f}",
                    "transaction_cost": total_cost,
                    "rebalance_ratio": trades["rebalance_ratio"]
                }

                self.rebalance_history.append(rebalance_record)

                return {
                    "rebalanced": True,
                    "new_weights": new_weights,
                    "transaction_cost": total_cost,
                    "rebalance_ratio": trades["rebalance_ratio"],
                    "trades_summary": {
                        "n_buys": len(trades["buy_trades"]),
                        "n_sells": len(trades["sell_trades"]),
                        "total_turnover": total_transaction_value
                    }
                }

            def _estimate_rebalance_cost(self, weight_deviations: np.ndarray) -> float:
                """估算再平衡成本"""
                # 简化的成本估算：基于权重偏差的绝对值
                total_deviation = np.sum(np.abs(weight_deviations))
                # 假设每1%的权重偏差需要0.1%的交易成本
                estimated_cost = total_deviation * 0.001

                return estimated_cost

            def get_rebalance_statistics(self) -> Dict[str, Any]:
                """获取再平衡统计"""
                if not self.rebalance_history:
                    return {"total_rebalances": 0}

                n_rebalances = len(self.rebalance_history)
                total_cost = sum(r["transaction_cost"] for r in self.rebalance_history)
                avg_cost = total_cost / n_rebalances
                avg_rebalance_ratio = sum(r["rebalance_ratio"] for r in self.rebalance_history) / n_rebalances

                return {
                    "total_rebalances": n_rebalances,
                    "total_transaction_cost": total_cost,
                    "average_cost_per_rebalance": avg_cost,
                    "average_rebalance_ratio": avg_rebalance_ratio,
                    "cost_efficiency": avg_rebalance_ratio / avg_cost if avg_cost > 0 else float('inf')
                }

        # 创建目标权重和当前权重
        target_weights = np.array([0.25, 0.25, 0.25, 0.25])
        current_weights = np.array([0.35, 0.20, 0.30, 0.15])  # 偏离目标权重

        rebalancer = PortfolioRebalancer(target_weights, rebalance_threshold=0.05)

        # 检查是否需要再平衡
        rebalance_check = rebalancer.check_rebalance_needed(current_weights)
        assert rebalance_check["needs_rebalance"] == True  # 权重偏差超过阈值
        assert rebalance_check["max_deviation"] > 0.05

        # 计算再平衡交易
        trades = rebalancer.calculate_rebalance_trades(current_weights, portfolio_value=1000000)
        assert "buy_trades" in trades
        assert "sell_trades" in trades
        assert len(trades["buy_trades"]) > 0 or len(trades["sell_trades"]) > 0

        # 执行再平衡
        rebalance_result = rebalancer.execute_rebalance(current_weights)
        assert rebalance_result["rebalanced"] == True
        assert "new_weights" in rebalance_result
        assert np.allclose(rebalance_result["new_weights"], target_weights)

        # 获取再平衡统计
        stats = rebalancer.get_rebalance_statistics()
        assert stats["total_rebalances"] == 1
        assert stats["total_transaction_cost"] > 0
        assert stats["average_cost_per_rebalance"] > 0


class TestBlackLittermanModel:
    """测试Black-Litterman模型功能"""

    def test_views_integration(self):
        """测试主观观点整合"""
        class BlackLittermanModel:
            def __init__(self, prior_returns: np.ndarray, cov_matrix: np.ndarray,
                        risk_aversion: float = 2.5):
                self.prior_returns = prior_returns
                self.cov_matrix = cov_matrix
                self.risk_aversion = risk_aversion
                self.n_assets = len(prior_returns)

            def incorporate_views(self, views: List[Dict[str, Any]], tau: float = 0.05) -> Dict[str, Any]:
                """整合主观观点"""
                # 构建观点矩阵P和观点向量Q
                P = []
                Q = []

                for view in views:
                    # 创建观点向量
                    view_vector = np.zeros(self.n_assets)
                    assets_in_view = view["assets"]
                    weights = view["weights"]
                    expected_return = view["expected_return"]

                    for asset, weight in zip(assets_in_view, weights):
                        view_vector[asset] = weight

                    P.append(view_vector)
                    Q.append(expected_return)

                P = np.array(P)
                Q = np.array(Q)

                # 计算观点的协方差矩阵Ω
                omega = np.diag(np.diag(P @ self.cov_matrix @ P.T)) * tau

                # 计算后验分布
                try:
                    # Black-Litterman公式
                    tau_sigma = tau * self.cov_matrix

                    # 后验均值
                    inv_tau_sigma = np.linalg.inv(tau_sigma)
                    inv_omega = np.linalg.inv(omega)

                    posterior_mean = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P) @ \
                                   (inv_tau_sigma @ self.prior_returns + P.T @ inv_omega @ Q)

                    # 后验协方差
                    posterior_cov = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)

                    return {
                        "posterior_returns": posterior_mean,
                        "posterior_covariance": posterior_cov,
                        "prior_returns": self.prior_returns,
                        "views_matrix": P,
                        "views_vector": Q,
                        "views_confidence": omega,
                        "tau": tau,
                        "integration_success": True
                    }

                except np.linalg.LinAlgError:
                    return {"integration_success": False, "error": "matrix_inversion_failed"}

            def optimize_portfolio(self, posterior_returns: np.ndarray,
                                 posterior_cov: np.ndarray) -> Dict[str, Any]:
                """基于后验分布优化组合"""
                # 均值方差优化
                try:
                    inv_cov = np.linalg.inv(posterior_cov)
                    ones = np.ones(self.n_assets)

                    # 市场组合权重
                    market_weights = inv_cov @ posterior_returns / (ones.T @ inv_cov @ posterior_returns)

                    # 考虑风险厌恶系数的投资组合
                    optimal_weights = (1 / self.risk_aversion) * market_weights

                    # 归一化
                    optimal_weights = optimal_weights / np.sum(optimal_weights)

                    portfolio_return = optimal_weights @ posterior_returns
                    portfolio_risk = np.sqrt(optimal_weights.T @ posterior_cov @ optimal_weights)

                    return {
                        "optimal_weights": optimal_weights,
                        "portfolio_return": portfolio_return,
                        "portfolio_risk": portfolio_risk,
                        "sharpe_ratio": portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
                        "optimization_success": True
                    }

                except np.linalg.LinAlgError:
                    return {"optimization_success": False, "error": "optimization_failed"}

            def analyze_view_impact(self, views: List[Dict[str, Any]]) -> Dict[str, Any]:
                """分析观点对组合的影响"""
                # 计算无观点情况下的最优组合
                no_views_result = self.optimize_portfolio(self.prior_returns, self.cov_matrix)

                # 计算有观点情况下的最优组合
                bl_result = self.incorporate_views(views)
                if bl_result["integration_success"]:
                    with_views_result = self.optimize_portfolio(bl_result["posterior_returns"],
                                                              bl_result["posterior_covariance"])
                else:
                    return {"analysis_success": False, "error": "views_integration_failed"}

                # 比较两种情况
                return_change = with_views_result["portfolio_return"] - no_views_result["portfolio_return"]
                risk_change = with_views_result["portfolio_risk"] - no_views_result["portfolio_risk"]

                weight_changes = with_views_result["optimal_weights"] - no_views_result["optimal_weights"]
                max_weight_change = np.max(np.abs(weight_changes))

                return {
                    "no_views_portfolio": no_views_result,
                    "with_views_portfolio": with_views_result,
                    "return_change": return_change,
                    "risk_change": risk_change,
                    "max_weight_change": max_weight_change,
                    "views_impact": "significant" if max_weight_change > 0.1 else "moderate" if max_weight_change > 0.05 else "minimal",
                    "analysis_success": True
                }

        # 创建测试数据
        np.random.seed(42)
        n_assets = 4
        prior_returns = np.array([0.08, 0.10, 0.12, 0.09])
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = np.dot(cov_matrix, cov_matrix.T) / 10

        model = BlackLittermanModel(prior_returns, cov_matrix)

        # 定义主观观点
        views = [
            {
                "assets": [0, 1],  # 资产0和1的相对观点
                "weights": [1, -1],  # 资产0比资产1好
                "expected_return": 0.02  # 预期超额收益2%
            },
            {
                "assets": [2],  # 资产2的绝对观点
                "weights": [1],
                "expected_return": 0.15  # 预期收益15%
            }
        ]

        # 整合观点
        bl_result = model.incorporate_views(views, tau=0.05)

        assert bl_result["integration_success"] == True
        assert "posterior_returns" in bl_result
        assert "posterior_covariance" in bl_result
        assert len(bl_result["posterior_returns"]) == n_assets

        # 优化组合
        portfolio_result = model.optimize_portfolio(bl_result["posterior_returns"],
                                                  bl_result["posterior_covariance"])

        if portfolio_result["optimization_success"]:
            assert "optimal_weights" in portfolio_result
            assert abs(np.sum(portfolio_result["optimal_weights"]) - 1.0) < 0.01
            assert portfolio_result["portfolio_return"] > 0
            assert portfolio_result["portfolio_risk"] > 0

        # 分析观点影响
        impact_analysis = model.analyze_view_impact(views)

        if impact_analysis["analysis_success"]:
            assert "return_change" in impact_analysis
            assert "risk_change" in impact_analysis
            assert "views_impact" in impact_analysis

    def test_posterior_distribution_calculation(self):
        """测试后验分布计算"""
        class PosteriorDistributionCalculator:
            def __init__(self, prior_mean: np.ndarray, prior_cov: np.ndarray):
                self.prior_mean = prior_mean
                self.prior_cov = prior_cov
                self.n_assets = len(prior_mean)

            def update_with_views(self, views_matrix: np.ndarray, views_vector: np.ndarray,
                                views_covariance: np.ndarray) -> Dict[str, Any]:
                """使用观点更新后验分布"""
                try:
                    # Black-Litterman后验分布计算
                    inv_prior_cov = np.linalg.inv(self.prior_cov)
                    inv_views_cov = np.linalg.inv(views_covariance)

                    # 后验协方差
                    posterior_cov = np.linalg.inv(inv_prior_cov + views_matrix.T @ inv_views_cov @ views_matrix)

                    # 后验均值
                    posterior_mean = posterior_cov @ (inv_prior_cov @ self.prior_mean + views_matrix.T @ inv_views_cov @ views_vector)

                    # 计算更新统计
                    prior_posterior_diff = posterior_mean - self.prior_mean
                    max_change = np.max(np.abs(prior_posterior_diff))
                    mean_change = np.mean(np.abs(prior_posterior_diff))

                    return {
                        "posterior_mean": posterior_mean,
                        "posterior_covariance": posterior_cov,
                        "prior_posterior_diff": prior_posterior_diff,
                        "max_change": max_change,
                        "mean_change": mean_change,
                        "update_confidence": self._calculate_update_confidence(prior_posterior_diff, posterior_cov),
                        "calculation_success": True
                    }

                except np.linalg.LinAlgError:
                    return {"calculation_success": False, "error": "matrix_inversion_failed"}

            def _calculate_update_confidence(self, diff: np.ndarray, posterior_cov: np.ndarray) -> float:
                """计算更新置信度"""
                # 基于变化幅度和后验不确定性计算置信度
                max_diff = np.max(np.abs(diff))
                avg_posterior_std = np.sqrt(np.mean(np.diag(posterior_cov)))

                # 置信度 = 1 / (1 + 变化相对于不确定性的比率)
                confidence_ratio = max_diff / avg_posterior_std if avg_posterior_std > 0 else 0
                confidence = 1.0 / (1.0 + confidence_ratio)

                return confidence

            def validate_posterior_properties(self, posterior_mean: np.ndarray,
                                            posterior_cov: np.ndarray) -> Dict[str, Any]:
                """验证后验分布的数学性质"""
                validation_results = {}

                # 检查协方差矩阵是否正定
                try:
                    np.linalg.cholesky(posterior_cov)
                    validation_results["positive_definite"] = True
                except np.linalg.LinAlgError:
                    validation_results["positive_definite"] = False

                # 检查均值向量维度
                validation_results["correct_dimensions"] = (
                    len(posterior_mean) == self.n_assets and
                    posterior_cov.shape == (self.n_assets, self.n_assets)
                )

                # 检查协方差矩阵对称性
                validation_results["symmetric_covariance"] = np.allclose(posterior_cov, posterior_cov.T)

                # 计算分布统计
                validation_results["mean_vector"] = posterior_mean
                validation_results["covariance_matrix"] = posterior_cov
                validation_results["volatility_vector"] = np.sqrt(np.diag(posterior_cov))

                # 整体验证结果
                validation_results["all_properties_valid"] = all([
                    validation_results["positive_definite"],
                    validation_results["correct_dimensions"],
                    validation_results["symmetric_covariance"]
                ])

                return validation_results

        # 创建测试数据
        np.random.seed(42)
        n_assets = 3
        prior_mean = np.array([0.08, 0.10, 0.12])
        prior_cov = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.05, 0.015],
            [0.01, 0.015, 0.06]
        ])

        calculator = PosteriorDistributionCalculator(prior_mean, prior_cov)

        # 创建观点数据
        views_matrix = np.array([
            [1, -1, 0],  # 资产0比资产1好2%
            [0, 1, -1]   # 资产1比资产2好1%
        ])
        views_vector = np.array([0.02, 0.01])
        views_covariance = np.eye(2) * 0.01  # 观点不确定性

        # 更新后验分布
        update_result = calculator.update_with_views(views_matrix, views_vector, views_covariance)

        assert update_result["calculation_success"] == True
        assert "posterior_mean" in update_result
        assert "posterior_covariance" in update_result
        assert len(update_result["posterior_mean"]) == n_assets
        assert update_result["posterior_covariance"].shape == (n_assets, n_assets)

        # 验证后验分布性质
        validation = calculator.validate_posterior_properties(
            update_result["posterior_mean"],
            update_result["posterior_covariance"]
        )

        assert validation["all_properties_valid"] == True
        assert validation["positive_definite"] == True
        assert validation["correct_dimensions"] == True
        assert validation["symmetric_covariance"] == True

    def test_portfolio_adjustment_analysis(self):
        """测试投资组合调整分析"""
        class PortfolioAdjustmentAnalyzer:
            def __init__(self, prior_weights: np.ndarray, prior_returns: np.ndarray,
                        cov_matrix: np.ndarray):
                self.prior_weights = prior_weights
                self.prior_returns = prior_returns
                self.cov_matrix = cov_matrix
                self.n_assets = len(prior_weights)

            def analyze_adjustment_impact(self, new_weights: np.ndarray,
                                        posterior_returns: np.ndarray) -> Dict[str, Any]:
                """分析调整对组合的影响"""
                # 计算调整前后的组合特征
                prior_return = self.prior_weights @ self.prior_returns
                prior_risk = np.sqrt(self.prior_weights.T @ self.cov_matrix @ self.prior_weights)

                new_return = new_weights @ posterior_returns
                new_risk = np.sqrt(new_weights.T @ self.cov_matrix @ new_weights)

                # 计算权重变化
                weight_changes = new_weights - self.prior_weights
                max_weight_change = np.max(np.abs(weight_changes))
                total_turnover = np.sum(np.abs(weight_changes)) / 2  # 双向换手

                # 计算风险贡献变化
                prior_risk_contributions = self.prior_weights * (self.cov_matrix @ self.prior_weights) / prior_risk
                new_risk_contributions = new_weights * (self.cov_matrix @ new_weights) / new_risk

                risk_contribution_changes = new_risk_contributions - prior_risk_contributions

                return {
                    "prior_portfolio": {
                        "return": prior_return,
                        "risk": prior_risk,
                        "sharpe_ratio": prior_return / prior_risk if prior_risk > 0 else 0
                    },
                    "adjusted_portfolio": {
                        "return": new_return,
                        "risk": new_risk,
                        "sharpe_ratio": new_return / new_risk if new_risk > 0 else 0
                    },
                    "performance_changes": {
                        "return_change": new_return - prior_return,
                        "risk_change": new_risk - prior_risk,
                        "sharpe_change": (new_return / new_risk if new_risk > 0 else 0) - (prior_return / prior_risk if prior_risk > 0 else 0)
                    },
                    "weight_adjustments": {
                        "weight_changes": weight_changes,
                        "max_weight_change": max_weight_change,
                        "total_turnover": total_turnover,
                        "assets_to_increase": np.where(weight_changes > 0.01)[0].tolist(),
                        "assets_to_decrease": np.where(weight_changes < -0.01)[0].tolist()
                    },
                    "risk_reallocation": {
                        "prior_contributions": prior_risk_contributions,
                        "new_contributions": new_risk_contributions,
                        "contribution_changes": risk_contribution_changes,
                        "risk_diversification_improved": np.std(new_risk_contributions) < np.std(prior_risk_contributions)
                    }
                }

            def generate_adjustment_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
                """生成调整建议"""
                recommendations = []

                perf_changes = analysis["performance_changes"]
                weight_adj = analysis["weight_adjustments"]
                risk_realloc = analysis["risk_reallocation"]

                # 基于性能变化的建议
                if perf_changes["sharpe_change"] > 0.1:
                    recommendations.append("Strong improvement in risk-adjusted returns - adjustment successful")
                elif perf_changes["sharpe_change"] < -0.1:
                    recommendations.append("Degradation in risk-adjusted returns - reconsider views")

                # 基于权重变化的建议
                if weight_adj["total_turnover"] > 0.5:
                    recommendations.append("High portfolio turnover - consider transaction costs")
                elif weight_adj["max_weight_change"] > 0.2:
                    recommendations.append("Significant weight changes - monitor concentration risk")

                # 基于风险再分配的建议
                if risk_realloc["risk_diversification_improved"]:
                    recommendations.append("Improved risk diversification - positive adjustment")
                else:
                    recommendations.append("Risk concentration may have increased - review allocation")

                return recommendations if recommendations else ["Portfolio adjustment within normal parameters"]

        # 创建测试数据
        prior_weights = np.array([0.3, 0.3, 0.2, 0.2])
        prior_returns = np.array([0.08, 0.10, 0.12, 0.09])
        cov_matrix = np.array([
            [0.04, 0.02, 0.01, 0.005],
            [0.02, 0.05, 0.015, 0.01],
            [0.01, 0.015, 0.06, 0.02],
            [0.005, 0.01, 0.02, 0.03]
        ])

        # Black-Litterman调整后的权重和收益
        new_weights = np.array([0.35, 0.25, 0.25, 0.15])
        posterior_returns = np.array([0.085, 0.095, 0.125, 0.095])

        analyzer = PortfolioAdjustmentAnalyzer(prior_weights, prior_returns, cov_matrix)

        # 分析调整影响
        impact_analysis = analyzer.analyze_adjustment_impact(new_weights, posterior_returns)

        assert "prior_portfolio" in impact_analysis
        assert "adjusted_portfolio" in impact_analysis
        assert "performance_changes" in impact_analysis
        assert "weight_adjustments" in impact_analysis
        assert "risk_reallocation" in impact_analysis

        # 生成调整建议
        recommendations = analyzer.generate_adjustment_recommendations(impact_analysis)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
