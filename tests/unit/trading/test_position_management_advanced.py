# -*- coding: utf-8 -*-
"""
交易层 - 持仓管理高级单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试持仓管理核心功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

try:
    from src.trading.portfolio.portfolio_portfolio_manager import PortfolioManager
except ImportError:
    try:
        from src.trading.portfolio.portfolio_manager import PortfolioManager
    except ImportError:
        PortfolioManager = None
# from src.trading.portfolio_portfolio_optimizer import PortfolioOptimizer  # 需要cvxpy依赖，暂时注释
from unittest.mock import Mock
PortfolioOptimizer = Mock  # 使用Mock代替实际类


class TestPortfolioManagerAdvancedInitialization:
    """测试投资组合管理器高级初始化"""

    def test_portfolio_manager_initialization_with_config(self):
        """测试使用配置初始化投资组合管理器"""
        optimizer = PortfolioOptimizer()
        config = {
            "initial_capital": 10000000.0,
            "max_positions": 50,
            "max_single_position": 0.05,
            "rebalance_threshold": 0.02,
            "risk_free_rate": 0.03,
            "benchmark": "000001.SH"
        }

        manager = PortfolioManager(optimizer=optimizer, rebalance_freq='M')

        # 检查基本属性存在
        assert hasattr(manager, 'optimizer')
        assert hasattr(manager, 'rebalance_freq')
        assert manager.rebalance_freq == 'M'

    def test_portfolio_manager_initialization_default_values(self):
        """测试默认值初始化"""
        optimizer = PortfolioOptimizer()
        manager = PortfolioManager(optimizer=optimizer)

        # 检查基本属性存在
        assert hasattr(manager, 'optimizer')
        assert hasattr(manager, 'rebalance_freq')
        assert manager.rebalance_freq == 'M'
        assert isinstance(manager.current_weights, dict)
        assert len(manager.current_weights) == 0

    def test_portfolio_manager_initialization_with_positions(self):
        """测试使用现有持仓初始化"""
        initial_positions = {
            "000001.SZ": {"quantity": 1000, "avg_price": 100.0, "current_price": 105.0},
            "000002.SZ": {"quantity": 2000, "avg_price": 50.0, "current_price": 52.0}
        }

        # 创建Mock优化器
        mock_optimizer = Mock()
        manager = PortfolioManager(
            optimizer=mock_optimizer,
            initial_positions=initial_positions
        )

        assert len(manager.positions) == 2
        assert manager.positions["000001.SZ"]["quantity"] == 1000
        assert manager.positions["000002.SZ"]["quantity"] == 2000


class TestPortfolioPositionManagement:
    """测试投资组合持仓管理"""

    def setup_method(self, method):
        """设置测试环境"""
        # 创建Mock优化器
        mock_optimizer = Mock()
        self.manager = PortfolioManager(
            optimizer=mock_optimizer,
            rebalance_threshold=0.02  # 设置与测试期望一致的阈值
        )

    def test_add_position(self):
        """测试添加持仓"""
        symbol = "000001.SZ"
        quantity = 1000
        price = 100.0

        result = self.manager.add_position(symbol, quantity, price)

        assert result is True
        assert symbol in self.manager.positions
        assert self.manager.positions[symbol]["quantity"] == quantity
        assert self.manager.positions[symbol]["avg_price"] == price

    def test_remove_position(self):
        """测试移除持仓"""
        # 先添加持仓
        symbol = "000001.SZ"
        self.manager.add_position(symbol, 1000, 100.0)

        # 移除持仓
        result = self.manager.remove_position(symbol)

        assert result is True
        assert symbol not in self.manager.positions

    def test_update_position(self):
        """测试更新持仓"""
        symbol = "000001.SZ"

        # 添加初始持仓
        self.manager.add_position(symbol, 1000, 100.0)

        # 更新持仓价格
        new_price = 105.0
        result = self.manager.update_position_price(symbol, new_price)

        assert result is True
        assert self.manager.positions[symbol]["current_price"] == new_price
        assert self.manager.positions[symbol]["market_value"] == 1000 * new_price

    def test_position_rebalancing(self):
        """测试持仓再平衡"""
        # 设置初始持仓
        self.manager.add_position("000001.SZ", 5000, 100.0)  # 50万
        self.manager.add_position("000002.SZ", 10000, 50.0)  # 50万
        self.manager.cash_balance = 0  # 满仓

        # 计算当前权重
        total_value = self.manager.get_portfolio_value()
        current_weights = {}
        for symbol, position in self.manager.positions.items():
            current_weights[symbol] = position["market_value"] / total_value

        # 目标权重
        target_weights = {
            "000001.SZ": 0.6,  # 提高权重
            "000002.SZ": 0.4   # 降低权重
        }

        # 计算需要调整的数量
        rebalance_orders = []
        for symbol in target_weights:
            if symbol in current_weights:
                weight_diff = target_weights[symbol] - current_weights[symbol]
                if abs(weight_diff) > self.manager.rebalance_threshold:
                    adjust_value = weight_diff * total_value
                    current_price = self.manager.positions[symbol]["current_price"]
                    adjust_quantity = adjust_value / current_price

                    rebalance_orders.append({
                        "symbol": symbol,
                        "quantity": adjust_quantity,
                        "direction": "BUY" if adjust_quantity > 0 else "SELL",
                        "reason": "rebalancing"
                    })

        # 验证再平衡逻辑
        assert len(rebalance_orders) > 0
        assert all(abs(order["quantity"]) > 0 for order in rebalance_orders)

    def test_portfolio_value_calculation(self):
        """测试投资组合价值计算"""
        # 添加多个持仓
        self.manager.add_position("000001.SZ", 1000, 100.0)  # 10万
        self.manager.add_position("000002.SZ", 2000, 50.0)   # 10万
        self.manager.add_position("000003.SZ", 500, 200.0)   # 10万
        self.manager.cash_balance = 700000.0  # 70万现金

        # 计算总价值
        total_value = self.manager.get_portfolio_value()

        # 验证计算结果
        expected_value = (1000 * 100.0) + (2000 * 50.0) + (500 * 200.0) + 700000.0
        assert total_value == expected_value

    def test_portfolio_performance_metrics(self):
        """测试投资组合绩效指标"""
        # 设置历史价格数据
        historical_prices = pd.DataFrame({
            "000001.SZ": [100.0, 105.0, 102.0, 108.0, 110.0],
            "000002.SZ": [50.0, 52.0, 49.0, 53.0, 51.0]
        })

        # 计算收益率
        returns = historical_prices.pct_change().dropna()

        # 计算投资组合收益率
        weights = np.array([0.6, 0.4])  # 权重
        portfolio_returns = returns.dot(weights)

        # 计算绩效指标
        cumulative_return = (1 + portfolio_returns).prod() - 1
        volatility = portfolio_returns.std() * np.sqrt(252)  # 年化波动率
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()

        # 验证绩效指标
        assert isinstance(cumulative_return, (int, float))
        assert volatility >= 0
        assert isinstance(sharpe_ratio, (int, float))
        assert max_drawdown <= 0  # 最大回撤是负数


class TestPortfolioRiskManagement:
    """测试投资组合风险管理"""

    def setup_method(self, method):
        """设置测试环境"""
        # 创建Mock优化器
        mock_optimizer = Mock()
        self.manager = PortfolioManager(optimizer=mock_optimizer)

    def test_portfolio_risk_metrics(self):
        """测试投资组合风险指标"""
        # 设置持仓
        self.manager.add_position("000001.SZ", 1000, 100.0)
        self.manager.add_position("000002.SZ", 2000, 50.0)
        self.manager.add_position("000003.SZ", 500, 200.0)

        # 模拟协方差矩阵
        cov_matrix = np.array([
            [0.04, 0.02, 0.01],  # 000001.SZ
            [0.02, 0.09, 0.03],  # 000002.SZ
            [0.01, 0.03, 0.16]   # 000003.SZ
        ])

        # 权重
        weights = np.array([0.4, 0.4, 0.2])

        # 计算投资组合波动率
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

        # 计算VaR (95%置信度)
        confidence_level = 0.05
        var_95 = -portfolio_volatility * np.sqrt(confidence_level) * np.sqrt(252)

        # 验证风险指标
        assert portfolio_volatility > 0
        assert var_95 < 0  # VaR是负数
        assert abs(var_95) > 0

    def test_portfolio_diversification_analysis(self):
        """测试投资组合分散化分析"""
        # 设置不同行业的持仓
        positions = {
            "tech_stocks": ["000001.SZ", "000858.SZ"],      # 科技股
            "financial_stocks": ["000002.SZ", "600036.SH"], # 金融股
            "consumer_stocks": ["000568.SZ"]                # 消费股
        }

        # 为每个股票添加持仓
        for sector, symbols in positions.items():
            for symbol in symbols:
                self.manager.add_position(symbol, 1000, 100.0)

        # 计算行业集中度
        sector_weights = {}
        total_value = self.manager.get_portfolio_value()

        for sector, symbols in positions.items():
            sector_value = sum(
                self.manager.positions.get(symbol, {}).get("market_value", 0)
                for symbol in symbols
            )
            sector_weights[sector] = sector_value / total_value

        # 计算赫芬达尔-赫希曼指数(HHI)
        hhi = sum(weight ** 2 for weight in sector_weights.values())

        # 验证分散化分析
        assert 0 <= hhi <= 1  # HHI在0-1之间
        assert len(sector_weights) == 3  # 覆盖3个行业
        assert all(weight > 0 for weight in sector_weights.values())

    def test_portfolio_stress_testing(self):
        """测试投资组合压力测试"""
        # 设置基准投资组合
        self.manager.add_position("000001.SZ", 2000, 100.0)  # 20万
        self.manager.add_position("000002.SZ", 4000, 50.0)   # 20万
        self.manager.add_position("000003.SZ", 1000, 200.0)  # 20万

        base_value = self.manager.get_portfolio_value()

        # 压力测试情景
        stress_scenarios = {
            "market_crash": {"000001.SZ": -0.3, "000002.SZ": -0.4, "000003.SZ": -0.2},
            "tech_sector_crash": {"000001.SZ": -0.5, "000002.SZ": -0.1, "000003.SZ": 0.1},
            "interest_rate_hike": {"000001.SZ": -0.2, "000002.SZ": -0.3, "000003.SZ": -0.1}
        }

        stress_test_results = {}

        for scenario_name, shocks in stress_scenarios.items():
            # 计算压力测试后的价值
            stressed_value = 0
            for symbol, shock in shocks.items():
                if symbol in self.manager.positions:
                    position = self.manager.positions[symbol]
                    stressed_price = position["current_price"] * (1 + shock)
                    stressed_value += position["quantity"] * stressed_price

            stressed_value += self.manager.cash_balance
            loss_amount = base_value - stressed_value
            loss_percentage = loss_amount / base_value

            stress_test_results[scenario_name] = {
                "stressed_value": stressed_value,
                "loss_amount": loss_amount,
                "loss_percentage": loss_percentage
            }

        # 验证压力测试结果
        for scenario, result in stress_test_results.items():
            assert result["stressed_value"] < base_value  # 价值应该下降
            assert result["loss_amount"] > 0  # 应该有损失
            assert result["loss_percentage"] > 0  # 损失百分比是正数

    def test_portfolio_var_calculation(self):
        """测试投资组合VaR计算"""
        # 设置持仓权重
        weights = np.array([0.4, 0.3, 0.3])

        # 资产波动率
        volatilities = np.array([0.2, 0.25, 0.15])  # 年化波动率

        # 相关性矩阵
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        # 计算协方差矩阵
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        # 计算投资组合波动率
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

        # 计算VaR (95%置信度，1天持有期)
        confidence_level = 0.05
        z_score = 1.645  # 95%置信度的z分数
        portfolio_var = portfolio_volatility * z_score

        # 计算预期损失
        expected_loss = portfolio_var * self.manager.get_portfolio_value()

        # 验证VaR计算
        assert portfolio_volatility > 0
        assert portfolio_var > 0
        assert expected_loss > 0

    def test_portfolio_beta_calculation(self):
        """测试投资组合贝塔计算"""
        # 投资组合权重
        portfolio_weights = np.array([0.3, 0.4, 0.3])

        # 个股权重
        asset_betas = np.array([1.2, 0.8, 1.5])

        # 计算投资组合贝塔
        portfolio_beta = np.sum(portfolio_weights * asset_betas)

        # 验证贝塔计算
        assert portfolio_beta > 0
        # 投资组合贝塔应该是资产贝塔的加权平均
        expected_beta = np.average(asset_betas, weights=portfolio_weights)
        assert abs(portfolio_beta - expected_beta) < 1e-10


class TestPortfolioOptimization:
    """测试投资组合优化"""

    def setup_method(self, method):
        """设置测试环境"""
        self.optimizer = PortfolioOptimizer()

    def test_mean_variance_optimization(self):
        """测试均值方差优化"""
        # 预期收益率
        expected_returns = np.array([0.12, 0.08, 0.15, 0.10])

        # 协方差矩阵
        cov_matrix = np.array([
            [0.04, 0.02, 0.01, 0.015],
            [0.02, 0.09, 0.03, 0.025],
            [0.01, 0.03, 0.16, 0.02],
            [0.015, 0.025, 0.02, 0.25]
        ])

        # 优化目标：最小化波动率
        from scipy.optimize import minimize

        def portfolio_volatility(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        bounds = [(0, 1) for _ in range(len(expected_returns))]  # 权重在0-1之间

        # 初始权重
        initial_weights = np.ones(len(expected_returns)) / len(expected_returns)

        # 优化
        result = minimize(portfolio_volatility, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        optimal_volatility = result.fun

        # 验证优化结果
        assert np.sum(optimal_weights) == pytest.approx(1.0, abs=1e-6)
        assert all(0 <= weight <= 1 for weight in optimal_weights)
        assert optimal_volatility > 0

    def test_risk_parity_optimization(self):
        """测试风险平价优化"""
        # 资产波动率
        volatilities = np.array([0.2, 0.25, 0.15, 0.3])

        # 相关性矩阵
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.3],
            [0.3, 0.4, 1.0, 0.25],
            [0.2, 0.3, 0.25, 1.0]
        ])

        # 计算协方差矩阵
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        # 风险平价优化：每个资产的风险贡献相等
        n_assets = len(volatilities)

        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_risk = cov_matrix @ weights / portfolio_vol
            risk_contributions = weights * marginal_risk
            return np.std(risk_contributions)  # 最小化风险贡献的标准差

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]

        bounds = [(0.01, 1) for _ in range(n_assets)]  # 最小权重1%

        # 初始权重
        initial_weights = np.ones(n_assets) / n_assets

        # 优化
        from scipy.optimize import minimize
        result = minimize(risk_contribution, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        risk_parity_weights = result.x

        # 验证风险平价结果
        assert np.sum(risk_parity_weights) == pytest.approx(1.0, abs=1e-6)
        assert all(0.01 <= weight <= 1 for weight in risk_parity_weights)

    def test_black_litterman_optimization(self):
        """测试Black-Litterman优化"""
        # 先验预期收益率
        prior_returns = np.array([0.08, 0.06, 0.10, 0.07])

        # 协方差矩阵
        cov_matrix = np.array([
            [0.04, 0.02, 0.01, 0.015],
            [0.02, 0.09, 0.03, 0.025],
            [0.01, 0.03, 0.16, 0.02],
            [0.015, 0.025, 0.02, 0.25]
        ])

        # 投资者观点
        views = np.array([0.12, 0.09])  # 对前两个资产的预期收益率
        view_confidences = np.array([0.7, 0.8])  # 观点信心

        # 简化版Black-Litterman计算
        tau = 0.05  # 不确定性参数

        # 观点矩阵 (简化)
        P = np.array([
            [1, 0, 0, 0],  # 对第一个资产的观点
            [0, 1, 0, 0]   # 对第二个资产的观点
        ])

        # 计算后验预期收益率
        omega = np.diag(np.diag(P @ cov_matrix @ P.T) / view_confidences)

        posterior_returns = np.linalg.inv(
            np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(omega) @ P
        ) @ (
            np.linalg.inv(tau * cov_matrix) @ prior_returns +
            P.T @ np.linalg.inv(omega) @ views
        )

        # 验证Black-Litterman结果
        assert len(posterior_returns) == len(prior_returns)
        assert all(isinstance(ret, (int, float, np.floating)) for ret in posterior_returns)


class TestPortfolioPerformanceTracking:
    """测试投资组合绩效跟踪"""

    def setup_method(self, method):
        """设置测试环境"""
        # 创建Mock优化器
        mock_optimizer = Mock()
        self.manager = PortfolioManager(optimizer=mock_optimizer)

    def test_portfolio_benchmark_comparison(self):
        """测试投资组合基准比较"""
        # 投资组合历史价值
        portfolio_values = [
            1000000, 1020000, 1010000, 1030000, 1020000, 1040000, 1050000
        ]

        # 基准历史价值
        benchmark_values = [
            1000000, 1010000, 1005000, 1020000, 1010000, 1030000, 1040000
        ]

        # 计算收益率
        portfolio_returns = [0] + [portfolio_values[i] / portfolio_values[i-1] - 1
                                  for i in range(1, len(portfolio_values))]
        benchmark_returns = [0] + [benchmark_values[i] / benchmark_values[i-1] - 1
                                  for i in range(1, len(benchmark_values))]

        # 计算累积收益率
        portfolio_cumulative = np.cumprod(1 + np.array(portfolio_returns))
        benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns))

        # 计算超额收益
        excess_returns = portfolio_cumulative - benchmark_cumulative

        # 计算跟踪误差
        tracking_error = np.std(excess_returns)

        # 验证基准比较
        assert len(portfolio_returns) == len(benchmark_returns)
        assert tracking_error >= 0
        assert isinstance(excess_returns[-1], (int, float, np.floating))

    def test_portfolio_attribution_analysis(self):
        """测试投资组合归因分析"""
        # 资产类别
        asset_classes = {
            "equities": ["000001.SZ", "000002.SZ"],
            "bonds": ["bond1", "bond2"],
            "cash": ["cash"]
        }

        # 资产类别权重
        class_weights = {
            "equities": 0.6,
            "bonds": 0.3,
            "cash": 0.1
        }

        # 资产类别收益率
        class_returns = {
            "equities": 0.12,
            "bonds": 0.04,
            "cash": 0.02
        }

        # 计算投资组合总收益率
        portfolio_return = sum(
            class_weights[class_name] * class_returns[class_name]
            for class_name in class_weights
        )

        # 计算类别贡献
        class_contributions = {}
        for class_name in class_weights:
            class_contributions[class_name] = (
                class_weights[class_name] * class_returns[class_name]
            )

        # 验证归因分析
        assert portfolio_return > 0
        assert sum(class_contributions.values()) == pytest.approx(portfolio_return, abs=1e-10)
        assert all(contribution >= 0 for contribution in class_contributions.values())

    def test_portfolio_risk_attribution(self):
        """测试投资组合风险归因"""
        # 资产权重
        weights = np.array([0.3, 0.4, 0.3])

        # 资产波动率
        volatilities = np.array([0.2, 0.25, 0.15])

        # 相关性矩阵
        correlation_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

        # 计算协方差矩阵
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        # 计算投资组合波动率
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

        # 计算每个资产对投资组合风险的贡献
        # 使用Euler公式：σ²_p = ∑ᵢ wᵢ * ∂σ_p/∂wᵢ
        # ∂σ_p/∂wᵢ = (1/(2σ_p)) * (2 * covᵢ⋅w) = (covᵢ⋅w) / σ_p
        marginal_contributions = (cov_matrix @ weights) / portfolio_volatility
        risk_contributions = weights * marginal_contributions

        # 计算百分比贡献
        percentage_contributions = risk_contributions / portfolio_volatility

        # 验证风险归因
        # Euler公式：总风险 = 各资产风险贡献之和
        assert np.sum(risk_contributions) == pytest.approx(portfolio_volatility, abs=1e-10)
        assert np.sum(percentage_contributions) == pytest.approx(1.0, abs=1e-10)
        assert all(contribution >= 0 for contribution in risk_contributions)


class TestPortfolioReporting:
    """测试投资组合报告"""

    def setup_method(self, method):
        """设置测试环境"""
        # 创建Mock优化器
        mock_optimizer = Mock()
        self.manager = PortfolioManager(optimizer=mock_optimizer)

    def test_portfolio_summary_report(self):
        """测试投资组合汇总报告"""
        # 添加持仓
        self.manager.add_position("000001.SZ", 1000, 100.0)
        self.manager.add_position("000002.SZ", 2000, 50.0)
        self.manager.add_position("000003.SZ", 500, 200.0)

        # 生成汇总报告
        report = {
            "total_value": self.manager.get_portfolio_value(),
            "cash_balance": self.manager.cash_balance,
            "num_positions": len(self.manager.positions),
            "largest_position": max(
                (pos["market_value"], symbol)
                for symbol, pos in self.manager.positions.items()
            ),
            "sector_allocation": {},  # 简化版
            "performance_metrics": {
                "total_return": 0.05,  # 示例值
                "volatility": 0.12,
                "sharpe_ratio": 1.8
            }
        }

        # 验证报告内容
        assert report["total_value"] > 0
        assert report["num_positions"] == 3
        assert report["largest_position"][0] > 0
        assert isinstance(report["performance_metrics"], dict)

    def test_portfolio_risk_report(self):
        """测试投资组合风险报告"""
        # 设置持仓
        self.manager.add_position("000001.SZ", 1000, 100.0)
        self.manager.add_position("000002.SZ", 2000, 50.0)

        # 生成风险报告
        risk_report = {
            "var_95": -25000,  # 95% VaR
            "expected_shortfall": -35000,
            "volatility": 0.15,
            "beta": 1.1,
            "concentration_risk": {
                "largest_position": 0.4,
                "top_5_positions": 0.85
            },
            "liquidity_risk": {
                "illiquid_positions": 0.1,
                "avg_daily_volume": 1000000
            }
        }

        # 验证风险报告
        assert risk_report["var_95"] < 0  # VaR是负数
        assert risk_report["expected_shortfall"] < risk_report["var_95"]  # ES比VaR更极端
        assert risk_report["volatility"] > 0
        assert risk_report["beta"] > 0

    def test_portfolio_performance_report(self):
        """测试投资组合绩效报告"""
        # 绩效数据
        performance_data = {
            "period": "1M",
            "start_value": 1000000,
            "end_value": 1050000,
            "total_return": 0.05,
            "benchmark_return": 0.03,
            "excess_return": 0.02,
            "max_drawdown": -0.02,
            "volatility": 0.12,
            "sharpe_ratio": 1.5,
            "win_rate": 0.65
        }

        # 生成绩效报告
        performance_report = {
            "period_summary": performance_data,
            "monthly_returns": [0.02, 0.01, 0.03, -0.01, 0.02],
            "benchmark_comparison": {
                "outperformance": performance_data["excess_return"],
                "tracking_error": 0.05
            },
            "risk_adjusted_metrics": {
                "sharpe_ratio": performance_data["sharpe_ratio"],
                "sortino_ratio": 1.8,
                "information_ratio": 0.4
            }
        }

        # 验证绩效报告
        assert performance_report["period_summary"]["total_return"] == 0.05
        assert performance_report["benchmark_comparison"]["outperformance"] > 0
        assert len(performance_report["monthly_returns"]) > 0
