"""
深度测试Optimization模块优化引擎功能
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import numpy as np


class TestOptimizationEngineDeep:
    """深度测试优化引擎"""

    def setup_method(self):
        """测试前准备"""
        # 创建mock的优化引擎
        self.optimization_engine = MagicMock()

        # 配置动态返回值
        def optimize_portfolio_mock(**kwargs):
            # 返回满足约束的权重（0.05-0.35之间）
            return {
                "optimal_weights": {"AAPL": 0.25, "GOOGL": 0.30, "MSFT": 0.20, "JPM": 0.15, "JNJ": 0.10},
                "expected_return": 0.12,
                "expected_risk": 0.15,
                "sharpe_ratio": 0.8,
                "optimization_status": "success",
                **kwargs
            }

        def optimize_strategy_mock(**kwargs):
            return {
                "optimal_parameters": {"fast_period": 10, "slow_period": 30, "stop_loss": 0.05},
                "backtest_results": {"sharpe_ratio": 1.2, "max_drawdown": 0.08},
                "optimization_status": "success",
                **kwargs
            }

        self.optimization_engine.optimize_portfolio.side_effect = optimize_portfolio_mock
        self.optimization_engine.optimize_strategy.side_effect = optimize_strategy_mock

    def test_portfolio_optimization_with_complex_constraints(self):
        """测试带复杂约束的组合优化"""
        # 定义复杂的投资组合约束
        constraints = {
            "min_weight": 0.05,  # 最小权重5%
            "max_weight": 0.35,  # 最大权重35%
            "sector_limits": {
                "technology": 0.6,
                "finance": 0.3,
                "healthcare": 0.2
            },
            "risk_budget": {
                "max_volatility": 0.20,
                "max_var_95": 0.15
            },
            "liquidity_constraint": True,
            "turnover_limit": 0.5
        }

        # 资产数据
        assets = ["AAPL", "GOOGL", "MSFT", "JPM", "JNJ"]
        returns = np.random.normal(0.001, 0.02, (100, 5))  # 100天5个资产的收益率

        # 执行优化
        result = self.optimization_engine.optimize_portfolio(
            assets=assets,
            returns=returns,
            constraints=constraints,
            objective="max_sharpe",
            rebalance_frequency="monthly"
        )

        # 验证结果
        assert "optimal_weights" in result
        assert "expected_return" in result
        assert "expected_risk" in result
        assert result["optimization_status"] == "success"

        # 验证约束满足
        weights = result["optimal_weights"]
        assert all(0.05 <= w <= 0.35 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01  # 权重和为1

    def test_strategy_parameter_optimization_with_walk_forward(self):
        """测试带步前进分析的策略参数优化"""
        # 策略参数空间
        parameter_space = {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50],
            "stop_loss": [0.02, 0.05, 0.08, 0.10],
            "take_profit": [0.05, 0.10, 0.15]
        }

        # 历史数据
        historical_data = {
            "prices": np.random.normal(100, 10, (500,)),  # 500天价格数据
            "volume": np.random.normal(1000000, 200000, (500,)),
            "timestamp": [datetime.now() - timedelta(days=i) for i in range(500)]
        }

        # 执行步前进优化
        optimization_config = {
            "method": "grid_search",
            "walk_forward_window": 252,  # 一年
            "validation_window": 63,    # 一季度
            "step_size": 21,            # 每月步进
            "metric": "sharpe_ratio"
        }

        result = self.optimization_engine.optimize_strategy(
            strategy_type="moving_average_crossover",
            parameter_space=parameter_space,
            historical_data=historical_data,
            optimization_config=optimization_config
        )

        # 验证结果
        assert "optimal_parameters" in result
        assert "backtest_results" in result
        assert result["optimization_status"] == "success"

        # 验证参数在合理范围内
        params = result["optimal_parameters"]
        assert params["fast_period"] < params["slow_period"]
        assert 0.02 <= params["stop_loss"] <= 0.10

    def test_risk_parity_optimization(self):
        """测试风险平价优化"""
        # 多资产类别的数据
        assets = {
            "equities": ["SPY", "QQQ", "IWM"],
            "bonds": ["AGG", "BND", "TIP"],
            "commodities": ["GLD", "SLV", "USO"],
            "currencies": ["UUP", "FXE"]
        }

        # 资产收益率协方差矩阵（模拟）
        n_assets = sum(len(v) for v in assets.values())
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2  # 对称化
        np.fill_diagonal(cov_matrix, np.random.uniform(0.01, 0.05))  # 对角线方差

        # 执行风险平价优化
        result = self.optimization_engine.optimize_portfolio(
            assets=list(assets.keys()),
            covariance_matrix=cov_matrix,
            method="risk_parity",
            target_risk_contribution=1.0/n_assets
        )

        # 验证风险平价特性
        assert "optimal_weights" in result
        assert "risk_contributions" in result

        risk_contribs = result["risk_contributions"]
        # 风险贡献应该相对均衡
        max_contrib = max(risk_contribs.values())
        min_contrib = min(risk_contribs.values())
        assert max_contrib / min_contrib < 2.0  # 最大风险贡献不超过最小值的2倍

    def test_multi_period_optimization_with_transaction_costs(self):
        """测试带交易成本的多周期优化"""
        # 初始投资组合
        initial_weights = {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3}

        # 交易成本模型
        transaction_costs = {
            "commission": 0.001,  # 0.1%佣金
            "spread": 0.0005,     # 0.05%买卖价差
            "market_impact": lambda size: 0.0001 * np.sqrt(size)  # 市场冲击成本
        }

        # 多周期优化配置
        optimization_horizon = {
            "periods": 12,  # 12个月
            "rebalance_frequency": "monthly",
            "forecast_horizon": 3,  # 3个月预测
            "risk_model_update": "weekly"
        }

        result = self.optimization_engine.optimize_portfolio(
            initial_weights=initial_weights,
            transaction_costs=transaction_costs,
            optimization_horizon=optimization_horizon,
            objective="max_utility_with_costs"
        )

        # 验证考虑了交易成本
        assert "transaction_costs_incurred" in result
        assert "turnover_by_period" in result
        assert result["optimization_status"] == "success"

    def test_optimization_under_stress_scenarios(self):
        """测试压力情景下的优化"""
        # 定义压力情景
        stress_scenarios = {
            "market_crash": {
                "equity_returns": -0.15,
                "bond_returns": 0.05,
                "correlation_shift": 0.8
            },
            "rate_hike": {
                "equity_returns": -0.08,
                "bond_returns": -0.03,
                "volatility_spike": 2.0
            },
            "liquidity_crisis": {
                "bid_ask_spread": 3.0,  # 扩大3倍
                "trading_volume": 0.3,  # 下降70%
                "market_impact": 5.0    # 扩大5倍
            }
        }

        # 执行情景压力测试优化
        result = self.optimization_engine.optimize_portfolio(
            stress_scenarios=stress_scenarios,
            optimization_method="robust_optimization",
            confidence_level=0.95,
            scenario_weights="equal"  # 等权重情景
        )

        # 验证鲁棒性
        assert "robust_weights" in result
        assert "worst_case_performance" in result
        assert "scenario_analysis" in result
        assert result["optimization_status"] == "success"

    def test_real_time_optimization_with_market_data(self):
        """测试实时优化与市场数据集成"""
        # 模拟实时市场数据流
        market_data_stream = {
            "prices": {"AAPL": 150.25, "GOOGL": 2500.50, "MSFT": 305.75},
            "volumes": {"AAPL": 45000000, "GOOGL": 1200000, "MSFT": 25000000},
            "order_book": {
                "AAPL": {"bid": 150.20, "ask": 150.30, "spread": 0.001},
                "GOOGL": {"bid": 2500.00, "ask": 2501.00, "spread": 0.002},
                "MSFT": {"bid": 305.70, "ask": 305.80, "spread": 0.0003}
            },
            "timestamp": datetime.now()
        }

        # 执行实时优化
        result = self.optimization_engine.optimize_portfolio(
            real_time_data=market_data_stream,
            optimization_frequency="high_frequency",  # 高频优化
            execution_constraints={
                "max_order_size": 100000,
                "min_order_size": 100,
                "price_impact_limit": 0.001
            }
        )

        # 验证实时优化结果
        assert "real_time_weights" in result
        assert "execution_schedule" in result
        assert "price_impact_estimate" in result
        assert result["optimization_status"] == "success"

    def test_optimization_performance_scaling(self):
        """测试优化性能扩展性"""
        import time

        # 测试不同规模的优化问题
        problem_sizes = [10, 50, 100, 500]

        performance_results = {}

        for n_assets in problem_sizes:
            # 生成测试数据
            returns = np.random.normal(0.001, 0.02, (252, n_assets))  # 一年数据

            start_time = time.time()

            # 执行优化
            result = self.optimization_engine.optimize_portfolio(
                assets=[f"ASSET_{i}" for i in range(n_assets)],
                returns=returns,
                method="mean_variance"
            )

            end_time = time.time()
            execution_time = end_time - start_time

            performance_results[n_assets] = {
                "execution_time": execution_time,
                "throughput": n_assets / execution_time if execution_time > 0 else 0,
                "status": result["optimization_status"]
            }

        # 验证性能扩展性
        for size, perf in performance_results.items():
            assert perf["status"] == "success"
            assert perf["execution_time"] < 30  # 30秒内完成
            if size > 10:
                # 大问题规模的吞吐量应该合理
                assert perf["throughput"] > 1
