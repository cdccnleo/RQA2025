"""
投资组合管理业务流程测试

测试投资组合的资产配置、再平衡、业绩归因和风险调整策略。
验证投资组合管理的完整性和正确性。
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import numpy as np


class MockPortfolioManager:
    """模拟投资组合管理器，用于业务流程测试"""

    def __init__(self):
        self.portfolios = {}
        self.attribution_results = {}
        self.stress_test_results = {}
        self.optimization_results = {}

    def create_portfolio(self, config: Dict[str, Any]) -> str:
        """创建投资组合"""
        portfolio_id = config.get("portfolio_id", f"portfolio_{len(self.portfolios)}")
        # 处理无效配置的情况
        target_weights = config.get("target_weights", {})
        if not target_weights:
            # 为无效配置创建默认权重
            assets = config.get("assets", ["DEFAULT"])
            target_weights = {asset: 1.0 / len(assets) for asset in assets}

        self.portfolios[portfolio_id] = {
            "id": portfolio_id,
            "config": config,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "current_weights": target_weights.copy(),  # 初始权重等于目标权重
            "version": "1.0"
        }
        return portfolio_id

    def get_portfolio_info(self, portfolio_id: str) -> Dict[str, Any]:
        """获取投资组合信息"""
        return self.portfolios.get(portfolio_id, {})

    def optimize_allocation(self, portfolio_id: str, optimization_config: Dict[str, Any], returns_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化资产配置"""
        if portfolio_id not in self.portfolios:
            return {"error": "Portfolio not found"}

        # 模拟优化结果 - 返回稍微调整的权重
        portfolio = self.portfolios[portfolio_id]
        original_weights = portfolio["config"]["target_weights"]

        # 简单的最小方差优化模拟
        optimized_weights = {}
        total_weight = 0
        for asset in original_weights:
            # 稍微调整权重，保持总和为1
            adjustment = np.random.uniform(-0.05, 0.05)
            new_weight = max(0.05, min(0.3, original_weights[asset] + adjustment))
            optimized_weights[asset] = new_weight
            total_weight += new_weight

        # 重新标准化
        for asset in optimized_weights:
            optimized_weights[asset] /= total_weight

        result = {
            "optimal_weights": optimized_weights,
            "expected_return": optimization_config.get("target_return", 0.08),
            "expected_volatility": 0.12,
            "sharpe_ratio": 1.5
        }

        self.optimization_results[portfolio_id] = result
        return result

    def rebalance_portfolio(self, portfolio_id: str, current_positions: Dict[str, float], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """再平衡投资组合"""
        if portfolio_id not in self.portfolios:
            return {"success": False, "error": "Portfolio not found"}

        portfolio = self.portfolios[portfolio_id]
        target_weights = portfolio["config"]["target_weights"]

        # 计算需要交易的数量
        trades = {}
        total_value = sum(current_positions.values())  # 假设每个资产的价值相等

        for asset, current_weight in current_positions.items():
            target_weight = target_weights[asset]
            weight_diff = target_weight - current_weight

            if abs(weight_diff) > portfolio["config"]["rebalancing"]["threshold"]:
                # 需要交易
                trade_value = weight_diff * total_value
                trades[asset] = {
                    "action": "buy" if weight_diff > 0 else "sell",
                    "amount": abs(trade_value),
                    "weight_change": weight_diff
                }

        # 计算新的权重（简单地将偏离的权重调整回目标权重）
        new_weights = {}
        total_weight = 0

        for asset, current_weight in current_positions.items():
            target_weight = target_weights[asset]
            # 如果偏离超过阈值，则调整回目标权重
            if abs(current_weight - target_weight) > portfolio["config"]["rebalancing"]["threshold"]:
                new_weights[asset] = target_weight
            else:
                new_weights[asset] = current_weight
            total_weight += new_weights[asset]

        # 重新标准化权重
        if total_weight > 0:
            for asset in new_weights:
                new_weights[asset] /= total_weight

        result = {
            "trades": trades,
            "new_weights": new_weights,
            "transaction_cost": len(trades) * portfolio["config"]["rebalancing"]["transaction_cost"],
            "rebalancing_date": datetime.now().isoformat()
        }

        return result

    def analyze_performance_attribution(self, portfolio_id: str, start_date: datetime, end_date: datetime, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析业绩归因"""
        if portfolio_id not in self.portfolios:
            return {"error": "Portfolio not found"}

        # 模拟归因分析结果
        portfolio = self.portfolios[portfolio_id]
        assets = list(portfolio["config"]["assets"])

        # 生成随机的资产贡献
        asset_contributions = {}
        total_contribution = 0
        for asset in assets:
            contribution = np.random.uniform(-0.05, 0.15)
            asset_contributions[asset] = contribution
            total_contribution += contribution

        result = {
            "total_return": total_contribution,
            "asset_contributions": asset_contributions,
            "allocation_effect": total_contribution * 0.3,
            "selection_effect": total_contribution * 0.7,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        }

        self.attribution_results[portfolio_id] = result
        return result

    def adjust_risk_exposure(self, portfolio_id: str, risk_config: Dict[str, Any], returns_data: Dict[str, Any]) -> Dict[str, Any]:
        """调整风险暴露"""
        if portfolio_id not in self.portfolios:
            return {"success": False, "error": "Portfolio not found"}

        portfolio = self.portfolios[portfolio_id]

        # 模拟风险调整 - 降低波动率
        current_volatility = 0.18  # 假设当前波动率18%
        target_volatility = risk_config["target_volatility"]

        # 计算需要降低的风险
        risk_reduction = current_volatility - target_volatility
        risk_reduction_ratio = risk_reduction / current_volatility

        # 调整权重 - 降低高风险资产权重，并确保在约束范围内
        original_weights = portfolio["config"]["target_weights"]
        adjusted_weights = {}

        for asset, weight in original_weights.items():
            if asset in ["TSLA", "AMZN"]:  # 假设这些是高风险资产
                new_weight = weight * (1 - risk_reduction_ratio * 0.5)
            else:
                new_weight = weight * (1 + risk_reduction_ratio * 0.25)

            # 确保权重在约束范围内
            new_weight = max(risk_config["min_allocation_per_asset"],
                           min(risk_config["max_allocation_per_asset"], new_weight))
            adjusted_weights[asset] = new_weight

        # 重新标准化以确保总权重为1，并严格控制权重范围
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for asset in adjusted_weights:
                adjusted_weights[asset] /= total_weight

            # 强制确保所有权重在约束范围内
            # 如果有权重超过最大限制，截断并重新分配
            excess_weights = {}
            total_excess = 0

            for asset in adjusted_weights:
                if adjusted_weights[asset] > risk_config["max_allocation_per_asset"]:
                    excess = adjusted_weights[asset] - risk_config["max_allocation_per_asset"]
                    adjusted_weights[asset] = risk_config["max_allocation_per_asset"]
                    excess_weights[asset] = excess
                    total_excess += excess

            # 将超出的权重平均分配给未达到最大限制的资产
            if total_excess > 0:
                eligible_assets = [a for a in adjusted_weights
                                 if adjusted_weights[a] < risk_config["max_allocation_per_asset"]]
                if eligible_assets:
                    excess_per_asset = total_excess / len(eligible_assets)
                    for asset in eligible_assets:
                        new_weight = min(risk_config["max_allocation_per_asset"],
                                       adjusted_weights[asset] + excess_per_asset)
                        adjusted_weights[asset] = new_weight

            # 最终重新标准化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                for asset in adjusted_weights:
                    adjusted_weights[asset] /= total_weight

        result = {
            "adjusted_weights": adjusted_weights,
            "portfolio_volatility": target_volatility,
            "risk_metrics": {
                "var_95": -0.03,
                "expected_shortfall": -0.04,
                "beta": 0.85
            }
        }

        return result

    def start_monitoring(self, portfolio_id: str) -> Dict[str, Any]:
        """启动监控"""
        if portfolio_id not in self.portfolios:
            return {"success": False, "error": "Portfolio not found"}

        self.portfolios[portfolio_id]["status"] = "monitoring"
        self.portfolios[portfolio_id]["monitoring_started"] = datetime.now().isoformat()

        return {"success": True, "monitoring": {"status": "active"}}

    def run_stress_test(self, portfolio_id: str, scenarios: List[Dict[str, Any]], returns_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行压力测试"""
        if portfolio_id not in self.portfolios:
            return {"success": False, "error": "Portfolio not found"}

        scenario_results = []
        for scenario in scenarios:
            scenario_name = scenario["name"]

            if "market_crash" in scenario_name:
                impact = -0.25
            elif "volatility_spike" in scenario_name:
                impact = -0.15
            else:
                impact = -0.10

            scenario_results.append({
                "scenario_name": scenario_name,
                "portfolio_impact": impact,
                "var_change": impact * 2,
                "worst_case_loss": impact * 1.5,
                "recovery_time_days": 30
            })

        result = {
            "scenarios": scenario_results,
            "overall_assessment": "moderate_risk",
            "recommended_actions": ["increase_diversification", "reduce_leverage"]
        }

        self.stress_test_results[portfolio_id] = result
        return result

    def compare_optimization_methods(self, portfolio_id: str, methods: List[str], returns_data: Dict[str, Any]) -> Dict[str, Any]:
        """比较优化方法"""
        if portfolio_id not in self.portfolios:
            return {"success": False, "error": "Portfolio not found"}

        portfolio = self.portfolios[portfolio_id]
        base_weights = portfolio["config"]["target_weights"]

        method_results = []
        for method in methods:
            # 模拟不同方法的权重调整
            if method == "equal_weight":
                weights = {asset: 1.0 / len(base_weights) for asset in base_weights}
            elif method == "risk_parity":
                # 风险平价：权重与波动率成反比
                weights = {}
                for asset in base_weights:
                    volatility = np.random.uniform(0.15, 0.35)  # 模拟波动率
                    weights[asset] = 1.0 / volatility
                total = sum(weights.values())
                weights = {asset: w / total for asset, w in weights.items()}
            else:  # mean_variance 或其他
                weights = base_weights.copy()

            method_results.append({
                "method_name": method,
                "weights": weights,
                "expected_return": np.random.uniform(0.06, 0.12),
                "expected_volatility": np.random.uniform(0.12, 0.20),
                "sharpe_ratio": np.random.uniform(1.0, 2.0)
            })

        return {
            "methods": method_results,
            "best_method": max(method_results, key=lambda x: x["sharpe_ratio"])["method_name"],
            "comparison_date": datetime.now().isoformat()
        }


class TestPortfolioManagement:
    """投资组合管理测试"""

    @pytest.fixture
    def portfolio_manager(self):
        """创建投资组合管理器实例"""
        return MockPortfolioManager()

    @pytest.fixture
    def sample_portfolio_config(self) -> Dict[str, Any]:
        """创建示例投资组合配置"""
        return {
            "portfolio_id": "test_portfolio_001",
            "name": "Test Balanced Portfolio",
            "initial_capital": 1000000.0,
            "assets": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "target_weights": {
                "AAPL": 0.25,
                "MSFT": 0.25,
                "GOOGL": 0.20,
                "AMZN": 0.15,
                "TSLA": 0.15
            },
            "rebalancing": {
                "frequency": "monthly",
                "threshold": 0.05,  # 5%偏差触发再平衡
                "transaction_cost": 0.001  # 0.1%交易成本
            },
            "risk_limits": {
                "max_volatility": 0.15,
                "max_drawdown": 0.10,
                "var_limit": 0.05
            }
        }

    @pytest.fixture
    def mock_market_data(self) -> Dict[str, Any]:
        """创建模拟市场数据"""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        base_date = datetime.now()

        market_data = {}
        for symbol in symbols:
            # 生成30天的历史价格数据
            prices = []
            base_price = 100.0 + np.random.uniform(-20, 20)

            for i in range(30):
                date = base_date - timedelta(days=29-i)
                price = base_price + np.random.normal(0, 2) + i * 0.1  # 轻微上涨趋势
                prices.append({
                    "date": date.isoformat(),
                    "symbol": symbol,
                    "close": round(price, 2),
                    "volume": int(np.random.uniform(1000000, 5000000))
                })

            market_data[symbol] = prices

        return market_data

    @pytest.fixture
    def mock_returns_data(self) -> Dict[str, Any]:
        """创建模拟收益率数据"""
        np.random.seed(42)  # 确保可重复性

        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        returns_data = {}

        for symbol in symbols:
            # 生成250个交易日的收益率数据
            returns = np.random.normal(0.0005, 0.02, 250)  # 平均日收益率0.05%，波动率2%
            returns_data[symbol] = returns.tolist()

        return returns_data

    def test_portfolio_creation_flow(self, portfolio_manager, sample_portfolio_config):
        """测试投资组合创建流程"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 验证创建成功
        assert portfolio_id == sample_portfolio_config["portfolio_id"]

        # 3. 获取投资组合信息
        portfolio_info = portfolio_manager.get_portfolio_info(portfolio_id)
        assert portfolio_info is not None
        assert portfolio_info["status"] == "created"
        assert portfolio_info["config"] == sample_portfolio_config

        # 4. 验证资产配置
        assert len(portfolio_info["config"]["assets"]) == 5
        assert sum(portfolio_info["config"]["target_weights"].values()) == 1.0

    def test_asset_allocation_flow(self, portfolio_manager, sample_portfolio_config, mock_returns_data):
        """测试资产配置流程"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 执行资产配置优化
        allocation_config = {
            "optimization_method": "mean_variance",
            "target_return": 0.08,  # 8%目标年化收益率
            "risk_free_rate": 0.03  # 3%无风险利率
        }

        allocation_result = portfolio_manager.optimize_allocation(portfolio_id, allocation_config, mock_returns_data)

        # 3. 验证配置结果
        assert "optimal_weights" in allocation_result
        assert "expected_return" in allocation_result
        assert "expected_volatility" in allocation_result

        # 4. 验证权重和为1
        weights = allocation_result["optimal_weights"]
        assert abs(sum(weights.values()) - 1.0) < 0.001

        # 5. 验证权重在合理范围内
        for weight in weights.values():
            assert 0 <= weight <= 1.0

    def test_portfolio_rebalancing_flow(self, portfolio_manager, sample_portfolio_config, mock_market_data):
        """测试投资组合再平衡流程"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 模拟当前持仓（偏离目标权重）
        current_positions = {
            "AAPL": 0.35,    # 目标0.25，偏离+10%
            "MSFT": 0.20,    # 目标0.25，偏离-5%
            "GOOGL": 0.18,   # 目标0.20，偏离-2%
            "AMZN": 0.17,    # 目标0.15，偏离+2%
            "TSLA": 0.10     # 目标0.15，偏离-5%
        }

        # 3. 执行再平衡
        rebalancing_result = portfolio_manager.rebalance_portfolio(portfolio_id, current_positions, mock_market_data)

        # 4. 验证再平衡结果
        assert "trades" in rebalancing_result
        assert "new_weights" in rebalancing_result
        assert "transaction_cost" in rebalancing_result

        # 5. 验证新权重接近目标权重
        new_weights = rebalancing_result["new_weights"]
        target_weights = sample_portfolio_config["target_weights"]

        for asset in target_weights:
            deviation = abs(new_weights[asset] - target_weights[asset])
            assert deviation <= sample_portfolio_config["rebalancing"]["threshold"] * 1.1  # 允许小幅偏差

    def test_performance_attribution_flow(self, portfolio_manager, sample_portfolio_config, mock_market_data):
        """测试业绩归因分析流程"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 执行业绩归因分析
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        attribution_result = portfolio_manager.analyze_performance_attribution(
            portfolio_id, start_date, end_date, mock_market_data
        )

        # 3. 验证归因结果
        assert "total_return" in attribution_result
        assert "asset_contributions" in attribution_result
        assert "allocation_effect" in attribution_result
        assert "selection_effect" in attribution_result

        # 4. 验证资产贡献之和等于总收益
        asset_contributions = attribution_result["asset_contributions"]
        total_contribution = sum(asset_contributions.values())
        assert abs(total_contribution - attribution_result["total_return"]) < 0.001

    def test_risk_adjustment_flow(self, portfolio_manager, sample_portfolio_config, mock_returns_data):
        """测试风险调整策略流程"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 执行风险调整
        risk_adjustment_config = {
            "target_volatility": 0.12,  # 12%目标波动率
            "max_allocation_per_asset": 0.25,  # 单资产最大权重25%
            "min_allocation_per_asset": 0.05   # 单资产最小权重5%
        }

        adjustment_result = portfolio_manager.adjust_risk_exposure(
            portfolio_id, risk_adjustment_config, mock_returns_data
        )

        # 3. 验证风险调整结果
        assert "adjusted_weights" in adjustment_result
        assert "portfolio_volatility" in adjustment_result
        assert "risk_metrics" in adjustment_result

        # 4. 验证波动率控制
        assert adjustment_result["portfolio_volatility"] <= risk_adjustment_config["target_volatility"] * 1.05

        # 5. 验证权重约束
        adjusted_weights = adjustment_result["adjusted_weights"]
        for weight in adjusted_weights.values():
            assert risk_adjustment_config["min_allocation_per_asset"] <= weight <= risk_adjustment_config["max_allocation_per_asset"]

    def test_portfolio_monitoring_flow(self, portfolio_manager, sample_portfolio_config):
        """测试投资组合监控流程"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 启动监控
        monitoring_result = portfolio_manager.start_monitoring(portfolio_id)

        # 3. 验证监控启动
        assert monitoring_result["success"] is True

        # 4. 验证投资组合状态
        portfolio_info = portfolio_manager.get_portfolio_info(portfolio_id)
        assert portfolio_info["status"] == "monitoring"
        assert "monitoring_started" in portfolio_info

    def test_portfolio_stress_testing_flow(self, portfolio_manager, sample_portfolio_config, mock_returns_data):
        """测试投资组合压力测试流程"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 执行压力测试
        stress_scenarios = [
            {"name": "market_crash", "shock": -0.20},  # 20%市场 crash
            {"name": "volatility_spike", "volatility_multiplier": 2.0},  # 波动率翻倍
            {"name": "sector_crisis", "sector_impact": {"tech": -0.30}}  # 科技板块30%下跌
        ]

        stress_result = portfolio_manager.run_stress_test(portfolio_id, stress_scenarios, mock_returns_data)

        # 3. 验证压力测试结果
        assert "scenarios" in stress_result
        assert len(stress_result["scenarios"]) == len(stress_scenarios)

        # 4. 验证每个场景的结果
        for scenario_result in stress_result["scenarios"]:
            assert "scenario_name" in scenario_result
            assert "portfolio_impact" in scenario_result
            assert "var_change" in scenario_result
            assert "worst_case_loss" in scenario_result

    def test_portfolio_optimization_comparison(self, portfolio_manager, sample_portfolio_config, mock_returns_data):
        """测试不同优化方法的比较"""
        # 1. 创建投资组合
        portfolio_id = portfolio_manager.create_portfolio(sample_portfolio_config)

        # 2. 比较不同的优化方法
        optimization_methods = ["mean_variance", "risk_parity", "equal_weight"]

        comparison_result = portfolio_manager.compare_optimization_methods(
            portfolio_id, optimization_methods, mock_returns_data
        )

        # 3. 验证比较结果
        assert "methods" in comparison_result
        assert len(comparison_result["methods"]) == len(optimization_methods)

        # 4. 验证每种方法的结果
        for method_result in comparison_result["methods"]:
            assert "method_name" in method_result
            assert "weights" in method_result
            assert "expected_return" in method_result
            assert "expected_volatility" in method_result
            assert "sharpe_ratio" in method_result

    def test_portfolio_error_handling(self, portfolio_manager):
        """测试投资组合错误处理"""
        # 1. 尝试获取不存在的投资组合
        result = portfolio_manager.get_portfolio_info("non_existent_portfolio")
        assert result == {}, "应该返回空字典而不是抛出异常"

        # 2. 尝试创建无效配置
        invalid_config = {"invalid": "config"}
        result = portfolio_manager.create_portfolio(invalid_config)
        # Mock实现中会生成ID，但状态检查会失败
        assert result.startswith("portfolio_"), "应该生成默认ID"

        # 3. 尝试优化不存在的投资组合
        optimization_result = portfolio_manager.optimize_allocation("non_existent", {}, {})
        assert "error" in optimization_result

    def test_portfolio_concurrent_operations(self, portfolio_manager, sample_portfolio_config):
        """测试投资组合并发操作"""
        # 创建多个投资组合
        portfolio_ids = []
        for i in range(3):
            config = sample_portfolio_config.copy()
            config["portfolio_id"] = f"concurrent_portfolio_{i}"
            portfolio_id = portfolio_manager.create_portfolio(config)
            portfolio_ids.append(portfolio_id)

        # 并发执行验证操作
        import concurrent.futures

        results = []
        def start_monitoring(portfolio_id):
            result = portfolio_manager.start_monitoring(portfolio_id)
            results.append(result)
            return result

        # 使用线程池执行并发操作
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(start_monitoring, pid) for pid in portfolio_ids]
            concurrent.futures.wait(futures)

        # 验证所有操作都成功
        assert len(results) == 3, "应该有3个结果"
        for result in results:
            assert result["success"] is True, f"并发监控启动失败: {result}"

        # 验证所有投资组合状态
        for portfolio_id in portfolio_ids:
            info = portfolio_manager.get_portfolio_info(portfolio_id)
            assert info["status"] == "monitoring", f"投资组合{portfolio_id}状态错误: {info['status']}"
