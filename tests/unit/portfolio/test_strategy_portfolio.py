import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.portfolio.strategy_portfolio import (
    StrategyStatus,
    StrategyAllocation,
    PortfolioOptimizer,
    CapitalAllocator,
    PerformanceAttribution,
    PortfolioRiskManager,
    StrategyPortfolio
)

@pytest.fixture
def sample_strategies():
    """创建测试策略"""
    return [
        StrategyAllocation(strategy_id="strategy1", weight=0.6, capital=0.0),
        StrategyAllocation(strategy_id="strategy2", weight=0.4, capital=0.0)
    ]

@pytest.fixture
def sample_returns():
    """生成测试收益率数据"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    return pd.DataFrame({
        "strategy1": np.random.normal(0.001, 0.02, 100),
        "strategy2": np.random.normal(0.0005, 0.015, 100)
    }, index=dates)

@pytest.fixture
def sample_benchmark():
    """生成基准收益率数据"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)
    return pd.Series(np.random.normal(0.0008, 0.018, 100), index=dates)

def test_portfolio_optimizer(sample_strategies, sample_returns):
    """测试组合优化器"""
    optimizer = PortfolioOptimizer(sample_strategies)

    # 测试权重优化
    optimized = optimizer.optimize_weights(sample_returns)
    assert len(optimized) == 2
    assert np.isclose(sum(optimized.values()), 1.0)

    # 验证权重更新
    assert np.isclose(sample_strategies[0].weight, optimized["strategy1"])
    assert np.isclose(sample_strategies[1].weight, optimized["strategy2"])

def test_capital_allocator(sample_strategies):
    """测试资金分配器"""
    allocator = CapitalAllocator(total_capital=1000000, max_strategy_capital=0.3)

    # 测试资金分配
    allocations = allocator.allocate_capital(sample_strategies)
    assert len(allocations) == 2
    assert allocations["strategy1"] == 600000
    assert allocations["strategy2"] == 400000

    # 测试资金限制
    high_weight_strategies = [
        StrategyAllocation(strategy_id="s1", weight=0.8, capital=0.0),
        StrategyAllocation(strategy_id="s2", weight=0.2, capital=0.0)
    ]
    allocations = allocator.allocate_capital(high_weight_strategies)
    assert allocations["s1"] == 300000  # 受限于max_strategy_capital
    assert allocations["s2"] == 200000

def test_performance_attribution(sample_returns, sample_benchmark):
    """测试绩效归因"""
    attribution = PerformanceAttribution()

    # 测试绩效计算
    stats = attribution.add_performance(
        "strategy1",
        sample_returns["strategy1"],
        sample_benchmark
    )
    assert "alpha" in stats
    assert "beta" in stats
    assert "information_ratio" in stats

    # 测试获取绩效
    retrieved = attribution.get_strategy_attribution("strategy1")
    assert retrieved == stats

def test_portfolio_risk_manager(sample_returns):
    """测试组合风险管理"""
    risk_manager = PortfolioRiskManager(
        max_drawdown=0.2,
        max_strategy_risk=0.15,
        correlation_threshold=0.7
    )

    # 测试组合风险检查
    assert risk_manager.check_portfolio_risk(sample_returns)

    # 测试高风险组合
    high_risk_returns = sample_returns.copy()
    high_risk_returns["strategy1"] = np.random.normal(0.01, 0.3, 100)
    assert not risk_manager.check_portfolio_risk(high_risk_returns)

    # 测试高相关性组合
    correlated_returns = pd.DataFrame({
        "s1": np.random.normal(0.001, 0.02, 100),
        "s2": np.random.normal(0.001, 0.02, 100) * 0.9 + 0.1
    })
    assert not risk_manager.check_portfolio_risk(correlated_returns)

def test_strategy_portfolio(sample_returns, sample_benchmark):
    """测试策略组合管理"""
    portfolio = StrategyPortfolio(total_capital=1000000)

    # 测试添加策略
    portfolio.add_strategy("strategy1", initial_weight=0.6)
    portfolio.add_strategy("strategy2", initial_weight=0.4)

    # 测试资金分配
    allocations = portfolio.allocate_capital()
    assert allocations["strategy1"] == 600000
    assert allocations["strategy2"] == 400000

    # 测试权重优化
    optimized = portfolio.optimize_weights(sample_returns)
    assert len(optimized) == 2

    # 测试绩效更新
    stats = portfolio.update_performance(
        "strategy1",
        sample_returns["strategy1"],
        sample_benchmark
    )
    assert "alpha" in stats

    # 测试风险检查
    assert portfolio.check_risk(sample_returns)

@pytest.mark.parametrize("weights,expected_allocation", [
    ([0.8, 0.2], [300000, 200000]),  # 受限于max_strategy_capital
    ([0.6, 0.4], [600000, 400000]),
    ([0.5, 0.5], [500000, 500000])
])
def test_capital_allocation_params(weights, expected_allocation):
    """测试资金分配参数化"""
    strategies = [
        StrategyAllocation(strategy_id=f"s{i+1}", weight=w, capital=0.0)
        for i, w in enumerate(weights)
    ]
    allocator = CapitalAllocator(total_capital=1000000, max_strategy_capital=0.3)
    allocations = allocator.allocate_capital(strategies)

    assert allocations["s1"] == expected_allocation[0]
    assert allocations["s2"] == expected_allocation[1]

def test_invalid_portfolio_weights():
    """测试无效组合权重"""
    strategies = [
        StrategyAllocation(strategy_id="s1", weight=0.7, capital=0.0),
        StrategyAllocation(strategy_id="s2", weight=0.5, capital=0.0)
    ]
    with pytest.raises(ValueError):
        PortfolioOptimizer(strategies)
