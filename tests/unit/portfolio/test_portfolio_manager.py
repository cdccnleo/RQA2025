import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.portfolio.portfolio_manager import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    PortfolioManager,
    StrategyPerformance,
    PortfolioConstraints,
    AttributionFactor
)

@pytest.fixture
def sample_performances():
    """生成测试策略绩效数据"""
    dates = pd.date_range('2023-01-01', periods=100)
    returns = {
        'strategy1': pd.Series(np.random.normal(0.001, 0.02, 100), index=dates),
        'strategy2': pd.Series(np.random.normal(0.0005, 0.015, 100), index=dates),
        'strategy3': pd.Series(np.random.normal(0.0008, 0.018, 100), index=dates)
    }
    return {
        name: StrategyPerformance(
            returns=ret,
            sharpe=ret.mean() / ret.std() * np.sqrt(252),
            max_drawdown=ret.cumsum().expanding().max() - ret.cumsum(),
            turnover=np.random.uniform(0.1, 0.3),
            factor_exposure={
                AttributionFactor.MARKET: np.random.uniform(0.5, 1.5),
                AttributionFactor.SIZE: np.random.uniform(-0.5, 0.5),
                AttributionFactor.VALUE: np.random.uniform(-0.3, 0.3),
                AttributionFactor.MOMENTUM: np.random.uniform(-0.2, 0.4),
                AttributionFactor.VOLATILITY: np.random.uniform(-0.1, 0.1)
            }
        )
        for name, ret in returns.items()
    }

@pytest.fixture
def sample_constraints():
    """生成测试约束条件"""
    return PortfolioConstraints(
        max_weight=0.5,
        min_weight=0.1,
        max_turnover=0.5,
        max_leverage=1.0
    )

def test_equal_weight_optimizer(sample_performances, sample_constraints):
    """测试等权重优化器"""
    optimizer = EqualWeightOptimizer()
    weights = optimizer.optimize(sample_performances, sample_constraints)

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    assert all(np.isclose(w, 1/3, rtol=1e-5) for w in weights.values())

def test_mean_variance_optimizer(sample_performances, sample_constraints):
    """测试均值方差优化器"""
    optimizer = MeanVarianceOptimizer()
    weights = optimizer.optimize(sample_performances, sample_constraints)

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    # 验证权重在约束范围内
    assert all(sample_constraints.min_weight <= w <= sample_constraints.max_weight
              for w in weights.values())

def test_risk_parity_optimizer(sample_performances, sample_constraints):
    """测试风险平价优化器"""
    optimizer = RiskParityOptimizer()
    weights = optimizer.optimize(sample_performances, sample_constraints)

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0, rtol=1e-5)
    # 验证风险贡献均衡
    returns = pd.DataFrame({name: perf.returns for name, perf in sample_performances.items()})
    cov = returns.cov()
    risk_contrib = {
        name: weight * (cov.values @ np.array(list(weights.values())))[i] / np.sqrt(weights @ cov.values @ weights)
        for i, (name, weight) in enumerate(weights.items())
    }
    assert np.allclose(list(risk_contrib.values()), [1/3]*3, atol=0.05)

def test_portfolio_manager_backtest(sample_performances, sample_constraints):
    """测试组合回测"""
    optimizer = EqualWeightOptimizer()
    manager = PortfolioManager(optimizer, rebalance_freq='M')
    weights_df = manager.run_backtest(
        sample_performances,
        sample_constraints,
        '2023-01-01',
        '2023-04-01'
    )

    assert not weights_df.empty
    assert len(weights_df.columns) == 4  # 3个月 + 初始月
    assert np.allclose(weights_df.sum(axis=0), 1.0, rtol=1e-5)

def test_portfolio_attribution(sample_performances, sample_constraints):
    """测试绩效归因"""
    optimizer = EqualWeightOptimizer()
    manager = PortfolioManager(optimizer)
    weights_df = pd.DataFrame({
        '2023-01-01': {'strategy1': 0.4, 'strategy2': 0.3, 'strategy3': 0.3},
        '2023-02-01': {'strategy1': 0.5, 'strategy2': 0.2, 'strategy3': 0.3}
    })

    attribution_df = manager.calculate_attribution(weights_df, sample_performances)
    assert not attribution_df.empty
    assert len(attribution_df.columns) == len(AttributionFactor)

def test_portfolio_visualization(sample_performances, sample_constraints):
    """测试组合可视化"""
    optimizer = EqualWeightOptimizer()
    manager = PortfolioManager(optimizer)
    weights_df = pd.DataFrame({
        '2023-01-01': {'strategy1': 0.4, 'strategy2': 0.3, 'strategy3': 0.3},
        '2023-02-01': {'strategy1': 0.5, 'strategy2': 0.2, 'strategy3': 0.3}
    })

    # 测试权重图
    fig = PortfolioVisualizer.plot_weights(weights_df)
    assert fig is not None

    # 测试归因图
    attribution_df = manager.calculate_attribution(weights_df, sample_performances)
    fig = PortfolioVisualizer.plot_attribution(attribution_df)
    assert fig is not None

    # 测试绩效图
    fig = PortfolioVisualizer.plot_performance(weights_df, sample_performances)
    assert fig is not None

def test_constraints_handling(sample_performances):
    """测试约束条件处理"""
    constraints = PortfolioConstraints(
        max_weight=0.4,
        min_weight=0.2,
        max_turnover=0.3,
        max_leverage=1.0
    )

    optimizer = MeanVarianceOptimizer()
    weights = optimizer.optimize(sample_performances, constraints)

    assert all(constraints.min_weight <= w <= constraints.max_weight
              for w in weights.values())

def test_empty_performance_input():
    """测试空绩效输入"""
    optimizer = EqualWeightOptimizer()
    with pytest.raises(ValueError):
        optimizer.optimize({}, PortfolioConstraints())
