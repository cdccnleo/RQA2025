import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.portfolio.portfolio_optimizer import (
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    PortfolioResult,
    OptimizationMethod,
    ConstraintType
)

@pytest.fixture
def sample_returns():
    """生成测试收益数据"""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    assets = ['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D']
    data = np.random.normal(0.001, 0.02, size=(252, 4))
    return pd.DataFrame(data, index=dates, columns=assets)

@pytest.fixture
def sample_constraints():
    """生成测试约束条件"""
    return {
        ConstraintType.LEVERAGE: 1.5,
        ConstraintType.CONCENTRATION: 0.3
    }

def test_mean_variance_optimizer(sample_returns, sample_constraints):
    """测试均值方差优化"""
    optimizer = MeanVarianceOptimizer(risk_aversion=1.0)
    result = optimizer.optimize(
        sample_returns,
        OptimizationMethod.MEAN_VARIANCE,
        sample_constraints
    )

    assert isinstance(result, PortfolioResult)
    assert len(result.weights) == 4
    assert pytest.approx(result.weights.sum()) == 1.0
    assert all(result.weights >= 0)
    assert 'expected_return' in result.performance
    assert 'volatility' in result.performance
    assert len(result.risk_contributions) == 4

def test_risk_parity_optimizer(sample_returns, sample_constraints):
    """测试风险平价优化"""
    optimizer = RiskParityOptimizer()
    result = optimizer.optimize(
        sample_returns,
        OptimizationMethod.RISK_PARITY,
        sample_constraints
    )

    assert isinstance(result, PortfolioResult)
    assert len(result.weights) == 4
    assert pytest.approx(result.weights.sum()) == 1.0
    assert all(result.weights >= 0)
    assert 'expected_return' in result.performance
    assert 'volatility' in result.performance
    assert len(result.risk_contributions) == 4

def test_optimizer_constraints(sample_returns):
    """测试优化器约束条件"""
    # 测试杠杆约束
    constraints = {ConstraintType.LEVERAGE: 1.0}
    optimizer = MeanVarianceOptimizer()
    result = optimizer.optimize(
        sample_returns,
        OptimizationMethod.MEAN_VARIANCE,
        constraints
    )
    assert pytest.approx(np.sum(np.abs(result.weights))) <= 1.0 + 1e-6

def test_risk_contributions(sample_returns):
    """测试风险贡献计算"""
    optimizer = MeanVarianceOptimizer()
    result = optimizer.optimize(
        sample_returns,
        OptimizationMethod.MEAN_VARIANCE,
        {}
    )
    risk_contrib = result.risk_contributions
    assert pytest.approx(risk_contrib.sum()) == 1.0
    assert all(risk_contrib >= 0)

def test_portfolio_visualization():
    """测试组合可视化"""
    from src.portfolio.portfolio_optimizer import PortfolioVisualizer

    # 测试权重分布图
    weights = pd.Series([0.4, 0.3, 0.2, 0.1],
                       index=['A', 'B', 'C', 'D'])
    fig = PortfolioVisualizer.plot_weights(weights)
    assert fig is not None

    # 测试风险贡献图
    risk_contrib = pd.Series([0.35, 0.35, 0.2, 0.1],
                           index=['A', 'B', 'C', 'D'])
    fig = PortfolioVisualizer.plot_risk_contributions(risk_contrib)
    assert fig is not None

    # 测试有效前沿
    returns = pd.DataFrame(np.random.normal(0.001, 0.02, size=(100, 4)),
                          columns=['A', 'B', 'C', 'D'])
    fig = PortfolioVisualizer.plot_efficient_frontier(returns)
    assert fig is not None

def test_portfolio_manager(sample_returns, sample_constraints):
    """测试组合管理器"""
    from src.portfolio.portfolio_optimizer import (
        PortfolioManager,
        MeanVarianceOptimizer,
        RiskParityOptimizer
    )

    optimizers = {
        OptimizationMethod.MEAN_VARIANCE: MeanVarianceOptimizer(),
        OptimizationMethod.RISK_PARITY: RiskParityOptimizer()
    }
    manager = PortfolioManager(optimizers)

    # 测试再平衡
    result = manager.rebalance(
        sample_returns,
        OptimizationMethod.MEAN_VARIANCE,
        sample_constraints
    )
    assert isinstance(result, PortfolioResult)

    # 测试绩效分析
    perf = manager.analyze_performance(sample_returns, result.weights)
    assert 'annual_return' in perf
    assert 'sharpe_ratio' in perf
    assert 'max_drawdown' in perf

def test_edge_cases():
    """测试边界情况"""
    from src.portfolio.portfolio_optimizer import MeanVarianceOptimizer

    # 测试单资产情况
    single_asset = pd.DataFrame(np.random.normal(0.001, 0.02, size=(100, 1)),
                               columns=['A'])
    optimizer = MeanVarianceOptimizer()
    result = optimizer.optimize(single_asset, OptimizationMethod.MEAN_VARIANCE, {})
    assert pytest.approx(result.weights.iloc[0]) == 1.0

    # 测试零收益情况
    zero_returns = pd.DataFrame(np.zeros((100, 4)),
                              columns=['A', 'B', 'C', 'D'])
    result = optimizer.optimize(zero_returns, OptimizationMethod.MEAN_VARIANCE, {})
    assert pytest.approx(result.weights.sum()) == 1.0
