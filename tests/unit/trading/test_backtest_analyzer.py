import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.trading.backtest_analyzer import (
    BacktestAnalyzer,
    PerformanceMetrics,
    PortfolioAnalyzer,
    TransactionCostModel
)

@pytest.fixture
def sample_returns():
    """生成测试收益数据"""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 252),
                   index=pd.date_range('2023-01-01', periods=252))

@pytest.fixture
def sample_benchmark():
    """生成测试基准数据"""
    np.random.seed(24)
    return pd.Series(np.random.normal(0.0008, 0.015, 252),
                   index=pd.date_range('2023-01-01', periods=252))

@pytest.fixture
def sample_portfolio():
    """生成测试组合数据"""
    np.random.seed(42)
    data = {
        'StockA': np.random.normal(0.001, 0.02, 252),
        'StockB': np.random.normal(0.0008, 0.018, 252),
        'StockC': np.random.normal(0.0012, 0.015, 252)
    }
    return pd.DataFrame(data,
                      index=pd.date_range('2023-01-01', periods=252))

def test_backtest_analyzer(sample_returns, sample_benchmark):
    """测试回测分析器"""
    analyzer = BacktestAnalyzer(sample_returns, sample_benchmark)

    # 测试绩效指标计算
    metrics = analyzer.calculate_performance()
    assert isinstance(metrics, PerformanceMetrics)
    assert not np.isnan(metrics.annualized_return)
    assert metrics.annualized_volatility > 0
    assert not np.isnan(metrics.sharpe_ratio)
    assert metrics.max_drawdown < 0

    # 测试无基准情况
    analyzer_no_bench = BacktestAnalyzer(sample_returns)
    metrics_no_bench = analyzer_no_bench.calculate_performance()
    assert not np.isnan(metrics_no_bench.annualized_return)

def test_performance_metrics_calculation(sample_returns):
    """测试绩效指标计算逻辑"""
    analyzer = BacktestAnalyzer(sample_returns)

    # 测试年化收益率
    annual_return = analyzer._annualized_return()
    cum_return = (1 + sample_returns).prod()
    expected = cum_return ** (1/(len(sample_returns)/252)) - 1
    assert pytest.approx(annual_return, rel=1e-3) == expected

    # 测试年化波动率
    annual_vol = analyzer._annualized_volatility()
    expected = sample_returns.std() * np.sqrt(252)
    assert pytest.approx(annual_vol, rel=1e-3) == expected

    # 测试最大回撤
    max_dd = analyzer._max_drawdown()
    cum_returns = (1 + sample_returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    expected = drawdown.min()
    assert pytest.approx(max_dd, rel=1e-3) == expected

def test_portfolio_analyzer(sample_portfolio):
    """测试组合分析器"""
    analyzer = PortfolioAnalyzer(sample_portfolio)

    # 测试归因分析
    attribution = analyzer.calculate_attribution()
    assert isinstance(attribution, pd.DataFrame)
    assert attribution.shape == (3, 3)

    # 测试相关性矩阵
    corr = analyzer.calculate_attribution()
    assert (corr.values >= -1).all() and (corr.values <= 1).all()

def test_transaction_cost_model():
    """测试交易成本模型"""
    cost_model = TransactionCostModel()

    # 测试成本估算
    cost = cost_model.estimate_cost(1000, 50.0, 1e6)
    assert cost > 0

    # 测试不同参数
    high_cost_model = TransactionCostModel(commission=0.001, slippage=0.001)
    high_cost = high_cost_model.estimate_cost(1000, 50.0, 1e6)
    assert high_cost > cost

def test_visualizations(sample_returns, sample_benchmark):
    """测试可视化组件"""
    analyzer = BacktestAnalyzer(sample_returns, sample_benchmark)

    # 测试收益曲线图
    fig = analyzer.plot_returns()
    assert fig is not None

    # 测试回撤曲线图
    fig = analyzer.plot_drawdown()
    assert fig is not None

    # 测试组合相关性图
    portfolio = PortfolioAnalyzer(pd.DataFrame({
        'A': sample_returns,
        'B': sample_benchmark
    }))
    fig = portfolio.plot_correlation()
    assert fig is not None

def test_edge_cases():
    """测试边界情况"""
    # 测试零收益
    zero_returns = pd.Series(np.zeros(100))
    analyzer = BacktestAnalyzer(zero_returns)
    metrics = analyzer.calculate_performance()
    assert metrics.annualized_return == 0
    assert metrics.max_drawdown == 0

    # 测试单日收益
    single_day = pd.Series([0.01], index=[pd.Timestamp('2023-01-01')])
    analyzer = BacktestAnalyzer(single_day)
    metrics = analyzer.calculate_performance()
    assert not np.isnan(metrics.annualized_return)
