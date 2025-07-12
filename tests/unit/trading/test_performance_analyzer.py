import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.trading.performance_analyzer import PerformanceAnalyzer

@pytest.fixture
def sample_returns():
    """创建测试收益率数据"""
    dates = pd.date_range(start='2023-01-01', periods=100)
    returns = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates)
    return returns

@pytest.fixture
def sample_benchmark():
    """创建测试基准数据"""
    dates = pd.date_range(start='2023-01-01', periods=100)
    returns = pd.Series(np.random.normal(0.0005, 0.008, 100), index=dates)
    return returns

@pytest.fixture
def analyzer(sample_returns, sample_benchmark):
    """创建分析器实例"""
    return PerformanceAnalyzer(sample_returns, sample_benchmark)

def test_total_return(analyzer, sample_returns):
    """测试累计收益率计算"""
    expected = (1 + sample_returns).prod() - 1
    result = analyzer._calculate_total_return()
    assert np.isclose(result, expected)

def test_annual_return(analyzer, sample_returns):
    """测试年化收益率计算"""
    total_return = (1 + sample_returns).prod() - 1
    days = len(sample_returns)
    expected = (1 + total_return) ** (252 / days) - 1
    result = analyzer._calculate_annual_return()
    assert np.isclose(result, expected)

def test_annual_volatility(analyzer, sample_returns):
    """测试年化波动率计算"""
    expected = sample_returns.std() * np.sqrt(252)
    result = analyzer._calculate_annual_volatility()
    assert np.isclose(result, expected)

def test_sharpe_ratio(analyzer, sample_returns):
    """测试夏普比率计算"""
    excess_returns = sample_returns - 0.0 / 252
    expected = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    result = analyzer._calculate_sharpe_ratio()
    assert np.isclose(result, expected)

def test_max_drawdown(analyzer, sample_returns):
    """测试最大回撤计算"""
    cum_returns = (1 + sample_returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    expected = drawdown.min()
    result = analyzer._calculate_max_drawdown()
    assert np.isclose(result, expected)

def test_alpha_beta(analyzer, sample_returns, sample_benchmark):
    """测试Alpha和Beta计算"""
    common_index = sample_returns.index.intersection(sample_benchmark.index)
    strategy_returns = sample_returns[common_index]
    benchmark_returns = sample_benchmark[common_index]

    from scipy import stats
    beta, alpha, _, _, _ = stats.linregress(
        benchmark_returns,
        strategy_returns
    )
    expected_alpha = (1 + alpha) ** 252 - 1

    result_alpha, result_beta = analyzer._calculate_alpha_beta()
    assert np.isclose(result_alpha, expected_alpha)
    assert np.isclose(result_beta, beta)

def test_information_ratio(analyzer, sample_returns, sample_benchmark):
    """测试信息比率计算"""
    active_returns = sample_returns - sample_benchmark
    expected = active_returns.mean() / active_returns.std() * np.sqrt(252)
    result = analyzer._calculate_information_ratio()
    assert np.isclose(result, expected)

def test_var_cvar(analyzer, sample_returns):
    """测试VaR和CVaR计算"""
    confidence_level = 0.95
    expected_var = np.percentile(sample_returns, (1 - confidence_level) * 100)
    expected_cvar = sample_returns[sample_returns <= expected_var].mean()

    result_var = analyzer._calculate_var(confidence_level)
    result_cvar = analyzer._calculate_cvar(confidence_level)

    assert np.isclose(result_var, expected_var)
    assert np.isclose(result_cvar, expected_cvar)

def test_win_rate(analyzer, sample_returns):
    """测试胜率计算"""
    expected = (sample_returns > 0).mean()
    result = analyzer._calculate_win_rate()
    assert np.isclose(result, expected)

def test_profit_factor(analyzer, sample_returns):
    """测试盈利因子计算"""
    gross_profit = sample_returns[sample_returns > 0].sum()
    gross_loss = abs(sample_returns[sample_returns < 0].sum())
    expected = gross_profit / gross_loss if gross_loss != 0 else np.inf
    result = analyzer._calculate_profit_factor()
    if np.isinf(expected):
        assert np.isinf(result)
    else:
        assert np.isclose(result, expected)

def test_analyze(analyzer):
    """测试全面分析"""
    results = analyzer.analyze()
    assert isinstance(results, dict)
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
    assert 'alpha' in results
    assert len(results) >= 15  # 确保所有指标都计算了

def test_plot_performance(analyzer):
    """测试绩效图表生成"""
    with patch('matplotlib.pyplot.subplots') as mock_subplots:
        fig = analyzer.plot_performance()
        mock_subplots.assert_called_once()
        assert isinstance(fig, plt.Figure)

def test_generate_report(analyzer):
    """测试报告生成"""
    report = analyzer.generate_report()
    assert isinstance(report, dict)
    assert 'return_metrics' in report
    assert 'risk_metrics' in report
    assert 'ratio_metrics' in report
    assert 'benchmark_metrics' in report

def test_no_benchmark_case(sample_returns):
    """测试无基准情况"""
    analyzer = PerformanceAnalyzer(sample_returns)
    results = analyzer.analyze()
    assert np.isnan(results['alpha'])
    assert np.isnan(results['beta'])

    report = analyzer.generate_report()
    assert 'benchmark_metrics' not in report

def test_error_handling():
    """测试异常处理"""
    with pytest.raises(ValueError, match="returns cannot be empty"):
        PerformanceAnalyzer(pd.Series(dtype=float))

    # 测试空收益率序列
    empty_returns = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        PerformanceAnalyzer(empty_returns)
