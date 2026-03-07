#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析器单元测试

测试目标：提升performance_analyzer.py的覆盖率到90%+
按照业务流程驱动架构设计测试性能分析功能
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.trading.performance.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzerInitialization:
    """测试性能分析器初始化"""

    def test_init_with_returns(self):
        """测试使用收益率序列初始化"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        assert len(analyzer.returns) == 5
        assert analyzer.benchmark is None
        assert analyzer._results is None

    def test_init_with_benchmark(self):
        """测试使用基准收益率初始化"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.01])
        analyzer = PerformanceAnalyzer(returns, benchmark)

        assert len(analyzer.returns) == 5
        assert len(analyzer.benchmark) == 5

    def test_init_empty_returns_raises_error(self):
        """测试空收益率序列抛出错误"""
        with pytest.raises(ValueError):
            PerformanceAnalyzer(pd.Series([]))

    def test_init_none_returns_raises_error(self):
        """测试None收益率序列抛出错误"""
        with pytest.raises(ValueError):
            PerformanceAnalyzer(None)


class TestPerformanceAnalyzerBasicMetrics:
    """测试性能分析器基础指标"""

    def test_calculate_total_return(self):
        """测试计算累计收益率"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        total_return = analyzer._calculate_total_return()
        expected = (1 + 0.01) * (1 - 0.02) * (1 + 0.03) * (1 - 0.01) * (1 + 0.02) - 1

        assert abs(total_return - expected) < 1e-6

    def test_calculate_annual_return(self):
        """测试计算年化收益率"""
        returns = pd.Series([0.01] * 252)  # 252个交易日
        analyzer = PerformanceAnalyzer(returns)

        annual_return = analyzer._calculate_annual_return()
        expected = (1 + 0.01) ** 252 - 1

        assert abs(annual_return - expected) < 1e-3

    def test_calculate_annual_volatility(self):
        """测试计算年化波动率"""
        returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01])
        analyzer = PerformanceAnalyzer(returns)

        annual_vol = analyzer._calculate_annual_volatility()
        daily_vol = returns.std()
        expected = daily_vol * np.sqrt(252)

        assert abs(annual_vol - expected) < 1e-6

    def test_calculate_sharpe_ratio(self):
        """测试计算夏普比率"""
        returns = pd.Series([0.01] * 100)
        analyzer = PerformanceAnalyzer(returns)

        sharpe = analyzer._calculate_sharpe_ratio()
        assert sharpe > 0

    def test_calculate_max_drawdown(self):
        """测试计算最大回撤"""
        returns = pd.Series([0.1, -0.2, 0.05, -0.15, 0.1])
        analyzer = PerformanceAnalyzer(returns)

        max_dd = analyzer._calculate_max_drawdown()
        assert max_dd <= 0

    def test_calculate_calmar_ratio(self):
        """测试计算Calmar比率"""
        returns = pd.Series([0.01] * 252)
        analyzer = PerformanceAnalyzer(returns)

        calmar = analyzer._calculate_calmar_ratio()
        # Calmar比率可能是NaN（当max_drawdown为0时），或者>=0
        assert np.isnan(calmar) or calmar >= 0

    def test_calculate_sortino_ratio(self):
        """测试计算Sortino比率"""
        returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01])
        analyzer = PerformanceAnalyzer(returns)

        sortino = analyzer._calculate_sortino_ratio()
        # Sortino比率可能是NaN（当downside为0时），或者>=0
        assert np.isnan(sortino) or sortino >= 0


class TestPerformanceAnalyzerBenchmarkMetrics:
    """测试性能分析器基准比较指标"""

    def test_calculate_alpha_beta(self):
        """测试计算Alpha和Beta"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.01])
        analyzer = PerformanceAnalyzer(returns, benchmark)

        alpha, beta = analyzer._calculate_alpha_beta()

        assert not np.isnan(alpha)
        assert not np.isnan(beta)

    def test_calculate_alpha_beta_no_benchmark(self):
        """测试无基准时Alpha和Beta返回NaN"""
        returns = pd.Series([0.01, -0.02, 0.03])
        analyzer = PerformanceAnalyzer(returns)

        alpha, beta = analyzer._calculate_alpha_beta()

        assert np.isnan(alpha)
        assert np.isnan(beta)

    def test_calculate_information_ratio(self):
        """测试计算信息比率"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.01])
        analyzer = PerformanceAnalyzer(returns, benchmark)

        ir = analyzer._calculate_information_ratio()

        assert not np.isnan(ir)

    def test_calculate_tracking_error(self):
        """测试计算跟踪误差"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.01])
        analyzer = PerformanceAnalyzer(returns, benchmark)

        te = analyzer._calculate_tracking_error()

        assert not np.isnan(te)
        assert te >= 0

    def test_calculate_outperformance(self):
        """测试计算超额收益率"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.01])
        analyzer = PerformanceAnalyzer(returns, benchmark)

        outperformance = analyzer._calculate_outperformance()

        assert not np.isnan(outperformance)


class TestPerformanceAnalyzerAdvancedMetrics:
    """测试性能分析器高级指标"""

    def test_calculate_skewness(self):
        """测试计算收益偏度"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        skewness = analyzer._calculate_skewness()

        assert not np.isnan(skewness)

    def test_calculate_kurtosis(self):
        """测试计算收益峰度"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        kurtosis = analyzer._calculate_kurtosis()

        assert not np.isnan(kurtosis)

    def test_calculate_var(self):
        """测试计算VaR"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        var_95 = analyzer._calculate_var(0.95)

        assert not np.isnan(var_95)

    def test_calculate_cvar(self):
        """测试计算CVaR"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        cvar_95 = analyzer._calculate_cvar(0.95)

        assert not np.isnan(cvar_95)

    def test_calculate_win_rate(self):
        """测试计算胜率"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        win_rate = analyzer._calculate_win_rate()

        assert 0 <= win_rate <= 1

    def test_calculate_profit_factor(self):
        """测试计算盈利因子"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        analyzer = PerformanceAnalyzer(returns)

        pf = analyzer._calculate_profit_factor()

        assert pf >= 0


class TestPerformanceAnalyzerFullAnalysis:
    """测试性能分析器完整分析"""

    def test_analyze_without_benchmark(self):
        """测试无基准的完整分析"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 50)  # 250个交易日
        analyzer = PerformanceAnalyzer(returns)

        results = analyzer.analyze()

        assert 'total_return' in results
        assert 'annual_return' in results
        assert 'annual_volatility' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'calmar_ratio' in results
        assert 'sortino_ratio' in results
        assert 'skewness' in results
        assert 'kurtosis' in results
        assert 'var_95' in results
        assert 'cvar_95' in results
        assert 'win_rate' in results
        assert 'profit_factor' in results
        assert np.isnan(results['beta'])
        assert np.isnan(results['alpha'])

    def test_analyze_with_benchmark(self):
        """测试有基准的完整分析"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02] * 50)
        benchmark = pd.Series([0.005, -0.01, 0.02, -0.005, 0.01] * 50)
        analyzer = PerformanceAnalyzer(returns, benchmark)

        results = analyzer.analyze()

        assert 'beta' in results
        assert 'alpha' in results
        assert 'information_ratio' in results
        assert 'tracking_error' in results
        assert 'outperformance' in results
        assert not np.isnan(results['beta'])
        assert not np.isnan(results['alpha'])

    def test_analyze_stores_results(self):
        """测试分析结果存储"""
        returns = pd.Series([0.01, -0.02, 0.03])
        analyzer = PerformanceAnalyzer(returns)

        assert analyzer._results is None

        results = analyzer.analyze()

        assert analyzer._results is not None
        assert analyzer._results == results


class TestPerformanceAnalyzerEdgeCases:
    """测试性能分析器边界情况"""

    def test_all_positive_returns(self):
        """测试全部正收益"""
        returns = pd.Series([0.01] * 100)
        analyzer = PerformanceAnalyzer(returns)

        results = analyzer.analyze()

        assert results['total_return'] > 0
        assert results['win_rate'] == 1.0

    def test_all_negative_returns(self):
        """测试全部负收益"""
        returns = pd.Series([-0.01] * 100)
        analyzer = PerformanceAnalyzer(returns)

        results = analyzer.analyze()

        assert results['total_return'] < 0
        assert results['win_rate'] == 0.0

    def test_zero_returns(self):
        """测试零收益"""
        returns = pd.Series([0.0] * 100)
        analyzer = PerformanceAnalyzer(returns)

        results = analyzer.analyze()

        assert abs(results['total_return']) < 1e-6
        assert results['annual_volatility'] == 0.0

    def test_single_return(self):
        """测试单个收益率"""
        returns = pd.Series([0.01])
        analyzer = PerformanceAnalyzer(returns)

        results = analyzer.analyze()

        assert 'total_return' in results
        # 使用近似比较处理浮点数精度问题
        assert abs(results['total_return'] - 0.01) < 1e-10

    def test_benchmark_different_length(self):
        """测试基准长度不同"""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        benchmark = pd.Series([0.005, -0.01, 0.02])  # 不同长度
        analyzer = PerformanceAnalyzer(returns, benchmark)

        # 应该使用交集索引
        alpha, beta = analyzer._calculate_alpha_beta()

        assert not np.isnan(alpha) or not np.isnan(beta)

