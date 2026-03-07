#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
math_utils 扩展测试，覆盖核心计算路径及异常场景。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.utils.tools import math_utils
from src.infrastructure.utils.tools.math_utils import (
    MathUtils,
    annualized_volatility,
    calculate_correlation,
    calculate_decay,
    calculate_log_returns,
    calculate_max_drawdown,
    calculate_returns,
    calculate_rolling_quantile,
    calculate_sharpe_ratio,
    calculate_volatility,
    ewma,
    normalize,
    rolling_zscore,
    sharpe_ratio,
    standardize,
)


# Fixtures for test data reuse
@pytest.fixture
def sample_series():
    """Sample time series for testing"""
    return pd.Series([10, 30, 20, 40, 10, 25, 35, 15])


@pytest.fixture
def sample_returns():
    """Sample returns series for testing"""
    return np.array([0.01, -0.005, 0.008, 0.002, -0.003])


@pytest.fixture
def sample_prices():
    """Sample price series for testing"""
    return pd.Series([100, 102, 98, 105, 103, 108, 106, 110, 107, 112])


def test_math_utils_basic_statistics():
    utils = MathUtils()
    values = [1.0, 2.0, 3.0, 4.0]
    assert utils.mean(values) == pytest.approx(2.5)
    assert utils.median(values) == pytest.approx(2.5)
    assert utils.std_dev(values) == pytest.approx(np.std(values))
    assert utils.round(3.14159, 3) == pytest.approx(3.142)


def test_normalize_and_standardize_constant_series():
    data = [5.0, 5.0, 5.0]
    assert np.array_equal(normalize(data), np.zeros(3))
    assert np.array_equal(math_utils._normalize_data(data), np.zeros(3))  # type: ignore[attr-defined]
    assert np.array_equal(standardize(data), np.zeros(3))


def test_rolling_zscore_handles_zero_std():
    series = pd.Series([1, 1, 1, 1, 2, 2, 2])
    result = rolling_zscore(series, window=3)
    assert len(result) == len(series)
    # 前若干窗口标准差为0，应该被填充为0
    assert result.iloc[:3].eq(0).all()


def test_calculate_returns_and_log_returns():
    prices = [100, 105, 102]
    pct = calculate_returns(prices, period=1)
    assert pct[0] == pytest.approx(0.0)
    assert pct[1] == pytest.approx(0.05)
    log_returns = calculate_log_returns(pd.Series(prices))
    assert log_returns.iloc[1] == pytest.approx(np.log(105 / 100))

    with pytest.raises(ValueError):
        calculate_returns([])


def test_ewma_and_quantile_and_decay():
    series = pd.Series([1, 2, 3, 4, 5])
    ewma_values = ewma(series, span=2)
    assert len(ewma_values) == len(series)

    quantiles = calculate_rolling_quantile(series, window=2, quantile=0.5)
    assert quantiles.iloc[1] == pytest.approx(1.5)

    decay = calculate_decay(series, half_life=2)
    assert decay.iloc[-1] > decay.iloc[0]


def test_correlation_and_volatility():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    assert calculate_correlation(x, y, method="pearson") == pytest.approx(1.0)
    assert calculate_correlation(x, y, method="spearman") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        calculate_correlation(x, y, method="kendall")

    returns = np.array([0.01, 0.02, -0.01])
    assert calculate_volatility(returns, annualize=False) == pytest.approx(np.std(returns))
    assert annualized_volatility(returns) == pytest.approx(np.std(returns) * np.sqrt(252))
    assert calculate_volatility([], annualize=False) == 0.0


def test_sharpe_ratio_and_aliases():
    returns = np.array([0.01, 0.01, 0.01])
    # 零波动率情况下应返回0，避免除0
    assert calculate_sharpe_ratio(returns, risk_free_rate=0.01) == 0.0
    assert sharpe_ratio(returns, risk_free_rate=0.0) == 0.0


def test_max_drawdown_and_rank():
    series = pd.Series([100, 120, 110, 130, 90])
    drawdown = calculate_max_drawdown(series)
    assert drawdown == pytest.approx((90 - 130) / 130)

    ranked = math_utils.calculate_rank(series, method="dense", ascending=False)
    assert ranked.iloc[3] == 1  # 最大值排名1
    assert ranked.iloc[0] > ranked.iloc[3]


def test_calculate_decay_comprehensive(sample_series):
    """测试指数衰减计算的全面功能"""
    # 基本功能测试
    series = sample_series

    # 默认半衰期
    decay_default = calculate_decay(series)
    assert len(decay_default) == len(series)
    assert decay_default.iloc[-1] > decay_default.iloc[0]  # 应该递增

    # 自定义半衰期
    decay_short = calculate_decay(series, half_life=2)
    decay_long = calculate_decay(series, half_life=20)

    # 验证衰减权重序列是递增的（指数衰减的累积效果）
    assert decay_default.iloc[-1] > decay_default.iloc[0]

    # 测试边界条件
    single_value = pd.Series([5.0])
    decay_single = calculate_decay(single_value)
    assert decay_single.iloc[0] == 5.0

    # 测试空序列
    empty_series = pd.Series([], dtype=float)
    decay_empty = calculate_decay(empty_series)
    assert len(decay_empty) == 0

    # 测试不同半衰期对权重的影响
    series_osc = pd.Series([10, 1, 10, 1, 10])
    decay_fast = calculate_decay(series_osc, half_life=1)
    decay_slow = calculate_decay(series_osc, half_life=10)

    # 快速衰减应该对近期变化更敏感
    assert abs(decay_fast.iloc[-1] - 10) < abs(decay_slow.iloc[-1] - 10)


def test_calculate_rank_comprehensive(sample_series):
    """测试排名计算的全面功能"""
    series = sample_series

    # 测试不同排名方法
    rank_dense = math_utils.calculate_rank(series, method="dense", ascending=False)
    rank_average = math_utils.calculate_rank(series, method="average", ascending=False)
    rank_min = math_utils.calculate_rank(series, method="min", ascending=False)

    # 验证最大值排名
    assert rank_dense.iloc[3] == 1  # 40应该是第1名
    assert rank_average.iloc[3] == 1
    assert rank_min.iloc[3] == 1

    # 验证相同值处理
    assert rank_dense.iloc[0] == rank_dense.iloc[4]  # 两个10应该是相同排名

    # 测试升序排名
    rank_asc = math_utils.calculate_rank(series, method="dense", ascending=True)
    # 验证排名结果基本属性
    assert len(rank_asc) == len(series)  # 排名长度与原序列相同
    assert rank_asc.min() >= 1  # 最小排名至少为1
    assert isinstance(rank_asc, pd.Series)  # 返回类型正确

    # 测试边界条件
    single_series = pd.Series([5.0])
    rank_single = math_utils.calculate_rank(single_series)
    assert rank_single.iloc[0] == 1

    # 测试空序列
    empty_series = pd.Series([], dtype=float)
    rank_empty = math_utils.calculate_rank(empty_series)
    assert len(rank_empty) == 0


def test_annualized_volatility_comprehensive(sample_returns):
    """测试年化波动率的全面功能"""
    # 基本功能测试
    returns = sample_returns

    # 默认参数（252个交易日）
    vol_252 = annualized_volatility(returns)
    expected_vol_252 = np.std(returns) * np.sqrt(252)
    assert vol_252 == pytest.approx(expected_vol_252)

    # 自定义周期数
    vol_365 = annualized_volatility(returns, periods=365)
    expected_vol_365 = np.std(returns) * np.sqrt(365)
    assert vol_365 == pytest.approx(expected_vol_365)

    # 验证周期数影响
    assert vol_365 > vol_252  # 更多周期应该导致更高波动率

    # 测试与基础函数的一致性
    vol_basic = calculate_volatility(returns, annualize=True, periods=252)
    assert vol_252 == pytest.approx(vol_basic)

    # 测试边界条件
    zero_returns = np.array([0.0, 0.0, 0.0])
    vol_zero = annualized_volatility(zero_returns)
    assert vol_zero == 0.0

    constant_returns = np.array([0.01, 0.01, 0.01])
    vol_constant = annualized_volatility(constant_returns)
    assert vol_constant == 0.0  # 常数收益率的波动率为0

    # 测试空数组
    vol_empty = annualized_volatility(np.array([]))
    assert vol_empty == 0.0


def test_quantitative_algorithms_integration(sample_prices):
    """测试量化算法的集成使用"""
    # 创建模拟的股票数据
    prices = sample_prices

    # 计算收益率
    returns = calculate_returns(prices)

    # 计算波动率
    vol = calculate_volatility(returns)
    vol_annual = annualized_volatility(returns)

    # 计算夏普比率
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

    # 计算最大回撤
    max_dd = calculate_max_drawdown(prices)

    # 验证计算结果的合理性
    assert vol > 0
    assert vol_annual >= vol  # 年化波动率应该大于等于非年化波动率（当periods>=1时）
    assert isinstance(sharpe, float)
    assert max_dd <= 0  # 最大回撤是负数（下跌幅度）

    # 测试滚动计算
    rolling_vol = prices.rolling(window=5).std()
    assert len(rolling_vol.dropna()) == len(prices) - 4

    # 测试指数衰减
    returns_series = pd.Series(returns)  # 转换为Series
    decay_weights = calculate_decay(returns_series, half_life=5)
    assert len(decay_weights) == len(returns)


def test_edge_cases_for_small_inputs():
    # 空输入应该返回空数组/零
    assert normalize([]).size == 0
    assert standardize([]).size == 0
    assert calculate_max_drawdown([]) == 0.0
    assert np.array_equal(ewma([], span=3), np.array([]))

