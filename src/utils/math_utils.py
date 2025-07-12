"""
数学工具模块 - 提供量化分析中常用的数学计算功能
"""

import numpy as np
import pandas as pd
from typing import Union, List

def normalize(values: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """归一化数据到[0,1]范围

    Args:
        values: 输入数据

    Returns:
        归一化后的numpy数组
    """
    arr = np.asarray(values)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def standardize(values: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """标准化数据(均值0,标准差1)

    Args:
        values: 输入数据

    Returns:
        标准化后的numpy数组
    """
    arr = np.asarray(values)
    return (arr - np.mean(arr)) / np.std(arr)

def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """计算滚动Z-Score

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        滚动Z-Score序列
    """
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def calculate_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    """计算收益率

    Args:
        prices: 价格序列
        period: 收益率计算周期

    Returns:
        收益率序列
    """
    return prices.pct_change(periods=period)

def calculate_log_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    """计算对数收益率

    Args:
        prices: 价格序列
        period: 收益率计算周期

    Returns:
        对数收益率序列
    """
    return np.log(prices / prices.shift(periods=period))

def ewma(values: Union[np.ndarray, pd.Series], span: int = 20) -> np.ndarray:
    """指数加权移动平均

    Args:
        values: 输入数据
        span: 衰减系数

    Returns:
        指数加权移动平均结果
    """
    return pd.Series(values).ewm(span=span).mean().values

def calculate_correlation(x: Union[np.ndarray, pd.Series],
                         y: Union[np.ndarray, pd.Series],
                         method: str = 'pearson') -> float:
    """计算两个序列的相关性

    Args:
        x: 第一个序列
        y: 第二个序列
        method: 相关性计算方法('pearson'/'spearman')

    Returns:
        相关系数
    """
    if method == 'pearson':
        return np.corrcoef(x, y)[0, 1]
    elif method == 'spearman':
        from scipy.stats import spearmanr
        return spearmanr(x, y)[0]
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

def calculate_volatility(returns: Union[np.ndarray, pd.Series],
                         annualize: bool = True,
                         periods: int = 252) -> float:
    """计算波动率

    Args:
        returns: 收益率序列
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        波动率
    """
    vol = np.std(returns)
    if annualize:
        vol *= np.sqrt(periods)
    return vol

def calculate_sharpe_ratio(returns: Union[np.ndarray, pd.Series],
                           risk_free_rate: float = 0.0,
                           annualize: bool = True,
                           periods: int = 252) -> float:
    """计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        夏普比率
    """
    excess_returns = returns - risk_free_rate
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    if annualize:
        sharpe *= np.sqrt(periods)
    return sharpe

def calculate_max_drawdown(values: Union[np.ndarray, pd.Series]) -> float:
    """计算最大回撤

    Args:
        values: 净值序列

    Returns:
        最大回撤(0-1之间)
    """
    peak = values.expanding().max()
    drawdown = (values - peak) / peak
    return drawdown.min()

def calculate_rolling_quantile(series: pd.Series,
                               window: int = 20,
                               quantile: float = 0.5) -> pd.Series:
    """计算滚动分位数

    Args:
        series: 输入序列
        window: 滚动窗口大小
        quantile: 分位数(0-1之间)

    Returns:
        滚动分位数序列
    """
    return series.rolling(window).quantile(quantile)

def calculate_rank(series: pd.Series,
                   method: str = 'average',
                   ascending: bool = True) -> pd.Series:
    """计算排名

    Args:
        series: 输入序列
        method: 排名方法('average'/'min'/'max'/'first'/'dense')
        ascending: 是否升序排列

    Returns:
        排名序列
    """
    return series.rank(method=method, ascending=ascending)

def calculate_decay(series: pd.Series,
                   half_life: float = 10.0) -> pd.Series:
    """计算指数衰减权重

    Args:
        series: 输入序列
        half_life: 半衰期

    Returns:
        指数衰减权重序列
    """
    return series.ewm(halflife=half_life).mean()

def annualized_volatility(returns: Union[np.ndarray, pd.Series],
                         periods: int = 252) -> float:
    """计算年化波动率(calculate_volatility的别名)
    
    Args:
        returns: 收益率序列
        periods: 年化周期数
        
    Returns:
        年化波动率
    """
    return calculate_volatility(returns, annualize=True, periods=periods)

def sharpe_ratio(returns: Union[np.ndarray, pd.Series],
                 risk_free_rate: float = 0.0,
                 periods: int = 252) -> float:
    """计算夏普比率(calculate_sharpe_ratio的别名)
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        periods: 年化周期数
        
    Returns:
        夏普比率
    """
    return calculate_sharpe_ratio(returns, risk_free_rate, True, periods)
