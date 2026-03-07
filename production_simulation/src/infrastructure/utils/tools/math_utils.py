import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from typing import Union, List

"""
数学工具模块 - 提供量化分析中常用的数学计算功能
"""


class MathUtils:
    """数学工具类"""
    
    def __init__(self):
        """初始化数学工具"""
        pass
    
    def mean(self, values: List[float]) -> float:
        """计算平均值"""
        return np.mean(values) if values else 0.0
    
    def median(self, values: List[float]) -> float:
        """计算中位数"""
        return np.median(values) if values else 0.0
    
    def std_dev(self, values: List[float]) -> float:
        """计算标准差"""
        return np.std(values) if values else 0.0
    
    def round(self, value: float, decimals: int = 2) -> float:
        """四舍五入"""
        return np.round(value, decimals)


def normalize(values: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """
    归一化数据到[0,1]范围

    Args:
        values: 输入数据

    Returns:
        归一化后的numpy数组
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)
    denom = max_val - min_val
    if denom == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / denom


def _normalize_data(values):
    """归一化数据到[0,1]范围"

    Args:
        values: 输入数据

    Returns:
        归一化后的numpy数组
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)
    denom = max_val - min_val
    if denom == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / denom


def standardize(values: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """标准化数据(均值0,标准差1)"

    Args:
        values: 输入数据

    Returns:
        标准化后的numpy数组
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """计算滚动Z - Score"

    Args:
        series: 输入序列
        window: 滚动窗口大小

    Returns:
        滚动Z - Score序列
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    safe_std = rolling_std.replace({0.0: np.nan})
    zscore = (series - rolling_mean) / safe_std
    return zscore.fillna(0.0)


def calculate_returns(prices, period: int = 1):
    """计算收益率"

    Args:
        prices: 价格序列 (list, numpy array, or pandas Series)
        period: 收益率计算周期

    Returns:
        收益率序列
    """
    if isinstance(prices, list):
        if len(prices) == 0:
            raise ValueError("价格序列不能为空")
        prices = pd.Series(prices)
    elif hasattr(prices, "pct_change"):
        # pandas Series or DataFrame
        if len(prices) == 0:
            raise ValueError("价格序列不能为空")
    else:
        prices = pd.Series(prices)
        if len(prices) == 0:
            raise ValueError("价格序列不能为空")

    result = prices.pct_change(periods=period)
    return result.fillna(0.0).tolist()  # 返回list而不是Series


def calculate_log_returns(prices: pd.Series, period: int = 1) -> pd.Series:
    """计算对数收益率"

    Args:
        prices: 价格序列
        period: 收益率计算周期

    Returns:
        对数收益率序列
    """
    return np.log(prices / prices.shift(periods=period))


def ewma(values: Union[np.ndarray, pd.Series], span: int = 20) -> np.ndarray:
    """指数加权移动平均"

    Args:
        values: 输入数据
        span: 衰减系数

    Returns:
        指数加权移动平均结果
    """
    return pd.Series(values).ewm(span=span).mean().values


def calculate_correlation(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    method: str = "pearson",
):
    """计算两个序列的相关性

    Args:
        x: 第一个序列
        y: 第二个序列
        method: 相关性计算方法('pearson'/'spearman')

    Returns:
        相关系数
    """
    if method == "pearson":
        return np.corrcoef(x, y)[0, 1]
    elif method == "spearman":
        return spearmanr(x, y)[0]
    else:
        raise ValueError(f"Unsupported correlation method: {method}")


def calculate_volatility(
    returns: Union[np.ndarray, pd.Series], annualize: bool = True, periods: int = 252
):
    """计算波动率

    Args:
        returns: 收益率序列
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        波动率
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    vol = np.std(arr)
    if annualize:
        vol *= np.sqrt(periods)
    return float(vol)


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    periods: int = 252,
):
    """计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        夏普比率
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    excess_returns = arr - risk_free_rate
    std = np.std(excess_returns)
    if std == 0:
        return 0.0
    sharpe = np.mean(excess_returns) / std
    if annualize:
        sharpe *= np.sqrt(periods)
    return float(sharpe)


def calculate_max_drawdown(values: Union[np.ndarray, pd.Series]) -> float:
    """计算最大回撤"

    Args:
        values: 净值序列

    Returns:
        最大回撤(0 - 1之间)
    """
    if not isinstance(values, pd.Series):
        values = pd.Series(values, dtype=float)
    if values.empty:
        return 0.0
    peak = values.expanding().max()
    drawdown = (values - peak) / peak
    return float(drawdown.min())


def calculate_rolling_quantile(
    series: pd.Series, window: int = 20, quantile: float = 0.5
):
    """计算滚动分位数

    Args:
        series: 输入序列
        window: 滚动窗口大小
        quantile: 分位数(0 - 1之间)

    Returns:
        滚动分位数序列
    """
    return series.rolling(window).quantile(quantile)


def calculate_rank(series: pd.Series, method: str = "average", ascending: bool = True):
    """计算排名

    Args:
        series: 输入序列
        method: 排名方法('average'/'min'/'max'/'first'/'dense')
        ascending: 是否升序排列

    Returns:
        排名序列
    """
    return series.rank(method=method, ascending=ascending)


def calculate_decay(series: pd.Series, half_life: float = 10.0):
    """计算指数衰减权重

    Args:
        series: 输入序列
        half_life: 半衰期

    Returns:
        指数衰减权重序列
    """
    return series.ewm(halflife=half_life).mean()


def annualized_volatility(returns: Union[np.ndarray, pd.Series], periods: int = 252):
    """计算年化波动率(calculate_volatility的别名)

    Args:
        returns: 收益率序列
        periods: 年化周期数

    Returns:
        年化波动率
    """
    return calculate_volatility(returns, annualize=True, periods=periods)


def sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods: int = 252,
):
    """计算夏普比率(calculate_sharpe_ratio的别名)

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        periods: 年化周期数

    Returns:
        夏普比率
    """
    return calculate_sharpe_ratio(returns, risk_free_rate, True, periods)
