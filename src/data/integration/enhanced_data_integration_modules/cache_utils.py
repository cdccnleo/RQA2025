"""
缓存工具模块

提供缓存相关的工具函数，用于检查、存储和管理缓存数据。
"""

from typing import Dict, List
import pandas as pd


def check_cache_for_symbols(
    cache_strategy, symbols: List[str], start_date: str, end_date: str, frequency: str
) -> Dict[str, pd.DataFrame]:
    """
    检查股票数据缓存
    
    Args:
        cache_strategy: 缓存策略实例
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        frequency: 数据频率
    
    Returns:
        缓存的股票数据字典
    """
    cached_data = {}
    for symbol in symbols:
        cache_key = f"stock_{symbol}_{start_date}_{end_date}_{frequency}"
        data = cache_strategy.get(cache_key)
        if data is not None:
            cached_data[symbol] = data
    return cached_data


def check_cache_for_indices(
    cache_strategy, indices: List[str], start_date: str, end_date: str, frequency: str
) -> Dict[str, pd.DataFrame]:
    """
    检查指数数据缓存
    
    Args:
        cache_strategy: 缓存策略实例
        indices: 指数代码列表
        start_date: 开始日期
        end_date: 结束日期
        frequency: 数据频率
    
    Returns:
        缓存的指数数据字典
    """
    cached_data = {}
    for index in indices:
        cache_key = f"index_{index}_{start_date}_{end_date}_{frequency}"
        data = cache_strategy.get(cache_key)
        if data is not None:
            cached_data[index] = data
    return cached_data


def check_cache_for_financial(
    symbols: List[str], start_date: str, end_date: str, data_type: str, cache_strategy
) -> Dict[str, pd.DataFrame]:
    """
    检查财务数据缓存
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        data_type: 数据类型
        cache_strategy: 缓存策略实例
    
    Returns:
        缓存的财务数据字典
    """
    cached_data = {}
    for symbol in symbols:
        cache_key = f"financial_{symbol}_{start_date}_{end_date}_{data_type}"
        data = cache_strategy.get(cache_key)
        if data is not None:
            cached_data[symbol] = data
    return cached_data


def cache_data(
    cache_strategy,
    symbol: str,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    frequency: str,
):
    """
    缓存股票数据
    
    Args:
        cache_strategy: 缓存策略实例
        symbol: 股票代码
        data: 数据DataFrame
        start_date: 开始日期
        end_date: 结束日期
        frequency: 数据频率
    """
    cache_key = f"stock_{symbol}_{start_date}_{end_date}_{frequency}"
    cache_strategy.set(cache_key, data, ttl=3600)


def cache_index_data(
    cache_strategy, index: str, data: pd.DataFrame, start_date: str, end_date: str, frequency: str
):
    """
    缓存指数数据
    
    Args:
        cache_strategy: 缓存策略实例
        index: 指数代码
        data: 数据DataFrame
        start_date: 开始日期
        end_date: 结束日期
        frequency: 数据频率
    """
    cache_key = f"index_{index}_{start_date}_{end_date}_{frequency}"
    cache_strategy.set(cache_key, data, ttl=3600)


def cache_financial_data(
    cache_strategy,
    symbol: str,
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    data_type: str,
):
    """
    缓存财务数据
    
    Args:
        cache_strategy: 缓存策略实例
        symbol: 股票代码
        data: 数据DataFrame
        start_date: 开始日期
        end_date: 结束日期
        data_type: 数据类型
    """
    cache_key = f"financial_{symbol}_{start_date}_{end_date}_{data_type}"
    cache_strategy.set(cache_key, data, ttl=3600)

