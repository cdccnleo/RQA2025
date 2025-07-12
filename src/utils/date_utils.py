"""
日期处理工具 - 提供与A股交易日历相关的功能
"""

from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List

# A股交易日历缓存
_trading_days_cache = None

def _load_trading_calendar():
    """加载交易日历"""
    global _trading_days_cache
    # TODO: 从数据库或文件加载实际交易日历
    _trading_days_cache = pd.bdate_range(
        start='2020-01-01',
        end='2025-12-31',
        freq='B'
    ).to_pydatetime().tolist()

def get_business_date(date: datetime = None) -> datetime:
    """获取最近的交易日

    Args:
        date: 参考日期(默认当前日期)

    Returns:
        最近的交易日
    """
    if date is None:
        date = datetime.now()

    if _trading_days_cache is None:
        _load_trading_calendar()

    while date not in _trading_days_cache:
        date -= timedelta(days=1)
    return date

def is_trading_day(date: datetime) -> bool:
    """判断是否为交易日

    Args:
        date: 要检查的日期

    Returns:
        bool: 是否为交易日
    """
    if _trading_days_cache is None:
        _load_trading_calendar()
    return date in _trading_days_cache

def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """转换时区

    Args:
        dt: 原始时间
        from_tz: 原始时区
        to_tz: 目标时区

    Returns:
        转换后的时间
    """
    # 简化实现，实际项目中应使用pytz或zoneinfo
    return dt.astimezone(to_tz)

def next_trading_day(date: datetime) -> datetime:
    """获取下一个交易日"""
    current = date + timedelta(days=1)
    while not is_trading_day(current):
        current += timedelta(days=1)
    return current

def prev_trading_day(date: datetime) -> datetime:
    """获取上一个交易日"""
    current = date - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current

def get_trading_days(start: datetime, end: datetime) -> List[datetime]:
    """获取指定日期范围内的交易日列表"""
    if _trading_days_cache is None:
        _load_trading_calendar()
    return [day for day in _trading_days_cache if start <= day <= end]

def is_trading_time(dt: datetime) -> bool:
    """判断是否为交易时间 (A股: 9:30-11:30, 13:00-15:00)"""
    if not is_trading_day(dt):
        return False
    time = dt.time()
    return (
        (time >= datetime.strptime("09:30:00", "%H:%M:%S").time()
         and time <= datetime.strptime("11:30:00", "%H:%M:%S").time()) or
        (time >= datetime.strptime("13:00:00", "%H:%M:%S").time()
         and time <= datetime.strptime("15:00:00", "%H:%M:%S").time())
    )
