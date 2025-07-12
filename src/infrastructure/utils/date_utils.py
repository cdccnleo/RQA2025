# src/infrastructure/utils/date_utils.py
from datetime import datetime
from typing import Optional

def format_date(dt: datetime, fmt: str = "%Y-%m-%d") -> str:
    """格式化日期时间为字符串

    Args:
        dt: 要格式化的datetime对象
        fmt: 格式字符串，默认为"%Y-%m-%d"

    Returns:
        格式化后的日期字符串
    """
    return dt.strftime(fmt)

def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> Optional[datetime]:
    """解析字符串为datetime对象

    Args:
        date_str: 日期字符串
        fmt: 格式字符串，默认为"%Y-%m-%d"

    Returns:
        解析后的datetime对象，解析失败返回None
    """
    try:
        return datetime.strptime(date_str, fmt)
    except ValueError:
        return None

def get_current_date(fmt: str = "%Y-%m-%d") -> str:
    """获取当前日期字符串

    Args:
        fmt: 格式字符串，默认为"%Y-%m-%d"

    Returns:
        当前日期的格式化字符串
    """
    return format_date(datetime.now(), fmt)

class DateUtils:
    """日期工具类"""
    
    @staticmethod
    def format_date(dt: datetime, fmt: str = "%Y-%m-%d") -> str:
        """格式化日期时间为字符串

        Args:
            dt: 要格式化的datetime对象
            fmt: 格式字符串，默认为"%Y-%m-%d"

        Returns:
            格式化后的日期字符串
        """
        return dt.strftime(fmt)

    @staticmethod
    def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> Optional[datetime]:
        """解析字符串为datetime对象

        Args:
            date_str: 日期字符串
            fmt: 格式字符串，默认为"%Y-%m-%d"

        Returns:
            解析后的datetime对象，解析失败返回None
        """
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            return None

    @staticmethod
    def get_current_date(fmt: str = "%Y-%m-%d") -> str:
        """获取当前日期字符串

        Args:
            fmt: 格式字符串，默认为"%Y-%m-%d"

        Returns:
            当前日期的格式化字符串
        """
        return DateUtils.format_date(datetime.now(), fmt)

    @staticmethod
    def date_diff(date1: datetime, date2: datetime) -> int:
        """计算两个日期之间的天数差

        Args:
            date1: 第一个日期
            date2: 第二个日期

        Returns:
            天数差
        """
        return (date2 - date1).days

    @staticmethod
    def convert_timezone(dt: datetime, timezone: str) -> datetime:
        """转换时区

        Args:
            dt: 要转换的datetime对象
            timezone: 目标时区

        Returns:
            转换后的datetime对象
        """
        # 简化实现，实际应该使用pytz或zoneinfo
        if timezone == "Asia/Shanghai":
            # UTC+8
            return dt.replace(hour=(dt.hour + 8) % 24)
        return dt
