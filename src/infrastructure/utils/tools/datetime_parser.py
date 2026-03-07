"""
datetime_parser 模块

提供 datetime_parser 相关功能和接口。
"""

import logging
import re

# src / infrastructure / utils / datetime_parser.py
import pandas as pd

from datetime import datetime, timedelta
from functools import lru_cache
from src.infrastructure.utils.core.exceptions import DataLoaderError
from typing import Tuple

"""
基础设施层 - 工具组件组件

datetime_parser 模块

通用工具组件
提供工具组件相关的功能实现。
"""

logger = logging.getLogger(__name__)


# 日期时间解析常量
class DateTimeConstants:
    """日期时间解析相关常量"""

    # 默认时间窗口配置
    DEFAULT_WINDOW_SIZE_DAYS = 30  # 默认时间窗口大小(天)

    # 日期格式验证
    DATE_YEAR_DIGITS = 4  # 年份位数
    DATE_MONTH_DIGITS = 2  # 月份位数
    DATE_DAY_DIGITS = 2  # 日期位数

    # 时区配置
    BEIJING_TIMEZONE_OFFSET = "+08:00"  # 北京时区偏移
    UTC_TIMEZONE_MARKER = "Z"  # UTC时区标记

    # 默认日期时间
    DEFAULT_DATE = "1970-01-01"  # 默认日期
    DEFAULT_TIME = "00:00:00"  # 默认时间
    DEFAULT_TIMEZONE = "+08:00"  # 默认时区

    # 正则表达式模式中的数字
    TIMEZONE_OFFSET_DIGITS = 4  # 时区偏移位数 (HHMM)
    TIMEZONE_OFFSET_SHORT_DIGITS = 2  # 时区偏移位数 (HH)

    # 缓存配置
    CACHE_MAX_SIZE = 1000  # LRU缓存最大大小
    CACHE_TTL_SECONDS = 3600  # 缓存过期时间(秒)
    PARSE_CACHE_SIZE = 500  # 解析缓存大小
    FORMAT_CACHE_SIZE = 200  # 格式化缓存大小


class DateTimeParser:
    """日期时间处理工具类，包含解析、验证和生成功能"""

    @staticmethod
    @lru_cache(maxsize=DateTimeConstants.CACHE_MAX_SIZE)
    def get_dynamic_dates(
        window_size: int = DateTimeConstants.DEFAULT_WINDOW_SIZE_DAYS,
    ) -> Tuple[str, str]:
        """动态生成训练 / 预测日期范围 (缓存优化版)

        生成当前时间往前推window_size天的时间窗口

        Args:
            window_size: 时间窗口大小(天数)，默认30天

        Returns:
            Tuple[str, str]: (开始日期, 结束日期) 的元组，格式为'YYYY - MM - DD'
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=window_size)).strftime("%Y-%m-%d")
        return start_date, end_date

    @staticmethod
    @lru_cache(maxsize=DateTimeConstants.PARSE_CACHE_SIZE)
    def validate_dates(start: str, end: str):
        """验证日期格式和逻辑 (缓存优化版)"""
        date_regex = (
            r"^(?P<year>\d{{{}}})-(?P<month>\d{{{}}})-(?P<day>\d{{{}}})$".format(
                DateTimeConstants.DATE_YEAR_DIGITS,
                DateTimeConstants.DATE_MONTH_DIGITS,
                DateTimeConstants.DATE_DAY_DIGITS,
            )
        )

        def is_valid_date(date_str):
            match = re.match(date_regex, date_str)
            if not match:
                return False
            year, month, day = (
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
            try:
                datetime(year, month, day)
                return True
            except ValueError:
                return False

        if not is_valid_date(start) or not is_valid_date(end):
            raise DataLoaderError("日期格式不正确或无效")

        if pd.to_datetime(start) > pd.to_datetime(end):
            raise DataLoaderError("开始日期不能晚于结束日期")

    @staticmethod
    def parse_datetime_static(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
        """
        解析日期时间列

        Args:
        df: 包含日期时间数据的DataFrame
        date_col: 日期列名
        time_col: 时间列名

        Returns:
        包含解析后时间戳的DataFrame
        """
        # 创建副本避免修改原始数据
        df = df.copy()

        # 处理空DataFrame
        if df.empty:
            df["publish_time"] = pd.Series([], dtype='datetime64[ns]')
            return df

        # 标准化日期和时间格式
        df = DateTimeParser._normalize_date_and_time(df, date_col, time_col)

        # 合并日期时间字符串
        datetime_str = DateTimeParser._merge_datetime_strings(df, date_col, time_col)

        # 检查并添加时区信息
        datetime_str = DateTimeParser._add_timezone_if_missing(datetime_str)

        # 解析为datetime对象
        df["publish_time"] = DateTimeParser._parse_to_datetime(datetime_str)

        # 转换为本地时区
        df["publish_time"] = DateTimeParser._convert_to_local_timezone(
            df["publish_time"]
        )

        return df

    @staticmethod
    def _normalize_date_and_time(
        df: pd.DataFrame, date_col: str, time_col: str
    ) -> pd.DataFrame:
        """标准化日期和时间格式"""
        # 检查是否为空DataFrame
        if df.empty:
            return df
        
        # 标准化日期格式
        df[date_col] = df[date_col].astype(str).apply(DateTimeParser._normalize_date_format)

        # 标准化时间格式（保留时区信息）
        df[time_col] = df[time_col].astype(str).apply(DateTimeParser._normalize_time_format)

        return df

    @staticmethod
    def _merge_datetime_strings(
        df: pd.DataFrame, date_col: str, time_col: str
    ) -> pd.Series:
        """合并日期时间字符串"""
        return df[date_col] + " " + df[time_col]

    @staticmethod
    def _add_timezone_if_missing(datetime_str: pd.Series) -> pd.Series:
        """检查并添加缺失的时区信息"""
        return datetime_str.apply(
            lambda s: (
                s
                if re.search(
                    r"[+-]\d{{{}}}:?\d{{{}}}$".format(
                        DateTimeConstants.TIMEZONE_OFFSET_SHORT_DIGITS,
                        DateTimeConstants.TIMEZONE_OFFSET_SHORT_DIGITS,
                    ),
                    s,
                )
                or DateTimeConstants.UTC_TIMEZONE_MARKER in s
                else s + DateTimeConstants.BEIJING_TIMEZONE_OFFSET
            )
        )

    @staticmethod
    def _parse_to_datetime(datetime_str: pd.Series) -> pd.Series:
        """解析为datetime对象"""
        return pd.to_datetime(datetime_str, errors="coerce", format="mixed", utc=False)

    @staticmethod
    def _convert_to_local_timezone(dt_series: pd.Series) -> pd.Series:
        """转换为本地时区并去掉时区信息"""
        # 1. 有时区的时间转换为本地时间
        # 2. 无时区的时间直接使用（假设已经是本地时间）
        return dt_series.apply(
            lambda dt: (
                dt.tz_convert("Asia/Shanghai").tz_localize(None)
                if hasattr(dt, "tz_convert") and dt.tzinfo is not None
                else dt.replace(tzinfo=None) if hasattr(dt, "replace") else dt
            )
        )

    @staticmethod
    @lru_cache(maxsize=DateTimeConstants.FORMAT_CACHE_SIZE)
    def _normalize_date_format(date_str: str) -> str:
        """标准化日期格式为 YYYY - MM - DD (缓存优化版)"""
        if pd.isna(date_str) or not date_str.strip():
            return DateTimeConstants.DEFAULT_DATE

        # 尝试常见日期格式
        formats_to_try = [
            "%Y/%m/%d",  # 2023 / 01 / 01
            "%Y-%m-%d",  # 2023 - 02 - 01
            "%Y%m%d",  # 20230301
            "%Y年%m月%d日",  # 2023年01月01日
        ]

        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # 记录警告但继续处理
        logger.warning(f"无法识别的日期格式: {date_str}")
        return "1970 - 01 - 01"

    @staticmethod
    @lru_cache(maxsize=DateTimeConstants.FORMAT_CACHE_SIZE)
    def _normalize_time_format(time_str: str) -> str:
        """标准化时间格式为 HH:MM:SS，保留时区信息 (缓存优化版)"""
        if pd.isna(time_str) or not time_str.strip():
            return (
                DateTimeConstants.DEFAULT_TIME
                + " "
                + DateTimeConstants.DEFAULT_TIMEZONE
            )  # 默认添加北京时间时区

        # 处理带时区的时间（保留时区信息）
        if "+" in time_str or "Z" in time_str:
            # 保留完整的时区信息
            return time_str

        # 补全时间部分并添加默认时区（北京时间）
        time_str = DateTimeParser._complete_time_parts(time_str)
        return time_str + DateTimeConstants.BEIJING_TIMEZONE_OFFSET  # 添加北京时间时区

    @staticmethod
    @lru_cache(maxsize=DateTimeConstants.FORMAT_CACHE_SIZE)
    def _complete_time_parts(time_str: str) -> str:
        """补全时间部分的秒数 (缓存优化版)"""
        parts = time_str.split(":")
        if len(parts) == 1:  # 只有小时
            return f"{time_str}:00:00"
        elif len(parts) == 2:  # 有小时和分钟
            return f"{time_str}:00"
        return time_str  # 已经是完整格式

    @staticmethod
    def parse_datetime(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
        """独立导出的parse_datetime函数"""
        # 避免递归，直接调用静态方法
        return DateTimeParser.parse_datetime_static(df, date_col, time_col)

    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> bool:
        """验证日期范围
        
        Args:
            start_date: 开始日期字符串
            end_date: 结束日期字符串
            
        Returns:
            bool: 日期范围是否有效
        """
        try:
            # 解析日期
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # 验证开始日期不晚于结束日期
            if start > end:
                raise DataLoaderError("开始日期不能晚于结束日期")
            
            return True
        except Exception as e:
            logger.warning(f"日期范围验证失败: {e}")
            raise DataLoaderError(f"日期范围验证失败: {e}")