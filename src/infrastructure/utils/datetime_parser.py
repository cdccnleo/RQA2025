# src/infrastructure/utils/datetime_parser.py
from datetime import datetime, timedelta
import re
import pandas as pd
from typing import Tuple
from .logger import get_logger
from src.infrastructure.utils.exception_utils import DataLoaderError

logger = get_logger(__name__)


class DateTimeParser:
    """日期时间处理工具类，包含解析、验证和生成功能"""
    
    @staticmethod
    def get_dynamic_dates(window_size: int = 30) -> Tuple[str, str]:
        """动态生成训练/预测日期范围

        生成当前时间往前推window_size天的时间窗口

        Args:
            window_size: 时间窗口大小（天数），默认30天

        Returns:
            Tuple[str, str]: (开始日期, 结束日期) 的元组，格式为'YYYY-MM-DD'
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=window_size)).strftime("%Y-%m-%d")
        return start_date, end_date

    @staticmethod
    def validate_dates(start: str, end: str):
        """验证日期格式和逻辑"""
        date_format = "%Y-%m-%d"
        date_regex = r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$"

        def is_valid_date(date_str):
            match = re.match(date_regex, date_str)
            if not match:
                return False
            year, month, day = int(match.group("year")), int(match.group("month")), int(match.group("day"))
            try:
                datetime(year, month, day)
                return True
            except ValueError:
                return False

        if not is_valid_date(start) or not is_valid_date(end):
            raise DataLoaderError("日期格式不正确或无效")

        if pd.to_datetime(start) > pd.to_datetime(end):
            raise DataLoaderError("开始日期不能晚于结束日期")
    """日期时间解析工具类"""

    @staticmethod
    def parse_datetime(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
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

        # 标准化日期格式
        df[date_col] = df[date_col].apply(DateTimeParser._normalize_date_format)

        # 标准化时间格式（保留时区信息）
        df[time_col] = df[time_col].apply(DateTimeParser._normalize_time_format)

        # 合并日期和时间
        datetime_str = df[date_col] + " " + df[time_col]

        # 检查并添加时区信息（如果缺失）
        datetime_str = datetime_str.apply(
            lambda s: s if re.search(r'[+-]\d{2}:?\d{2}$', s) or 'Z' in s else s + '+08:00'
        )

        # 解析为datetime对象
        df['publish_time'] = pd.to_datetime(
            datetime_str,
            errors='coerce',
            format='mixed',
            utc=False
        )

        # 新增：转换为本地时区并去掉时区信息
        # 1. 有时区的时间转换为本地时间
        # 2. 无时区的时间直接使用（假设已经是本地时间）
        df['publish_time'] = df['publish_time'].apply(
            lambda dt:
            dt.tz_convert('Asia/Shanghai').tz_localize(None)
            if dt.tzinfo is not None
            else dt.replace(tzinfo=None)
        )

        valid_df = df[df['publish_time'].notna()].copy()
        return valid_df

    @staticmethod
    def _normalize_date_format(date_str: str) -> str:
        """标准化日期格式为 YYYY-MM-DD"""
        if pd.isna(date_str) or not date_str.strip():
            return "1970-01-01"

        # 尝试常见日期格式
        formats_to_try = [
            "%Y/%m/%d",  # 2023/01/01
            "%Y-%m-%d",  # 2023-02-01
            "%Y%m%d",  # 20230301
            "%Y年%m月%d日"  # 2023年01月01日
        ]

        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # 记录警告但继续处理
        logger.warning(f"无法识别的日期格式: {date_str}")
        return "1970-01-01"

    @staticmethod
    def _normalize_time_format(time_str: str) -> str:
        """标准化时间格式为 HH:MM:SS，保留时区信息"""
        if pd.isna(time_str) or not time_str.strip():
            return "00:00:00+08:00"  # 默认添加北京时间时区

        # 处理带时区的时间（保留时区信息）
        if '+' in time_str or 'Z' in time_str:
            # 保留完整的时区信息
            return time_str

        # 补全时间部分并添加默认时区（北京时间）
        time_str = DateTimeParser._complete_time_parts(time_str)
        return time_str + "+08:00"  # 添加北京时间时区

    @staticmethod
    def _complete_time_parts(time_str: str) -> str:
        """补全时间部分的秒数"""
        parts = time_str.split(':')
        if len(parts) == 1:  # 只有小时
            return f"{time_str}:00:00"
        elif len(parts) == 2:  # 有小时和分钟
            return f"{time_str}:00"
        return time_str  # 已经是完整格式

def parse_datetime(df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
    """独立导出的parse_datetime函数"""
    return DateTimeParser.parse_datetime(df, date_col, time_col)
