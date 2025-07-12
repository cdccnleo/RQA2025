import pytest
from datetime import datetime
from src.infrastructure.utils.date_utils import DateUtils

class TestDateUtils:
    def test_parse_date(self):
        """测试日期解析功能"""
        date_str = "2024-01-01"
        result = DateUtils.parse_date(date_str)
        assert result == datetime(2024, 1, 1)

    def test_format_date(self):
        """测试日期格式化功能"""
        date = datetime(2024, 1, 1)
        result = DateUtils.format_date(date)
        assert result == "2024-01-01"

    def test_date_diff(self):
        """测试日期差值计算"""
        date1 = datetime(2024, 1, 1)
        date2 = datetime(2024, 1, 2)
        result = DateUtils.date_diff(date1, date2)
        assert result == 1

    def test_timezone_conversion(self):
        """测试时区转换"""
        date = datetime(2024, 1, 1, 12, 0, 0)  # UTC
        result = DateUtils.convert_timezone(date, "Asia/Shanghai")
        assert result.hour == 20  # UTC+8
