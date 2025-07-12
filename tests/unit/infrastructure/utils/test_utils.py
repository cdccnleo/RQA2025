#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 基础设施工具单元测试
测试src/infrastructure/utils下的工具模块
"""

import pytest
from datetime import datetime
from src.infrastructure.utils import date_utils, exception_utils

class TestDateUtils:
    """测试日期工具函数"""

    def test_format_date(self):
        """测试日期格式化"""
        dt = datetime(2023, 1, 15)
        formatted = date_utils.format_date(dt, "%Y-%m-%d")
        assert formatted == "2023-01-15"

    def test_parse_date(self):
        """测试日期解析"""
        dt = date_utils.parse_date("2023-01-15", "%Y-%m-%d")
        assert dt.year == 2023
        assert dt.month == 1
        assert dt.day == 15

class TestExceptionUtils:
    """测试异常工具函数"""

    def test_is_timeout_error(self):
        """测试超时错误识别"""
        class TimeoutError(Exception): pass

        assert exception_utils.is_timeout_error(TimeoutError()) is True
        assert exception_utils.is_timeout_error(ValueError()) is False

@pytest.fixture
def sample_date():
    """提供测试用的日期fixture"""
    return datetime(2023, 1, 15)

def test_date_utils_with_fixture(sample_date):
    """使用fixture测试日期工具"""
    formatted = date_utils.format_date(sample_date, "%Y/%m/%d")
    assert formatted == "2023/01/15"
