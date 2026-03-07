#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils Helpers Module测试

测试工具辅助模块的功能
"""

import pytest
from datetime import datetime


class TestUtilsHelpers:
    """测试UtilsHelpers类"""

    def test_utils_helpers_init(self):
        """测试UtilsHelpers初始化"""
        from src.utils.helpers import UtilsHelpers

        helpers = UtilsHelpers()
        assert helpers.initialized is True

    def test_format_datetime_valid(self):
        """测试format_datetime方法 - 有效日期时间"""
        from src.utils.helpers import UtilsHelpers

        helpers = UtilsHelpers()
        dt = datetime(2023, 12, 25, 10, 30, 45)
        result = helpers.format_datetime(dt)
        assert isinstance(result, str)
        assert "2023-12-25" in result

    def test_format_datetime_none(self):
        """测试format_datetime方法 - None值"""
        from src.utils.helpers import UtilsHelpers

        helpers = UtilsHelpers()
        result = helpers.format_datetime(None)
        assert result is None

    def test_safe_divide_normal(self):
        """测试safe_divide方法 - 正常除法"""
        from src.utils.helpers import UtilsHelpers

        helpers = UtilsHelpers()
        result = helpers.safe_divide(10, 2)
        assert result == 5.0

    def test_safe_divide_zero_denominator(self):
        """测试safe_divide方法 - 除数为零"""
        from src.utils.helpers import UtilsHelpers

        helpers = UtilsHelpers()
        result = helpers.safe_divide(10, 0)
        assert result == 0

    def test_safe_divide_custom_default(self):
        """测试safe_divide方法 - 自定义默认值"""
        from src.utils.helpers import UtilsHelpers

        helpers = UtilsHelpers()
        result = helpers.safe_divide(10, 0, default=999)
        assert result == 999

    def test_safe_divide_type_error(self):
        """测试safe_divide方法 - 类型错误"""
        from src.utils.helpers import UtilsHelpers

        helpers = UtilsHelpers()
        result = helpers.safe_divide("a", "b")
        assert result == 0
