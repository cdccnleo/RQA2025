#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层日期时间解析工具组件测试

测试目标：提升utils/tools/datetime_parser.py的真实覆盖率
实际导入和使用src.infrastructure.utils.tools.datetime_parser模块
"""

import pytest
from datetime import datetime, timedelta


class TestDateTimeConstants:
    """测试日期时间常量类"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeConstants
        
        assert DateTimeConstants.DEFAULT_WINDOW_SIZE_DAYS == 30
        assert DateTimeConstants.DATE_YEAR_DIGITS == 4
        assert DateTimeConstants.DATE_MONTH_DIGITS == 2
        assert DateTimeConstants.DATE_DAY_DIGITS == 2
        assert DateTimeConstants.BEIJING_TIMEZONE_OFFSET == "+08:00"
        assert DateTimeConstants.UTC_TIMEZONE_MARKER == "Z"
        assert DateTimeConstants.DEFAULT_DATE == "1970-01-01"
        assert DateTimeConstants.DEFAULT_TIME == "00:00:00"
        assert DateTimeConstants.CACHE_MAX_SIZE == 1000


class TestDateTimeParser:
    """测试日期时间解析器类"""
    
    def test_get_dynamic_dates(self):
        """测试动态生成日期范围"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeParser
        
        start_date, end_date = DateTimeParser.get_dynamic_dates(window_size=7)
        
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
        assert len(start_date) == 10  # YYYY-MM-DD格式
        assert len(end_date) == 10
    
    def test_get_dynamic_dates_default(self):
        """测试使用默认窗口大小生成日期范围"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeParser
        
        start_date, end_date = DateTimeParser.get_dynamic_dates()
        
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
    
    def test_validate_dates_valid(self):
        """测试验证有效日期"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeParser
        
        # validate_dates在验证通过时不返回任何值（None），验证失败时抛出异常
        try:
            DateTimeParser.validate_dates("2024-01-01", "2024-01-31")
            assert True  # 如果没有抛出异常，说明验证通过
        except Exception:
            assert False
    
    def test_validate_dates_invalid_format(self):
        """测试验证无效格式日期"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeParser
        from src.infrastructure.utils.core.exceptions import DataLoaderError
        
        # validate_dates在验证失败时会抛出异常
        with pytest.raises(DataLoaderError):
            DateTimeParser.validate_dates("2024/01/01", "2024-01-31")
    
    def test_validate_dates_invalid_logic(self):
        """测试验证逻辑无效日期（开始日期晚于结束日期）"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeParser
        from src.infrastructure.utils.core.exceptions import DataLoaderError
        
        # validate_dates在验证失败时会抛出异常
        with pytest.raises(DataLoaderError):
            DateTimeParser.validate_dates("2024-01-31", "2024-01-01")
    
    def test_validate_dates_invalid_date(self):
        """测试验证无效日期值"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeParser
        from src.infrastructure.utils.core.exceptions import DataLoaderError
        
        # validate_dates在验证失败时会抛出异常
        with pytest.raises(DataLoaderError):
            DateTimeParser.validate_dates("2024-13-01", "2024-01-31")

