#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 事件过滤器测试

全面测试事件过滤器模块的所有功能，包括：
- 事件类型过滤器
- 敏感数据过滤器
- 模式过滤器
- 时间范围过滤器
- 复合过滤器
- 过滤器链
- 过滤器工厂
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import re
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.infrastructure.security.filters.event_filters import (
    FilterType, IEventFilterComponent, EventTypeFilter, SensitiveDataFilter,
    PatternFilter, TimeRangeFilter, CompositeFilter, EventFilterChain,
    EventFilterFactory, IEventFilter
)


class TestFilterType:
    """FilterType枚举测试"""

    def test_filter_type_values(self):
        """测试过滤器类型值"""
        assert FilterType.INCLUDE.value == "include"
        assert FilterType.EXCLUDE.value == "exclude"

    def test_filter_type_members(self):
        """测试过滤器类型成员"""
        assert FilterType.INCLUDE in FilterType
        assert FilterType.EXCLUDE in FilterType
        assert len(FilterType) == 2


class TestEventTypeFilter:
    """EventTypeFilter测试"""

    def test_initialization(self):
        """测试初始化"""
        event_types = ["login", "logout"]
        filter_obj = EventTypeFilter(event_types, FilterType.INCLUDE)

        assert filter_obj.event_types == set(event_types)
        assert filter_obj.filter_type == FilterType.INCLUDE

    def test_should_process_include(self):
        """测试包含模式处理"""
        filter_obj = EventTypeFilter(["login", "access"], FilterType.INCLUDE)

        # 应该包含的事件
        assert filter_obj.should_process({"type": "login"}) == True
        assert filter_obj.should_process({"type": "access"}) == True

        # 不应该包含的事件
        assert filter_obj.should_process({"type": "logout"}) == False
        assert filter_obj.should_process({"type": "error"}) == False
        assert filter_obj.should_process({}) == False  # 无type字段

    def test_should_process_exclude(self):
        """测试排除模式处理"""
        filter_obj = EventTypeFilter(["error", "warning"], FilterType.EXCLUDE)

        # 应该排除的事件
        assert filter_obj.should_process({"type": "error"}) == False
        assert filter_obj.should_process({"type": "warning"}) == False

        # 不应该排除的事件
        assert filter_obj.should_process({"type": "login"}) == True
        assert filter_obj.should_process({"type": "access"}) == True

    def test_get_filter_info(self):
        """测试获取过滤器信息"""
        filter_obj = EventTypeFilter(["login", "logout"], FilterType.EXCLUDE)
        info = filter_obj.get_filter_info()

        assert info["type"] == "EventTypeFilter"
        assert set(info["event_types"]) == {"login", "logout"}
        assert info["filter_type"] == "exclude"


class TestSensitiveDataFilter:
    """SensitiveDataFilter测试"""

    def test_initialization(self):
        """测试初始化"""
        sensitive_keys = ["password", "token"]
        filter_obj = SensitiveDataFilter(sensitive_keys, "***", FilterType.EXCLUDE)

        assert filter_obj.sensitive_keys == set(sensitive_keys)
        assert filter_obj.replacement == "***"
        assert filter_obj.filter_type == FilterType.EXCLUDE

    def test_should_process_exclude_mode(self):
        """测试排除模式处理"""
        filter_obj = SensitiveDataFilter(["password"], filter_type=FilterType.EXCLUDE)

        # 包含敏感数据的事件 - 应该被排除
        event_with_sensitive = {"user": "john", "password": "secret123"}
        assert filter_obj.should_process(event_with_sensitive) == False

        # 不包含敏感数据的事件 - 应该被包含
        event_without_sensitive = {"user": "john", "action": "login"}
        assert filter_obj.should_process(event_without_sensitive) == True

    def test_should_process_include_mode(self):
        """测试包含模式处理"""
        filter_obj = SensitiveDataFilter(["password"], filter_type=FilterType.INCLUDE)

        # 包含敏感数据的事件 - 应该被包含
        event_with_sensitive = {"user": "john", "password": "secret123"}
        assert filter_obj.should_process(event_with_sensitive) == True

        # 不包含敏感数据的事件 - 应该被排除
        event_without_sensitive = {"user": "john", "action": "login"}
        assert filter_obj.should_process(event_without_sensitive) == False

    def test_contains_sensitive_data_dict(self):
        """测试字典中的敏感数据检测"""
        filter_obj = SensitiveDataFilter(["password", "token"])

        # 顶级敏感数据
        assert filter_obj._contains_sensitive_data({"password": "secret"}) == True
        assert filter_obj._contains_sensitive_data({"token": "abc123"}) == True

        # 嵌套敏感数据
        assert filter_obj._contains_sensitive_data({"user": {"password": "secret"}}) == True
        assert filter_obj._contains_sensitive_data({"data": {"auth": {"token": "xyz"}}}) == True

        # 无敏感数据
        assert filter_obj._contains_sensitive_data({"user": "john", "action": "login"}) == False

    def test_contains_sensitive_data_list(self):
        """测试列表中的敏感数据检测"""
        filter_obj = SensitiveDataFilter(["password"])

        # 列表中的敏感数据
        assert filter_obj._contains_sensitive_data([{"password": "secret"}, {"user": "john"}]) == True
        assert filter_obj._contains_sensitive_data([{"user": "john"}, {"action": "login"}]) == False

    def test_sanitize_data_dict(self):
        """测试字典数据清理"""
        filter_obj = SensitiveDataFilter(["password", "token"], "***")

        data = {
            "user": "john",
            "password": "secret123",
            "token": "abc123",
            "action": "login"
        }

        sanitized = filter_obj.sanitize_data(data)

        assert sanitized["user"] == "john"
        assert sanitized["password"] == "***"
        assert sanitized["token"] == "***"
        assert sanitized["action"] == "login"

    def test_sanitize_data_nested_dict(self):
        """测试嵌套字典数据清理"""
        filter_obj = SensitiveDataFilter(["password"], "***")

        data = {
            "user": "john",
            "auth": {
                "password": "secret123",
                "method": "basic"
            }
        }

        sanitized = filter_obj.sanitize_data(data)

        assert sanitized["user"] == "john"
        assert sanitized["auth"]["password"] == "***"
        assert sanitized["auth"]["method"] == "basic"

    def test_sanitize_data_list(self):
        """测试列表数据清理"""
        filter_obj = SensitiveDataFilter(["password"], "***")

        data = [
            {"user": "john", "password": "secret1"},
            {"user": "jane", "action": "login"}
        ]

        sanitized = filter_obj.sanitize_data(data)

        assert sanitized[0]["user"] == "john"
        assert sanitized[0]["password"] == "***"
        assert sanitized[1]["user"] == "jane"
        assert sanitized[1]["action"] == "login"

    def test_get_filter_info(self):
        """测试获取过滤器信息"""
        filter_obj = SensitiveDataFilter(["password", "token"], "###", FilterType.INCLUDE)
        info = filter_obj.get_filter_info()

        assert info["type"] == "SensitiveDataFilter"
        assert set(info["sensitive_keys"]) == {"password", "token"}
        assert info["replacement"] == "###"
        assert info["filter_type"] == "include"


class TestPatternFilter:
    """PatternFilter测试"""

    def test_initialization(self):
        """测试初始化"""
        filter_obj = PatternFilter(r"error.*", "message", FilterType.INCLUDE)

        assert filter_obj.pattern.pattern == r"error.*"
        assert filter_obj.field == "message"
        assert filter_obj.filter_type == FilterType.INCLUDE

    def test_should_process_include_mode(self):
        """测试包含模式处理"""
        filter_obj = PatternFilter(r"error.*", "message", FilterType.INCLUDE)

        # 匹配的事件
        assert filter_obj.should_process({"message": "error occurred"}) == True
        assert filter_obj.should_process({"message": "error: connection failed"}) == True

        # 不匹配的事件
        assert filter_obj.should_process({"message": "login successful"}) == False
        assert filter_obj.should_process({"message": "warning issued"}) == False

    def test_should_process_exclude_mode(self):
        """测试排除模式处理"""
        filter_obj = PatternFilter(r"debug.*", "message", FilterType.EXCLUDE)

        # 匹配的事件 - 应该被排除
        assert filter_obj.should_process({"message": "debug info"}) == False

        # 不匹配的事件 - 应该被包含
        assert filter_obj.should_process({"message": "error occurred"}) == True

    def test_should_process_non_string_field(self):
        """测试非字符串字段处理"""
        filter_obj = PatternFilter(r"test", "count", FilterType.INCLUDE)

        # 非字符串字段 - 不匹配
        assert filter_obj.should_process({"count": 42}) == False

    def test_should_process_missing_field(self):
        """测试缺失字段处理"""
        filter_obj = PatternFilter(r"test", "message", FilterType.INCLUDE)

        # 缺失字段 - 不匹配
        assert filter_obj.should_process({"type": "event"}) == False

    def test_get_filter_info(self):
        """测试获取过滤器信息"""
        filter_obj = PatternFilter(r"user.*", "data", FilterType.EXCLUDE)
        info = filter_obj.get_filter_info()

        assert info["type"] == "PatternFilter"
        assert info["pattern"] == r"user.*"
        assert info["field"] == "data"
        assert info["filter_type"] == "exclude"


class TestTimeRangeFilter:
    """TimeRangeFilter测试"""

    def test_initialization(self):
        """测试初始化"""
        start_time = "2023-01-01T00:00:00Z"
        end_time = "2023-12-31T23:59:59Z"

        filter_obj = TimeRangeFilter(start_time, end_time, FilterType.INCLUDE)

        assert filter_obj.start_time == start_time
        assert filter_obj.end_time == end_time
        assert filter_obj.filter_type == FilterType.INCLUDE

    def test_should_process_within_range(self):
        """测试时间范围内的处理"""
        start_time = "2023-01-01T00:00:00Z"
        end_time = "2023-12-31T23:59:59Z"

        filter_obj = TimeRangeFilter(start_time, end_time, FilterType.INCLUDE)

        # 在范围内的时间
        event_in_range = {"timestamp": "2023-06-15T12:00:00Z"}
        assert filter_obj.should_process(event_in_range) == True

    def test_should_process_before_range(self):
        """测试时间范围前的事件处理"""
        start_time = "2023-01-01T00:00:00Z"
        filter_obj = TimeRangeFilter(start_time, filter_type=FilterType.INCLUDE)

        # 在范围前的时间
        event_before = {"timestamp": "2022-12-31T23:59:59Z"}
        assert filter_obj.should_process(event_before) == False

    def test_should_process_after_range(self):
        """测试时间范围后的事件处理"""
        end_time = "2023-12-31T23:59:59Z"
        filter_obj = TimeRangeFilter(end_time=end_time, filter_type=FilterType.INCLUDE)

        # 在范围后的时间
        event_after = {"timestamp": "2024-01-01T00:00:01Z"}
        assert filter_obj.should_process(event_after) == False

    def test_should_process_no_timestamp(self):
        """测试无时间戳的事件处理"""
        filter_obj = TimeRangeFilter("2023-01-01T00:00:00Z", filter_type=FilterType.INCLUDE)

        # 无时间戳的事件
        event_no_time = {"type": "login", "user": "john"}
        assert filter_obj.should_process(event_no_time) == True

    def test_should_process_invalid_timestamp(self):
        """测试无效时间戳的事件处理"""
        filter_obj = TimeRangeFilter("2023-01-01T00:00:00Z", filter_type=FilterType.INCLUDE)

        # 无效时间戳的事件
        event_invalid_time = {"timestamp": "invalid-date"}
        assert filter_obj.should_process(event_invalid_time) == True

    def test_get_filter_info(self):
        """测试获取过滤器信息"""
        start_time = "2023-01-01T00:00:00Z"
        end_time = "2023-12-31T23:59:59Z"

        filter_obj = TimeRangeFilter(start_time, end_time, FilterType.EXCLUDE)
        info = filter_obj.get_filter_info()

        assert info["type"] == "TimeRangeFilter"
        assert info["start_time"] == start_time
        assert info["end_time"] == end_time
        assert info["filter_type"] == "exclude"


class TestCompositeFilter:
    """CompositeFilter测试"""

    def test_initialization(self):
        """测试初始化"""
        filters = [MagicMock(spec=IEventFilter), MagicMock(spec=IEventFilter)]
        filter_obj = CompositeFilter(filters, "AND")

        assert filter_obj.filters == filters
        assert filter_obj.operator == "AND"

    def test_should_process_and_operator(self):
        """测试AND操作符"""
        filter1 = MagicMock(spec=IEventFilter)
        filter1.should_process.return_value = True

        filter2 = MagicMock(spec=IEventFilter)
        filter2.should_process.return_value = True

        filter_obj = CompositeFilter([filter1, filter2], "AND")
        event = {"type": "test"}

        result = filter_obj.should_process(event)

        assert result == True
        filter1.should_process.assert_called_once_with(event)
        filter2.should_process.assert_called_once_with(event)

    def test_should_process_and_operator_false(self):
        """测试AND操作符（其中一个为False）"""
        filter1 = MagicMock(spec=IEventFilter)
        filter1.should_process.return_value = True

        filter2 = MagicMock(spec=IEventFilter)
        filter2.should_process.return_value = False

        filter_obj = CompositeFilter([filter1, filter2], "AND")
        event = {"type": "test"}

        result = filter_obj.should_process(event)

        assert result == False

    def test_should_process_or_operator(self):
        """测试OR操作符"""
        filter1 = MagicMock(spec=IEventFilter)
        filter1.should_process.return_value = False

        filter2 = MagicMock(spec=IEventFilter)
        filter2.should_process.return_value = True

        filter_obj = CompositeFilter([filter1, filter2], "OR")
        event = {"type": "test"}

        result = filter_obj.should_process(event)

        assert result == True

    def test_should_process_empty_filters(self):
        """测试空过滤器列表"""
        filter_obj = CompositeFilter([], "AND")
        event = {"type": "test"}

        result = filter_obj.should_process(event)

        assert result == True

    def test_should_process_invalid_operator(self):
        """测试无效操作符"""
        filter1 = MagicMock(spec=IEventFilter)
        filter1.should_process.return_value = True

        filter_obj = CompositeFilter([filter1], "INVALID")
        event = {"type": "test"}

        result = filter_obj.should_process(event)

        assert result == True

    def test_get_filter_info(self):
        """测试获取过滤器信息"""
        filter1 = MagicMock(spec=IEventFilter)
        filter1.get_filter_info.return_value = {"type": "MockFilter1"}

        filter2 = MagicMock(spec=IEventFilter)
        filter2.get_filter_info.return_value = {"type": "MockFilter2"}

        filter_obj = CompositeFilter([filter1, filter2], "OR")
        info = filter_obj.get_filter_info()

        assert info["type"] == "CompositeFilter"
        assert info["operator"] == "OR"
        assert len(info["filters"]) == 2
        assert info["filters"][0]["type"] == "MockFilter1"
        assert info["filters"][1]["type"] == "MockFilter2"


class TestEventFilterChain:
    """EventFilterChain测试"""

    def test_initialization(self):
        """测试初始化"""
        chain = EventFilterChain()
        assert chain.filters == []

    def test_add_filter(self):
        """测试添加过滤器"""
        chain = EventFilterChain()
        filter_obj = MagicMock(spec=IEventFilter)

        chain.add_filter(filter_obj)

        assert len(chain.filters) == 1
        assert chain.filters[0] == filter_obj

    def test_remove_filter(self):
        """测试移除过滤器"""
        chain = EventFilterChain()
        filter_obj = MagicMock(spec=IEventFilter)
        chain.add_filter(filter_obj)

        result = chain.remove_filter(filter_obj)

        assert result == True
        assert len(chain.filters) == 0

    def test_remove_nonexistent_filter(self):
        """测试移除不存在的过滤器"""
        chain = EventFilterChain()
        filter_obj = MagicMock(spec=IEventFilter)

        result = chain.remove_filter(filter_obj)

        assert result == False

    def test_clear_filters(self):
        """测试清空过滤器"""
        chain = EventFilterChain()
        chain.add_filter(MagicMock(spec=IEventFilter))
        chain.add_filter(MagicMock(spec=IEventFilter))

        chain.clear_filters()

        assert len(chain.filters) == 0

    def test_should_process_all_pass(self):
        """测试所有过滤器都通过的情况"""
        chain = EventFilterChain()

        filter1 = MagicMock(spec=IEventFilter)
        filter1.should_process.return_value = True

        filter2 = MagicMock(spec=IEventFilter)
        filter2.should_process.return_value = True

        chain.add_filter(filter1)
        chain.add_filter(filter2)

        event = {"type": "test"}
        result = chain.should_process(event)

        assert result == True
        filter1.should_process.assert_called_once_with(event)
        filter2.should_process.assert_called_once_with(event)

    def test_should_process_one_fail(self):
        """测试其中一个过滤器失败的情况"""
        chain = EventFilterChain()

        filter1 = MagicMock(spec=IEventFilter)
        filter1.should_process.return_value = True

        filter2 = MagicMock(spec=IEventFilter)
        filter2.should_process.return_value = False

        chain.add_filter(filter1)
        chain.add_filter(filter2)

        event = {"type": "test"}
        result = chain.should_process(event)

        assert result == False

    def test_process_event_should_process(self):
        """测试处理应该被处理的事件"""
        chain = EventFilterChain()

        filter_obj = MagicMock(spec=IEventFilter)
        filter_obj.should_process.return_value = True
        chain.add_filter(filter_obj)

        event = {"type": "test", "message": "hello"}
        result = chain.process_event(event)

        assert result == event

    def test_process_event_should_not_process(self):
        """测试处理不应该被处理的事件"""
        chain = EventFilterChain()

        filter_obj = MagicMock(spec=IEventFilter)
        filter_obj.should_process.return_value = False
        chain.add_filter(filter_obj)

        event = {"type": "test", "message": "hello"}
        result = chain.process_event(event)

        assert result is None

    def test_process_event_with_sensitive_data_filter(self):
        """测试带敏感数据过滤器的事件处理"""
        chain = EventFilterChain()

        # 添加类型过滤器
        type_filter = EventTypeFilter(["test"], FilterType.INCLUDE)
        chain.add_filter(type_filter)

        # 添加敏感数据过滤器
        sensitive_filter = SensitiveDataFilter(["password"], "***")
        chain.add_filter(sensitive_filter)

        event = {
            "type": "test",
            "message": "login attempt",
            "password": "secret123",
            "user": "john"
        }

        result = chain.process_event(event)

        assert result is not None
        assert result["type"] == "test"
        assert result["message"] == "login attempt"
        assert result["password"] == "***"
        assert result["user"] == "john"

    def test_get_filter_info(self):
        """测试获取过滤器链信息"""
        chain = EventFilterChain()

        filter1 = MagicMock(spec=IEventFilter)
        filter1.get_filter_info.return_value = {"type": "Filter1"}

        filter2 = MagicMock(spec=IEventFilter)
        filter2.get_filter_info.return_value = {"type": "Filter2"}

        chain.add_filter(filter1)
        chain.add_filter(filter2)

        info = chain.get_filter_info()

        assert info["filter_count"] == 2
        assert len(info["filters"]) == 2
        assert info["filters"][0]["type"] == "Filter1"
        assert info["filters"][1]["type"] == "Filter2"


class TestEventFilterFactory:
    """EventFilterFactory测试"""

    def test_create_type_filter(self):
        """测试创建事件类型过滤器"""
        event_types = ["login", "logout"]
        filter_obj = EventFilterFactory.create_type_filter(event_types, FilterType.EXCLUDE)

        assert isinstance(filter_obj, EventTypeFilter)
        assert filter_obj.event_types == set(event_types)
        assert filter_obj.filter_type == FilterType.EXCLUDE

    def test_create_sensitive_filter(self):
        """测试创建敏感数据过滤器"""
        sensitive_keys = ["password", "token"]
        filter_obj = EventFilterFactory.create_sensitive_filter(
            sensitive_keys, "###", FilterType.INCLUDE
        )

        assert isinstance(filter_obj, SensitiveDataFilter)
        assert filter_obj.sensitive_keys == set(sensitive_keys)
        assert filter_obj.replacement == "###"
        assert filter_obj.filter_type == FilterType.INCLUDE

    def test_create_pattern_filter(self):
        """测试创建模式过滤器"""
        filter_obj = EventFilterFactory.create_pattern_filter(
            r"user.*", "data", FilterType.EXCLUDE
        )

        assert isinstance(filter_obj, PatternFilter)
        assert filter_obj.pattern.pattern == r"user.*"
        assert filter_obj.field == "data"
        assert filter_obj.filter_type == FilterType.EXCLUDE

    def test_create_time_filter(self):
        """测试创建时间范围过滤器"""
        start_time = "2023-01-01T00:00:00Z"
        end_time = "2023-12-31T23:59:59Z"

        filter_obj = EventFilterFactory.create_time_filter(
            start_time, end_time, FilterType.INCLUDE
        )

        assert isinstance(filter_obj, TimeRangeFilter)
        assert filter_obj.start_time == start_time
        assert filter_obj.end_time == end_time
        assert filter_obj.filter_type == FilterType.INCLUDE

    def test_create_composite_filter(self):
        """测试创建复合过滤器"""
        filters = [MagicMock(spec=IEventFilter), MagicMock(spec=IEventFilter)]
        filter_obj = EventFilterFactory.create_composite_filter(filters, "OR")

        assert isinstance(filter_obj, CompositeFilter)
        assert filter_obj.filters == filters
        assert filter_obj.operator == "OR"

    def test_create_default_filter_chain(self):
        """测试创建默认过滤器链"""
        chain = EventFilterFactory.create_default_filter_chain()

        assert isinstance(chain, EventFilterChain)
        assert len(chain.filters) == 1

        # 第一个过滤器应该是敏感数据过滤器
        sensitive_filter = chain.filters[0]
        assert isinstance(sensitive_filter, SensitiveDataFilter)
        assert "password" in sensitive_filter.sensitive_keys
        assert "token" in sensitive_filter.sensitive_keys
        assert "secret" in sensitive_filter.sensitive_keys
        assert "key" in sensitive_filter.sensitive_keys
        assert "credential" in sensitive_filter.sensitive_keys


class TestEventFiltersIntegration:
    """事件过滤器集成测试"""

    def test_complete_filter_chain_workflow(self):
        """测试完整的过滤器链工作流程"""
        # 创建过滤器链
        chain = EventFilterChain()

        # 添加事件类型过滤器（只包含安全事件）
        type_filter = EventTypeFilter(["security", "auth"], FilterType.INCLUDE)
        chain.add_filter(type_filter)

        # 添加敏感数据过滤器
        sensitive_filter = SensitiveDataFilter(["password", "token"], "***")
        chain.add_filter(sensitive_filter)

        # 添加模式过滤器（排除调试消息）
        pattern_filter = PatternFilter(r"debug.*", "message", FilterType.EXCLUDE)
        chain.add_filter(pattern_filter)

        # 测试事件1: 应该被处理的安全事件
        event1 = {
            "type": "security",
            "message": "login successful",
            "user": "john",
            "password": "secret123",
            "timestamp": "2023-01-01T12:00:00Z"
        }

        assert chain.should_process(event1) == True
        processed_event1 = chain.process_event(event1)
        assert processed_event1 is not None
        assert processed_event1["password"] == "***"
        assert processed_event1["message"] == "login successful"

        # 测试事件2: 不匹配的事件类型
        event2 = {
            "type": "system",
            "message": "cpu usage high",
            "timestamp": "2023-01-01T12:00:00Z"
        }

        assert chain.should_process(event2) == False
        processed_event2 = chain.process_event(event2)
        assert processed_event2 is None

        # 测试事件3: 调试消息
        event3 = {
            "type": "security",
            "message": "debug: connection pool size",
            "timestamp": "2023-01-01T12:00:00Z"
        }

        assert chain.should_process(event3) == False
        processed_event3 = chain.process_event(event3)
        assert processed_event3 is None

    def test_factory_and_chain_integration(self):
        """测试工厂和过滤器链的集成"""
        # 使用工厂创建过滤器链
        chain = EventFilterFactory.create_default_filter_chain()

        # 使用工厂添加更多过滤器
        type_filter = EventFilterFactory.create_type_filter(["security"], FilterType.INCLUDE)
        pattern_filter = EventFilterFactory.create_pattern_filter(r"login.*", "message")

        chain.add_filter(type_filter)
        chain.add_filter(pattern_filter)

        # 测试集成效果
        event = {
            "type": "security",
            "message": "login attempt",
            "password": "secret123",
            "timestamp": "2023-01-01T12:00:00Z"
        }

        # 应该被处理（匹配类型和模式，且敏感数据被清理）
        assert chain.should_process(event) == True
        processed = chain.process_event(event)
        assert processed is not None
        assert processed["password"] == "***"
        assert processed["message"] == "login attempt"

    def test_complex_composite_filter(self):
        """测试复杂的复合过滤器"""
        # 创建多个基础过滤器
        type_filter1 = EventTypeFilter(["security"], FilterType.INCLUDE)
        type_filter2 = EventTypeFilter(["audit"], FilterType.INCLUDE)
        pattern_filter = PatternFilter(r"admin.*", "user", FilterType.INCLUDE)

        # 创建OR复合过滤器（安全事件或审计事件）
        type_composite = CompositeFilter([type_filter1, type_filter2], "OR")

        # 创建AND复合过滤器（类型复合 + 模式）
        final_composite = CompositeFilter([type_composite, pattern_filter], "AND")

        # 测试事件
        events = [
            {"type": "security", "user": "admin_john", "action": "login"},  # 应该匹配
            {"type": "audit", "user": "admin_jane", "action": "query"},     # 应该匹配
            {"type": "security", "user": "user_bob", "action": "view"},     # 不匹配（用户不是admin开头）
            {"type": "system", "user": "admin_tom", "action": "restart"},   # 不匹配（类型不匹配）
        ]

        results = [final_composite.should_process(event) for event in events]
        expected = [True, True, False, False]

        assert results == expected

    def test_filter_performance_with_large_dataset(self):
        """测试过滤器在大数据集上的性能"""
        # 创建过滤器链
        chain = EventFilterChain()
        chain.add_filter(EventTypeFilter(["security"], FilterType.INCLUDE))
        chain.add_filter(SensitiveDataFilter(["password", "token"]))

        # 生成大量测试事件
        events = []
        for i in range(1000):
            event = {
                "type": "security" if i % 2 == 0 else "system",
                "id": i,
                "message": f"event {i}",
                "password": f"secret{i}" if i % 10 == 0 else None,  # 每10个事件有一个敏感数据
                "timestamp": "2023-01-01T12:00:00Z"
            }
            events.append(event)

        # 批量处理
        import time
        start_time = time.time()

        processed_events = []
        for event in events:
            processed = chain.process_event(event)
            if processed:
                processed_events.append(processed)

        processing_time = time.time() - start_time

        # 验证结果
        assert len(processed_events) == 500  # 应该有一半的安全事件被处理

        # 检查敏感数据被清理
        for event in processed_events:
            if event.get("password"):
                assert event["password"] == "***"

        # 性能检查（1000个事件应该在合理时间内处理）
        assert processing_time < 2.0  # 2秒内完成

        print(".4f")

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        chain = EventFilterChain()

        # 添加可能抛出异常的过滤器（模拟）
        class FailingFilter(IEventFilter):
            def should_process(self, event):
                if event.get("fail", False):
                    raise ValueError("Simulated failure")
                return True

            def get_filter_info(self):
                return {"type": "FailingFilter"}

        chain.add_filter(FailingFilter())
        chain.add_filter(EventTypeFilter(["test"], FilterType.INCLUDE))

        # 正常事件 - 应该通过
        normal_event = {"type": "test", "message": "normal"}
        assert chain.should_process(normal_event) == True

        # 故障事件 - 应该失败
        failing_event = {"type": "test", "message": "fail", "fail": True}

        with pytest.raises(ValueError, match="Simulated failure"):
            chain.should_process(failing_event)
