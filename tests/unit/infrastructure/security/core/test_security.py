#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 安全管理器测试

测试SecurityManager类的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime
from typing import Dict, Any

from src.infrastructure.security.core.security import SecurityManager


class TestSecurityManager:
    """安全管理器测试"""

    @pytest.fixture
    def security_manager(self):
        """安全管理器fixture"""
        return SecurityManager()

    def test_initialization(self, security_manager):
        """测试初始化"""
        assert isinstance(security_manager.filters, list)
        assert isinstance(security_manager.audit_log, list)
        assert len(security_manager.filters) == 0
        assert len(security_manager.audit_log) == 0

    def test_add_filter(self, security_manager):
        """测试添加过滤器"""
        def test_filter(data):
            return data

        security_manager.add_filter(test_filter)
        assert len(security_manager.filters) == 1
        assert security_manager.filters[0] == test_filter

    def test_apply_filters_empty(self, security_manager):
        """测试应用空过滤器列表"""
        test_data = {"key": "value"}
        result = security_manager.apply_filters(test_data)
        assert result == test_data
        assert result is not test_data  # 应该返回副本

    def test_apply_filters_single(self, security_manager):
        """测试应用单个过滤器"""
        def upper_filter(data):
            return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}

        security_manager.add_filter(upper_filter)
        test_data = {"name": "test", "value": 123}
        result = security_manager.apply_filters(test_data)

        assert result["name"] == "TEST"
        assert result["value"] == 123

    def test_apply_filters_multiple(self, security_manager):
        """测试应用多个过滤器"""
        def add_prefix(data):
            return {k: f"prefix_{v}" if isinstance(v, str) else v for k, v in data.items()}

        def upper_case(data):
            return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}

        security_manager.add_filter(add_prefix)
        security_manager.add_filter(upper_case)

        test_data = {"name": "test"}
        result = security_manager.apply_filters(test_data)

        assert result["name"] == "PREFIX_TEST"

    def test_log_security_event(self, security_manager):
        """测试记录安全事件"""
        event = "test_event"
        details = {"action": "login", "user": "test_user"}

        security_manager.log_security_event(event, details)

        assert len(security_manager.audit_log) == 1
        log_entry = security_manager.audit_log[0]

        assert log_entry["event"] == event
        assert log_entry["details"] == details
        assert "timestamp" in log_entry

        # 验证时间戳格式
        timestamp = log_entry["timestamp"]
        assert isinstance(timestamp, str)
        # 应该能够解析为datetime
        datetime.fromisoformat(timestamp)

    def test_log_multiple_events(self, security_manager):
        """测试记录多个安全事件"""
        events = [
            ("login", {"user": "user1"}),
            ("logout", {"user": "user2"}),
            ("access", {"resource": "file.txt"})
        ]

        for event, details in events:
            security_manager.log_security_event(event, details)

        assert len(security_manager.audit_log) == 3

        for i, (expected_event, expected_details) in enumerate(events):
            log_entry = security_manager.audit_log[i]
            assert log_entry["event"] == expected_event
            assert log_entry["details"] == expected_details

    def test_get_security_status_empty(self, security_manager):
        """测试获取空状态"""
        status = security_manager.get_security_status()

        assert status["active_filters"] == 0
        assert status["audit_entries"] == 0
        assert status["status"] == "active"

    def test_get_security_status_with_data(self, security_manager):
        """测试获取有数据的状态"""
        # 添加过滤器
        security_manager.add_filter(lambda x: x)

        # 添加审计日志
        security_manager.log_security_event("test", {})

        status = security_manager.get_security_status()

        assert status["active_filters"] == 1
        assert status["audit_entries"] == 1
        assert status["status"] == "active"

    def test_filter_modification_isolation(self, security_manager):
        """测试过滤器修改的隔离性"""
        original_data = {"key": "original"}
        modified_data = {"key": "modified"}

        def modifying_filter(data):
            data["key"] = "modified"
            return data

        security_manager.add_filter(modifying_filter)

        # 应用过滤器
        result = security_manager.apply_filters(original_data)

        # 原始数据不应该被修改
        assert original_data["key"] == "original"
        assert result["key"] == "modified"

    def test_audit_log_immutability(self, security_manager):
        """测试审计日志的不可变性"""
        details = {"mutable": ["item"]}
        security_manager.log_security_event("test", details)

        # 修改原始details
        details["mutable"].append("new_item")

        # 审计日志中的条目不应该被修改
        logged_details = security_manager.audit_log[0]["details"]
        assert len(logged_details["mutable"]) == 1
        assert logged_details["mutable"][0] == "item"
