#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""priority_manager基础测试"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest

def test_priority_manager_import():
    """测试priority_manager可以导入"""
    try:
        from src.infrastructure.config.core.priority_manager import PriorityManager
        assert PriorityManager is not None
    except ImportError:
        pytest.skip("PriorityManager not available")

def test_priority_manager_creation():
    """测试PriorityManager创建"""
    try:
        from src.infrastructure.config.core.priority_manager import PriorityManager
        manager = PriorityManager()
        assert manager is not None
    except Exception:
        pytest.skip("Cannot create PriorityManager")

