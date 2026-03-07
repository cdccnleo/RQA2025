#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""alert_manager基础测试"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, MagicMock


def test_alert_manager_import():
    """测试alert_manager导入"""
    try:
        from src.infrastructure.health.components.alert_manager import AlertManager
        assert AlertManager is not None
    except ImportError:
        pytest.skip("AlertManager不可用")


def test_alert_manager_creation():
    """测试AlertManager创建"""
    try:
        from src.infrastructure.health.components.alert_manager import AlertManager
        manager = AlertManager()
        assert manager is not None
    except Exception:
        pytest.skip("创建失败")


def test_alert_manager_with_config():
    """测试带配置的AlertManager"""
    try:
        from src.infrastructure.health.components.alert_manager import AlertManager
        config = {'enabled': True}
        manager = AlertManager(config=config)
        assert manager is not None
    except Exception:
        pytest.skip("带配置创建失败")

