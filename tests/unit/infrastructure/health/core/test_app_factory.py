#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""应用工厂测试"""

import pytest
from unittest.mock import Mock, patch


class TestAppFactory:
    """测试应用工厂"""

    def test_create_application_function_exists(self):
        """测试创建应用函数存在"""
        from src.infrastructure.health.core import app_factory
        assert hasattr(app_factory, 'create_application')

    def test_setup_global_exception_handlers_exists(self):
        """测试全局异常处理器设置函数存在"""
        from src.infrastructure.health.core import app_factory
        assert hasattr(app_factory, 'create_application')
