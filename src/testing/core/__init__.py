# Testing Core Module
# 测试核心模块

# This module contains core testing framework components
# 此模块包含测试框架核心组件

from .test_framework import TestFramework, TestRunner, UnitTestRunner, IntegrationTestRunner
from .test_execution import TestCaseManager
from .test_models import TestCase, TestSuite

__all__ = [
    'TestFramework',
    'TestRunner',
    'UnitTestRunner',
    'IntegrationTestRunner',
    'TestCaseManager',
    'TestCase',
    'TestSuite'
]
