"""
RQA2025 测试层模块

提供完整的测试框架和质量保障体系，包括测试用例管理、执行引擎和结果分析

Test Layer Module for RQA2025

Provides comprehensive testing framework and quality assurance system.

Note: This module has been refactored to eliminate code duplication.
All class definitions have been extracted to separate modules in core/.

Author: RQA2025 Development Team
Date: 2025-11-01 (Emergency refactoring to fix 100% code duplication issue)
"""

from .core.test_models import (
    TestPriority,
    TestCategory,
    TestStatus,
    TestCase,
    TestSuite
)

from .core.test_execution import (
    TestExecutionResult,
    TestCaseManager,
    test_case_manager
)

__all__ = [
    'TestPriority',
    'TestCategory',
    'TestStatus',
    'TestCase',
    'TestSuite',
    'TestExecutionResult',
    'TestCaseManager',
    'test_case_manager'
]
