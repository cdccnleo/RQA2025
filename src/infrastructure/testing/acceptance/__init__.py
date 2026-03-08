"""
用户验收测试模块

User Acceptance Testing Module

Provides user acceptance testing framework and execution capabilities.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from .enums import UserRole, AcceptanceTestType, TestScenario
from .models import UserAcceptanceTest, TestExecutionResult, TestStatus
from .test_manager import AcceptanceTestManager
from .test_executor import UserAcceptanceTestExecutor

__all__ = [
    'UserRole', 'AcceptanceTestType', 'TestScenario',
    'UserAcceptanceTest', 'TestExecutionResult', 'TestStatus',
    'AcceptanceTestManager', 'UserAcceptanceTestExecutor'
]
