"""
用户验收测试数据模型

Data models for user acceptance testing.

Extracted from user_acceptance_tester.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# 需要从integration导入TestStatus，或创建本地定义
try:
    from ..integration.system_integration_tester import TestStatus
except ImportError:
    from enum import Enum
    class TestStatus(Enum):
        """测试状态枚举"""
        PENDING = "pending"
        RUNNING = "running"
        PASSED = "passed"
        FAILED = "failed"
        ERROR = "error"
        SKIPPED = "skipped"

from .enums import UserRole, AcceptanceTestType, TestScenario


@dataclass
class UserAcceptanceTest:
    """用户验收测试"""
    test_id: str
    test_name: str
    test_type: AcceptanceTestType
    scenario: TestScenario
    target_users: List[UserRole]
    prerequisites: List[str]
    test_steps: List[Dict[str, Any]]
    expected_results: List[str]
    acceptance_criteria: List[str]
    priority: int  # 1-5, 1最高
    estimated_duration: int  # 分钟
    status: TestStatus = TestStatus.PENDING
    assigned_tester: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'scenario': self.scenario.value,
            'target_users': [role.value for role in self.target_users],
            'prerequisites': self.prerequisites,
            'test_steps': self.test_steps,
            'expected_results': self.expected_results,
            'acceptance_criteria': self.acceptance_criteria,
            'priority': self.priority,
            'estimated_duration': self.estimated_duration,
            'status': self.status.value,
            'assigned_tester': self.assigned_tester
        }


@dataclass  
class TestExecutionResult:
    """测试执行结果"""
    execution_id: str
    test_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    status: TestStatus
    executed_by: str
    environment: str
    results: Dict[str, Any]
    defects_found: List[Dict[str, Any]]
    user_feedback: List[Dict[str, Any]]
    screenshots: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execution_id': self.execution_id,
            'test_id': self.test_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'status': self.status.value,
            'executed_by': self.executed_by,
            'environment': self.environment,
            'results': self.results,
            'defects_found': self.defects_found,
            'user_feedback': self.user_feedback,
            'screenshots': self.screenshots
        }


__all__ = [
    'UserAcceptanceTest',
    'TestExecutionResult',
    'TestStatus'
]
