"""
测试数据模型

提供测试层的核心数据模型，包括枚举、数据类等。

Extracted from __init__.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TestPriority(Enum):
    """测试优先级枚举"""
    CRITICAL = 1    # 关键测试 - 系统核心功能
    HIGH = 2        # 高优先级 - 重要功能
    MEDIUM = 3      # 中等优先级 - 一般功能
    LOW = 4         # 低优先级 - 辅助功能


class TestCategory(Enum):
    """测试类别枚举"""
    SMOKE = "smoke"                    # 冒烟测试
    FUNCTIONAL = "functional"          # 功能测试
    INTEGRATION = "integration"       # 集成测试
    REGRESSION = "regression"          # 回归测试
    PERFORMANCE = "performance"       # 性能测试
    SECURITY = "security"              # 安全测试
    COMPLIANCE = "compliance"          # 合规测试
    ACCEPTANCE = "acceptance"          # 验收测试


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"        # 待执行
    RUNNING = "running"        # 运行中
    PASSED = "passed"          # 通过
    FAILED = "failed"          # 失败
    SKIPPED = "skipped"        # 跳过
    BLOCKED = "blocked"        # 阻塞
    ERROR = "error"            # 错误


@dataclass
class TestCase:
    """测试用例类"""
    test_id: str
    name: str
    description: str
    category: TestCategory
    priority: TestPriority
    estimated_duration: float  # 预估执行时间（秒）
    dependencies: List[str]    # 依赖的测试用例ID
    tags: List[str]            # 标签
    test_function: Optional[Callable] = None

    # 运行时状态
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result_details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    # 统计信息
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_execution: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'priority': self.priority.value,
            'estimated_duration': self.estimated_duration,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result_details': self.result_details,
            'error_message': self.error_message,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'success_rate': self.success_count / max(self.execution_count, 1) * 100
        }


@dataclass
class TestSuite:
    """测试套件类"""
    suite_id: str
    name: str
    description: str
    test_cases: List[TestCase]
    priority: TestPriority = TestPriority.MEDIUM
    timeout: int = 3600  # 超时时间（秒）

    # 执行统计
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    execution_time: float = 0.0

    def __post_init__(self):
        self.total_tests = len(self.test_cases)


__all__ = [
    'TestPriority',
    'TestCategory',
    'TestStatus',
    'TestCase',
    'TestSuite'
]

