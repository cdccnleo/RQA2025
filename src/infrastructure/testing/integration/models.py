"""
系统集成测试数据模型

Data models for system integration testing.

Extracted from system_integration_tester.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .enums import IntegrationTestType, TestStatus, ComponentStatus


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    test_name: str
    test_type: IntegrationTestType
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    result_details: Dict[str, Any]
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'result_details': self.result_details,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics
        }


@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_name: str
    status: ComponentStatus
    last_check: datetime
    response_time: float
    error_count: int
    throughput: Optional[float]
    memory_usage: Optional[float]
    cpu_usage: Optional[float]
    custom_metrics: Optional[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'component_name': self.component_name,
            'status': self.status.value,
            'last_check': self.last_check.isoformat(),
            'response_time': self.response_time,
            'error_count': self.error_count,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'custom_metrics': self.custom_metrics
        }


__all__ = ['TestResult', 'ComponentHealth']
