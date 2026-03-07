"""
系统集成测试枚举定义

Enums for system integration testing module.

Extracted from system_integration_tester.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from enum import Enum

class IntegrationTestType(Enum):
    """集成测试类型枚举"""
    UNIT_TEST = "unit_test"                    # 单元测试
    COMPONENT_TEST = "component_test"          # 组件测试
    INTEGRATION_TEST = "integration_test"      # 集成测试
    END_TO_END_TEST = "end_to_end_test"        # 端到端测试
    PERFORMANCE_TEST = "performance_test"      # 性能测试
    STRESS_TEST = "stress_test"                # 压力测试
    LOAD_TEST = "load_test"                    # 负载测试
    SECURITY_TEST = "security_test"            # 安全测试
    COMPLIANCE_TEST = "compliance_test"        # 合规测试


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"        # 待执行
    RUNNING = "running"        # 运行中
    PASSED = "passed"          # 通过
    FAILED = "failed"          # 失败
    SKIPPED = "skipped"        # 跳过
    ERROR = "error"            # 错误


class ComponentStatus(Enum):
    """组件状态枚举"""
    HEALTHY = "healthy"        # 健康
    DEGRADED = "degraded"      # 降级
    UNHEALTHY = "unhealthy"    # 不健康
    OFFLINE = "offline"        # 离线


__all__ = ['IntegrationTestType', 'TestStatus', 'ComponentStatus']
