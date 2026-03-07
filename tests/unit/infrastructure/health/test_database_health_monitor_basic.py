"""
DatabaseHealthMonitor基础测试套件

针对database_health_monitor.py模块的基础测试覆盖
目标: 建立基础测试框架，从16.43%覆盖率开始
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest

# 只导入常量和枚举，避免导入可能导致死锁的类
from src.infrastructure.health.database.database_health_monitor import (
    HealthStatus,
    DEFAULT_CHECK_INTERVAL,
    WARNING_CONNECTION_COUNT,
    CRITICAL_CONNECTION_COUNT
)


class TestDatabaseHealthMonitorBasic:
    """DatabaseHealthMonitor基础测试"""


    def test_constants(self):
        """测试常量定义"""
        assert DEFAULT_CHECK_INTERVAL == 60
        assert WARNING_CONNECTION_COUNT == 80
        assert CRITICAL_CONNECTION_COUNT == 95

    def test_enums(self):
        """测试枚举定义"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_constants_ranges(self):
        """测试常量范围合理性"""
        # 检查警告阈值在合理范围内
        assert 0 < WARNING_CONNECTION_COUNT < CRITICAL_CONNECTION_COUNT
        assert 0 < DEFAULT_CHECK_INTERVAL < 3600  # 1小时内

        # 检查错误率阈值合理
        from src.infrastructure.health.database.database_health_monitor import WARNING_ERROR_RATE, CRITICAL_ERROR_RATE
        assert 0 < WARNING_ERROR_RATE < CRITICAL_ERROR_RATE < 1

    def test_import_safety(self):
        """测试安全导入"""
        # 验证能安全导入而不导致死锁
        try:
            from src.infrastructure.health.database.database_health_monitor import (
                DEFAULT_CHECK_INTERVAL,
                WARNING_CONNECTION_COUNT,
                CRITICAL_CONNECTION_COUNT,
                HealthStatus
            )
            assert DEFAULT_CHECK_INTERVAL == 60
            assert WARNING_CONNECTION_COUNT == 80
            assert CRITICAL_CONNECTION_COUNT == 95
            assert HealthStatus.HEALTHY.value == "healthy"
        except ImportError:
            pytest.fail("安全导入失败")

    def test_module_constants_consistency(self):
        """测试模块常量一致性"""
        # 验证所有相关的常量都能正确导入
        from src.infrastructure.health.database.database_health_monitor import (
            ERROR_RETRY_DELAY,
            WARNING_MEMORY_USAGE,
            WARNING_CPU_USAGE,
            WARNING_DISK_USAGE,
            CRITICAL_MEMORY_USAGE,
            CRITICAL_CPU_USAGE,
            CRITICAL_DISK_USAGE
        )

        # 验证重试延迟合理
        assert 0 < ERROR_RETRY_DELAY < DEFAULT_CHECK_INTERVAL

        # 验证资源使用率阈值在合理范围内
        assert 0 < WARNING_MEMORY_USAGE < CRITICAL_MEMORY_USAGE <= 1
        assert 0 < WARNING_CPU_USAGE < CRITICAL_CPU_USAGE <= 1
        assert 0 < WARNING_DISK_USAGE < CRITICAL_DISK_USAGE <= 1
