"""
HealthResult模块基础测试套件

针对health_result.py模块的基础测试覆盖
目标: 建立基础测试框架，从0%覆盖率开始
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime

# 导入被测试模块
from src.infrastructure.health.models.health_result import CheckType
from src.infrastructure.health.models.health_status import HealthStatus


class TestHealthResultBasic:
    """HealthResult基础测试"""

    def test_check_type_enum_values(self):
        """测试CheckType枚举值"""
        assert CheckType.BASIC.value == "basic"
        assert CheckType.DEEP.value == "deep"
        assert CheckType.PERFORMANCE.value == "performance"

    def test_check_type_from_string_valid(self):
        """测试CheckType.from_string方法-有效值"""
        assert CheckType.from_string("basic") == CheckType.BASIC
        assert CheckType.from_string("deep") == CheckType.DEEP
        assert CheckType.from_string("performance") == CheckType.PERFORMANCE
        assert CheckType.from_string("BASIC") == CheckType.BASIC  # 测试大小写

    def test_health_status_enum_values(self):
        """测试HealthStatus枚举值"""
        # 实际的HealthStatus使用大写值：UP, DOWN, DEGRADED, UNKNOWN, UNHEALTHY
        assert HealthStatus.UNKNOWN.value == "UNKNOWN"
        assert HealthStatus.UP.value == "UP"
        assert HealthStatus.DOWN.value == "DOWN"
        assert HealthStatus.DEGRADED.value == "DEGRADED"
        assert HealthStatus.UNHEALTHY.value == "UNHEALTHY"

    def test_health_status_from_string_valid(self):
        """测试HealthStatus.from_string方法-有效值"""
        # 实际的HealthStatus枚举值：UP, DOWN, DEGRADED, UNKNOWN, UNHEALTHY
        assert HealthStatus.from_string("unknown") == HealthStatus.UNKNOWN
        assert HealthStatus.from_string("up") == HealthStatus.UP
        assert HealthStatus.from_string("down") == HealthStatus.DOWN
        assert HealthStatus.from_string("degraded") == HealthStatus.DEGRADED
        assert HealthStatus.from_string("unhealthy") == HealthStatus.UNHEALTHY
        assert HealthStatus.from_string("UNKNOWN") == HealthStatus.UNKNOWN  # 测试大小写

    def test_enum_string_conversion_roundtrip(self):
        """测试枚举字符串转换往返"""
        # CheckType
        for check_type in CheckType:
            string_value = check_type.value
            converted_back = CheckType.from_string(string_value)
            assert converted_back == check_type

        # HealthStatus
        for status in HealthStatus:
            string_value = status.value
            converted_back = HealthStatus.from_string(string_value)
            assert converted_back == status
