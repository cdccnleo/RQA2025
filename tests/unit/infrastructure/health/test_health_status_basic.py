"""
HealthStatus模块基础测试套件

针对health_status.py模块的基础测试覆盖
目标: 建立基础测试框架，从0%覆盖率开始
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime

# 导入被测试模块
from src.infrastructure.health.models.health_status import (
    HealthStatus
)


class TestHealthStatusBasic:
    """HealthStatus基础测试"""

    def test_health_status_enum_values(self):
        """测试HealthStatus枚举值"""
        assert HealthStatus.UP.value == "UP"
        assert HealthStatus.DOWN.value == "DOWN"
        assert HealthStatus.DEGRADED.value == "DEGRADED"
        assert HealthStatus.UNKNOWN.value == "UNKNOWN"
        assert HealthStatus.UNHEALTHY.value == "UNHEALTHY"

    def test_from_string_valid_values(self):
        """测试from_string方法-有效值"""
        assert HealthStatus.from_string("UP") == HealthStatus.UP
        assert HealthStatus.from_string("down") == HealthStatus.DOWN
        assert HealthStatus.from_string("DeGrAdEd") == HealthStatus.DEGRADED
        assert HealthStatus.from_string("unknown") == HealthStatus.UNKNOWN
        assert HealthStatus.from_string("UNHEALTHY") == HealthStatus.UNHEALTHY

    def test_from_string_invalid_value(self):
        """测试from_string方法-无效值"""
        result = HealthStatus.from_string("INVALID")
        assert result == HealthStatus.UNKNOWN

    def test_to_string(self):
        """测试to_string方法"""
        assert HealthStatus.UP.to_string() == "UP"
        assert HealthStatus.DOWN.to_string() == "DOWN"
        assert HealthStatus.DEGRADED.to_string() == "DEGRADED"
        assert HealthStatus.UNKNOWN.to_string() == "UNKNOWN"
        assert HealthStatus.UNHEALTHY.to_string() == "UNHEALTHY"

    def test_is_healthy_method(self):
        """测试is_healthy方法"""
        assert HealthStatus.UP.is_healthy() == True
        assert HealthStatus.DOWN.is_healthy() == False
        assert HealthStatus.DEGRADED.is_healthy() == True  # DEGRADED is considered healthy
        assert HealthStatus.UNKNOWN.is_healthy() == False
        assert HealthStatus.UNHEALTHY.is_healthy() == False

    def test_is_critical_method(self):
        """测试is_critical方法"""
        assert HealthStatus.UP.is_critical() == False
        assert HealthStatus.DOWN.is_critical() == True
        assert HealthStatus.DEGRADED.is_critical() == False
        assert HealthStatus.UNKNOWN.is_critical() == False
        assert HealthStatus.UNHEALTHY.is_critical() == False

    def test_enum_members(self):
        """测试枚举成员数量"""
        members = list(HealthStatus)
        assert len(members) == 6
        assert HealthStatus.UP in members
        assert HealthStatus.DOWN in members
        assert HealthStatus.DEGRADED in members
        assert HealthStatus.UNKNOWN in members
        assert HealthStatus.UNHEALTHY in members

    def test_string_conversion_roundtrip(self):
        """测试字符串转换往返"""
        for status in HealthStatus:
            string_value = status.to_string()
            converted_back = HealthStatus.from_string(string_value)
            assert converted_back == status

    def test_case_insensitive_from_string(self):
        """测试from_string方法大小写不敏感"""
        assert HealthStatus.from_string("up") == HealthStatus.UP
        assert HealthStatus.from_string("Up") == HealthStatus.UP
        assert HealthStatus.from_string("UP") == HealthStatus.UP

    def test_empty_string_from_string(self):
        """测试from_string方法空字符串"""
        result = HealthStatus.from_string("")
        assert result == HealthStatus.UNKNOWN

    def test_none_string_from_string(self):
        """测试from_string方法None值"""
        # None值应该抛出TypeError或AttributeError
        with pytest.raises((TypeError, AttributeError)):
            HealthStatus.from_string(None)
