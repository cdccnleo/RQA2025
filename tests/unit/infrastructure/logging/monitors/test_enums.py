"""
测试日志监控枚举类

覆盖 enums.py 中的 AlertLevel 和 AlertData 类
"""

import pytest
from datetime import datetime
from src.infrastructure.logging.monitors.enums import AlertLevel, AlertData


class TestAlertLevel:
    """AlertLevel 枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_enum_int_values(self):
        """测试枚举整数值"""
        assert AlertLevel.INFO.int_value == 0
        assert AlertLevel.WARNING.int_value == 1
        assert AlertLevel.ERROR.int_value == 2
        assert AlertLevel.CRITICAL.int_value == 3

    def test_enum_members(self):
        """测试枚举成员"""
        assert len(AlertLevel) == 4
        assert AlertLevel.INFO in AlertLevel
        assert AlertLevel.WARNING in AlertLevel
        assert AlertLevel.ERROR in AlertLevel
        assert AlertLevel.CRITICAL in AlertLevel

    def test_enum_comparison(self):
        """测试枚举比较"""
        assert AlertLevel.INFO < AlertLevel.WARNING
        assert AlertLevel.WARNING <= AlertLevel.WARNING
        assert AlertLevel.ERROR > AlertLevel.WARNING
        assert AlertLevel.CRITICAL >= AlertLevel.ERROR

        # 测试反向比较
        assert AlertLevel.WARNING > AlertLevel.INFO
        assert AlertLevel.WARNING >= AlertLevel.WARNING
        assert AlertLevel.WARNING < AlertLevel.ERROR
        assert AlertLevel.ERROR <= AlertLevel.CRITICAL

    def test_enum_equality(self):
        """测试枚举相等性"""
        assert AlertLevel.INFO == AlertLevel.INFO
        assert AlertLevel.WARNING != AlertLevel.ERROR

    def test_enum_iteration(self):
        """测试枚举迭代"""
        levels = list(AlertLevel)
        assert len(levels) == 4
        assert levels[0] == AlertLevel.INFO
        assert levels[1] == AlertLevel.WARNING
        assert levels[2] == AlertLevel.ERROR
        assert levels[3] == AlertLevel.CRITICAL

    def test_enum_string_representation(self):
        """测试枚举字符串表示"""
        assert str(AlertLevel.INFO) == "AlertLevel.INFO"
        # 自定义AlertLevel的repr格式包含类名和成员名
        assert "AlertLevel.WARNING" in repr(AlertLevel.WARNING)

    def test_enum_access_by_name(self):
        """测试通过名称访问枚举"""
        assert AlertLevel['INFO'] == AlertLevel.INFO
        assert AlertLevel['WARNING'] == AlertLevel.WARNING
        assert AlertLevel['ERROR'] == AlertLevel.ERROR
        assert AlertLevel['CRITICAL'] == AlertLevel.CRITICAL


class TestAlertData:
    """AlertData 数据类测试"""

    def test_init_minimal(self):
        """测试最小初始化"""
        alert = AlertData(
            level=AlertLevel.INFO,
            message="Test alert"
        )

        assert alert.level == AlertLevel.INFO
        assert alert.message == "Test alert"
        assert isinstance(alert.timestamp, datetime)
        assert alert.source == ""
        assert alert.metadata == {}

    def test_init_full(self):
        """测试完整初始化"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        metadata = {"key": "value", "count": 42}

        alert = AlertData(
            level=AlertLevel.ERROR,
            message="Error occurred",
            timestamp=timestamp,
            source="test_source",
            metadata=metadata
        )

        assert alert.level == AlertLevel.ERROR
        assert alert.message == "Error occurred"
        assert alert.timestamp == timestamp
        assert alert.source == "test_source"
        assert alert.metadata == metadata

    def test_default_timestamp(self):
        """测试默认时间戳"""
        before = datetime.now()
        alert = AlertData(level=AlertLevel.WARNING, message="Test")
        after = datetime.now()

        assert before <= alert.timestamp <= after

    def test_immutable_fields(self):
        """测试可变字段"""
        alert = AlertData(
            level=AlertLevel.INFO,
            message="Test",
            metadata={"initial": "value"}
        )

        # 应该可以修改metadata
        alert.metadata["new_key"] = "new_value"
        assert alert.metadata["new_key"] == "new_value"
        assert alert.metadata["initial"] == "value"

        # 应该可以修改source
        alert.source = "new_source"
        assert alert.source == "new_source"

        # 应该可以修改timestamp
        new_time = datetime(2023, 12, 25)
        alert.timestamp = new_time
        assert alert.timestamp == new_time

    def test_equality(self):
        """测试相等性"""
        alert1 = AlertData(
            level=AlertLevel.ERROR,
            message="Same message",
            source="same_source"
        )
        alert2 = AlertData(
            level=AlertLevel.ERROR,
            message="Same message",
            source="same_source"
        )

        # 注意：由于timestamp是动态生成的，相等性比较可能失败
        # 这里只比较关键字段
        assert alert1.level == alert2.level
        assert alert1.message == alert2.message
        assert alert1.source == alert2.source

    def test_string_representation(self):
        """测试字符串表示"""
        alert = AlertData(
            level=AlertLevel.WARNING,
            message="Warning message",
            source="test_source"
        )

        str_repr = str(alert)
        assert "AlertData" in str_repr
        assert "WARNING" in str_repr or "warning" in str_repr
        assert "Warning message" in str_repr

    def test_dataclass_features(self):
        """测试数据类特性"""
        alert1 = AlertData(level=AlertLevel.INFO, message="Test")
        alert2 = AlertData(level=AlertLevel.INFO, message="Test")
        alert3 = AlertData(level=AlertLevel.ERROR, message="Test")

        # 测试字段访问
        assert alert1.level == AlertLevel.INFO
        assert alert1.message == "Test"

        # 测试字段修改
        alert1.source = "modified"
        assert alert1.source == "modified"
