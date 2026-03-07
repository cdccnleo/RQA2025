# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
测试配置事件模块

测试ConfigEvent和ConfigChangeEvent类的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import sys
import os
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.infrastructure.config.config_event import ConfigEvent, ConfigChangeEvent


class TestConfigEvent(unittest.TestCase):
    """测试配置事件基类"""

    def test_config_event_initialization(self):
        """测试配置事件初始化"""
        event = ConfigEvent("test_event", {"key": "value"}, "test_source")

        self.assertEqual(event.event_type, "test_event")
        self.assertEqual(event.data, {"key": "value"})
        self.assertEqual(event.source, "test_source")
        self.assertIsInstance(event.timestamp, float)
        self.assertIsInstance(event.event_id, str)
        self.assertEqual(len(event.event_id), 36)  # UUID4 length

    def test_config_event_default_values(self):
        """测试配置事件默认值"""
        event = ConfigEvent("test_event")

        self.assertEqual(event.event_type, "test_event")
        self.assertEqual(event.data, {})
        self.assertEqual(event.source, "config_system")
        self.assertIsInstance(event.timestamp, float)
        self.assertIsInstance(event.event_id, str)

    def test_config_event_to_dict(self):
        """测试配置事件转换为字典"""
        event = ConfigEvent("test_event", {"key": "value"}, "test_source")
        event_dict = event.to_dict()

        self.assertIn("event_id", event_dict)
        self.assertIn("event_type", event_dict)
        self.assertIn("data", event_dict)
        self.assertIn("source", event_dict)
        self.assertIn("timestamp", event_dict)
        self.assertIn("datetime", event_dict)

        self.assertEqual(event_dict["event_type"], "test_event")
        self.assertEqual(event_dict["data"], {"key": "value"})
        self.assertEqual(event_dict["source"], "test_source")


class TestConfigChangeEvent(unittest.TestCase):
    """测试配置变更事件"""

    def test_config_change_event_creation(self):
        """测试配置变更事件创建"""
        event = ConfigChangeEvent("test.key", "old_value", "new_value", "test_source")

        self.assertEqual(event.event_type, "config_changed")
        self.assertEqual(event.source, "test_source")
        self.assertEqual(event.data["key"], "test.key")
        self.assertEqual(event.data["old_value"], "old_value")
        self.assertEqual(event.data["new_value"], "new_value")
        self.assertIn("change_type", event.data)

    def test_config_change_event_default_source(self):
        """测试配置变更事件默认源"""
        event = ConfigChangeEvent("test.key", "old_value", "new_value")

        self.assertEqual(event.source, "config_system")

    def test_change_type_determination(self):
        """测试变更类型确定"""
        # 创建事件
        event = ConfigChangeEvent("test.key", "old", "new")

        # 修改事件以测试不同变更类型
        event.data["change_type"] = event._determine_change_type(None, "new")
        self.assertEqual(event.data["change_type"], "added")

        event.data["change_type"] = event._determine_change_type("old", None)
        self.assertEqual(event.data["change_type"], "deleted")

        event.data["change_type"] = event._determine_change_type("old", "new")
        self.assertEqual(event.data["change_type"], "modified")

        event.data["change_type"] = event._determine_change_type("same", "same")
        self.assertEqual(event.data["change_type"], "unchanged")

    def test_config_change_event_to_dict(self):
        """测试配置变更事件转换为字典"""
        event = ConfigChangeEvent("test.key", "old_value", "new_value")
        event_dict = event.to_dict()

        self.assertIn("event_id", event_dict)
        self.assertIn("event_type", event_dict)
        self.assertIn("data", event_dict)
        self.assertIn("source", event_dict)
        self.assertIn("timestamp", event_dict)
        self.assertIn("datetime", event_dict)

        self.assertEqual(event_dict["event_type"], "config_changed")
        self.assertEqual(event_dict["data"]["key"], "test.key")
        self.assertEqual(event_dict["data"]["old_value"], "old_value")
        self.assertEqual(event_dict["data"]["new_value"], "new_value")
        self.assertIn("change_type", event_dict["data"])

    def test_config_change_event_with_none_values(self):
        """测试配置变更事件处理None值"""
        event = ConfigChangeEvent("test.key", None, None)

        self.assertEqual(event.data["old_value"], None)
        self.assertEqual(event.data["new_value"], None)
        self.assertEqual(event.data["change_type"], "unchanged")

    def test_config_change_event_complex_values(self):
        """测试配置变更事件处理复杂值"""
        old_value = {"nested": {"key": "old"}}
        new_value = {"nested": {"key": "new"}}
        event = ConfigChangeEvent("test.key", old_value, new_value)

        self.assertEqual(event.data["old_value"], old_value)
        self.assertEqual(event.data["new_value"], new_value)
        self.assertEqual(event.data["change_type"], "modified")


if __name__ == '__main__':
    unittest.main()
