import unittest
import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from src.infrastructure.versioning.storage_adapter import VersionedStorageAdapter

class TestVersionedStorageAdapter(unittest.TestCase):
    def setUp(self):
        # 创建模拟基础存储
        self.mock_storage = MagicMock()
        self.mock_storage.get.return_value = None
        self.mock_storage.get_all.return_value = {}

        # 创建带版本控制的适配器
        self.adapter = VersionedStorageAdapter(self.mock_storage)

    def test_basic_operations(self):
        # 测试基本设置和获取
        ts = self.adapter.set("test_key", "test_value")
        self.assertIsNotNone(ts)

        # 验证基础存储被调用
        self.mock_storage.set.assert_called_with("test_key", "test_value")

        # 测试获取最新值
        self.mock_storage.get.return_value = "test_value"
        self.assertEqual(self.adapter.get("test_key"), "test_value")

    def test_versioned_get(self):
        # 准备测试数据
        self.mock_storage.get_all.return_value = {"v1": "value1"}
        ts1 = self.adapter.set("test_key", "value1")

        time.sleep(0.1)
        self.mock_storage.get_all.return_value = {"v1": "value1", "v2": "value2"}
        ts2 = self.adapter.set("test_key", "value2")

        # 测试获取历史版本
        self.assertEqual(
            self.adapter.get("test_key", version=ts1)["v1"],
            "value1"
        )

    def test_history_and_rollback(self):
        # 记录变更历史
        self.mock_storage.get_all.return_value = {}
        ts1 = self.adapter.set("key1", "value1")

        time.sleep(0.1)
        self.mock_storage.get_all.return_value = {"key1": "value1"}
        ts2 = self.adapter.delete("key1")

        # 测试获取变更历史
        history = self.adapter.get_history("key1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["operation"], "set")
        self.assertEqual(history[1]["operation"], "delete")

        # 测试回滚
        self.assertTrue(self.adapter.rollback(ts1))
        self.mock_storage.set.assert_called_with("key1", "value1")

if __name__ == "__main__":
    unittest.main()
