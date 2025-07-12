import unittest
import time
from src.infrastructure.versioning.data_version_manager import DataVersionManager

class TestDataVersionManager(unittest.TestCase):
    def setUp(self):
        self.dvm = DataVersionManager(max_snapshots=5, max_changelog=10)

    def test_basic_operations(self):
        # 测试基本快照功能
        data1 = {"key": "value1"}
        ts1 = self.dvm.take_snapshot(data1)
        self.assertIsNotNone(ts1)

        # 测试获取版本
        retrieved = self.dvm.get_version(ts1)
        self.assertEqual(retrieved["key"], "value1")

        # 测试变更记录
        self.dvm.record_change("update", key="key", value="value2")
        changes = self.dvm.get_changelog()
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["operation"], "update")

    def test_snapshot_cleanup(self):
        # 测试快照清理
        for i in range(10):
            self.dvm.take_snapshot({"data": i})
            time.sleep(0.01)

        self.assertLessEqual(len(self.dvm._snapshots), 5)

    def test_thread_safety(self):
        from threading import Thread
        import random

        results = []

        def worker():
            data = {"value": random.random()}
            ts = self.dvm.take_snapshot(data)
            results.append(ts is not None)

        threads = [Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(all(results))
        self.assertEqual(len(self.dvm._snapshots), 5)

if __name__ == "__main__":
    unittest.main()
