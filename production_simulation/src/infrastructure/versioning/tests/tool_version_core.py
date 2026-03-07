"""
版本管理核心模块测试

测试版本类和比较器的功能。
"""

import unittest
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ...core.version import Version, VersionComparator


class TestVersion(unittest.TestCase):
    """版本类测试"""

    def test_version_creation(self):
        """测试版本创建"""
        v = Version(1, 2, 3)
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 2)
        self.assertEqual(v.patch, 3)
        self.assertIsNone(v.prerelease)
        self.assertIsNone(v.build)

    def test_version_from_string(self):
        """测试从字符串创建版本"""
        v = Version("1.2.3")
        self.assertEqual(str(v), "1.2.3")

        v = Version("2.0.0-alpha.1")
        self.assertEqual(str(v), "2.0.0-alpha.1")

        v = Version("1.0.0+build.123")
        self.assertEqual(str(v), "1.0.0+build.123")

    def test_version_comparison(self):
        """测试版本比较"""
        v1 = Version("1.0.0")
        v2 = Version("1.0.1")
        v3 = Version("1.1.0")

        self.assertTrue(v1 < v2)
        self.assertTrue(v2 < v3)
        self.assertTrue(v3 > v1)

        self.assertEqual(v1, Version("1.0.0"))

    def test_prerelease_handling(self):
        """测试预发布版本处理"""
        stable = Version("1.0.0")
        prerelease = Version("1.0.0-alpha.1")

        self.assertTrue(stable.is_stable())
        self.assertFalse(prerelease.is_stable())
        self.assertTrue(prerelease.is_prerelease())

    def test_version_increment(self):
        """测试版本递增"""
        v = Version("1.2.3")
        self.assertEqual(str(v.increment_patch()), "1.2.4")
        self.assertEqual(str(v.increment_minor()), "1.3.0")
        self.assertEqual(str(v.increment_major()), "2.0.0")


class TestVersionComparator(unittest.TestCase):
    """版本比较器测试"""

    def test_version_comparison(self):
        """测试版本比较功能"""
        result = VersionComparator.compare_versions("1.0.0", "1.0.1")
        self.assertEqual(result, -1)  # 1.0.0 < 1.0.1

        result = VersionComparator.compare_versions("1.0.1", "1.0.0")
        self.assertEqual(result, 1)   # 1.0.1 > 1.0.0

        result = VersionComparator.compare_versions("1.0.0", "1.0.0")
        self.assertEqual(result, 0)   # 1.0.0 == 1.0.0

    def test_range_matching(self):
        """测试范围匹配"""
        # 大于等于
        self.assertTrue(VersionComparator.is_version_in_range("2.0.0", ">=1.0.0"))
        self.assertFalse(VersionComparator.is_version_in_range("0.5.0", ">=1.0.0"))

        # 兼容版本 (^)
        self.assertTrue(VersionComparator.is_version_in_range("1.5.0", "^1.0.0"))
        self.assertFalse(VersionComparator.is_version_in_range("2.0.0", "^1.0.0"))

        # 约等于 (~)
        self.assertTrue(VersionComparator.is_version_in_range("1.0.5", "~1.0.0"))
        self.assertFalse(VersionComparator.is_version_in_range("1.1.0", "~1.0.0"))

    def test_find_latest(self):
        """测试查找最新版本"""
        versions = ["1.0.0", "1.1.0", "1.0.5", "2.0.0"]
        latest = VersionComparator.find_latest_version(versions)
        self.assertEqual(str(latest), "2.0.0")

    def test_sort_versions(self):
        """测试版本排序"""
        versions = ["1.1.0", "1.0.0", "1.0.5", "2.0.0"]
        sorted_versions = VersionComparator.sort_versions(versions)
        expected = ["1.0.0", "1.0.5", "1.1.0", "2.0.0"]
        self.assertEqual([str(v) for v in sorted_versions], expected)


if __name__ == '__main__':
    unittest.main()
