"""
测试版本管理器
"""
import unittest
import os
import shutil
import pandas as pd
from pathlib import Path
import tempfile

from src.data.version_control.version_manager import DataVersionManager
from src.data.data_manager import DataModel
from src.infrastructure.utils.exceptions import DataVersionError


class TestDataVersionManager(unittest.TestCase):
    """测试版本管理器"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.version_dir = os.path.join(self.temp_dir, 'versions')

        # 创建版本管理器
        self.version_manager = DataVersionManager(self.version_dir)

        # 创建测试数据
        self.test_data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        self.test_metadata1 = {
            'source': 'test',
            'created_at': '2023-01-01'
        }
        self.test_model1 = DataModel(self.test_data1, self.test_metadata1)

        # 创建第二个测试数据（用于比较）
        self.test_data2 = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 30, 35, 40],
            'gender': ['F', 'M', 'M', 'M']
        })
        self.test_metadata2 = {
            'source': 'test_updated',
            'created_at': '2023-01-02',
            'updated_by': 'tester'
        }
        self.test_model2 = DataModel(self.test_data2, self.test_metadata2)

    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)

    def test_create_version(self):
        """测试创建版本"""
        # 创建版本
        version = self.version_manager.create_version(
            self.test_model1,
            description="Test version",
            tags=["test"],
            creator="tester"
        )

        # 验证版本是否创建成功
        self.assertIsNotNone(version)
        self.assertEqual(self.version_manager.current_version, version)

        # 验证版本文件是否存在
        version_file = Path(self.version_dir) / f"{version}.parquet"
        self.assertTrue(version_file.exists())

        # 验证版本信息
        version_info = self.version_manager.get_version_info(version)
        self.assertIsNotNone(version_info)
        self.assertEqual(version_info['description'], "Test version")
        self.assertEqual(version_info['tags'], ["test"])
        self.assertEqual(version_info['creator'], "tester")

    def test_get_version(self):
        """测试获取版本"""
        # 创建版本
        version = self.version_manager.create_version(
            self.test_model1,
            description="Test version",
            tags=["test"],
            creator="tester"
        )

        # 获取版本
        model = self.version_manager.get_version(version)

        # 验证数据
        self.assertIsNotNone(model)
        pd.testing.assert_frame_equal(model.data, self.test_data1)
        self.assertEqual(model.get_metadata(), self.test_metadata1)

    def test_list_versions(self):
        """测试列出版本"""
        # 创建多个版本
        version1 = self.version_manager.create_version(
            self.test_model1,
            description="Test version 1",
            tags=["test", "v1"],
            creator="tester1"
        )

        version2 = self.version_manager.create_version(
            self.test_model2,
            description="Test version 2",
            tags=["test", "v2"],
            creator="tester2"
        )

        # 列出所有版本
        versions = self.version_manager.list_versions()
        self.assertEqual(len(versions), 2)

        # 按标签筛选
        versions = self.version_manager.list_versions(tags=["v1"])
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]['version_id'], version1)

        # 按创建者筛选
        versions = self.version_manager.list_versions(creator="tester2")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]['version_id'], version2)

    def test_delete_version(self):
        """测试删除版本"""
        # 创建多个版本
        version1 = self.version_manager.create_version(
            self.test_model1,
            description="Test version 1",
            tags=["test", "v1"],
            creator="tester1"
        )

        version2 = self.version_manager.create_version(
            self.test_model2,
            description="Test version 2",
            tags=["test", "v2"],
            creator="tester2"
        )

        # 删除第一个版本
        result = self.version_manager.delete_version(version1)
        self.assertTrue(result)

        # 验证版本是否已删除
        self.assertIsNone(self.version_manager.get_version(version1))

        # 验证版本文件是否已删除
        version_file = Path(self.version_dir) / f"{version1}.parquet"
        self.assertFalse(version_file.exists())

        # 验证元数据是否已更新
        self.assertNotIn(version1, self.version_manager.metadata['versions'])

        # 尝试删除当前版本（应该失败）
        with self.assertRaises(DataVersionError):
            self.version_manager.delete_version(version2)

    def test_rollback(self):
        """测试回滚版本"""
        # 创建多个版本
        version1 = self.version_manager.create_version(
            self.test_model1,
            description="Test version 1",
            tags=["test", "v1"],
            creator="tester1"
        )

        version2 = self.version_manager.create_version(
            self.test_model2,
            description="Test version 2",
            tags=["test", "v2"],
            creator="tester2"
        )

        # 回滚到第一个版本
        new_version = self.version_manager.rollback(version1)

        # 验证回滚是否成功
        self.assertIsNotNone(new_version)
        self.assertNotEqual(new_version, version1)
        self.assertNotEqual(new_version, version2)

        # 验证回滚后的数据
        model = self.version_manager.get_version(new_version)
        pd.testing.assert_frame_equal(model.data, self.test_data1)
        self.assertEqual(model.get_metadata(), self.test_metadata1)

        # 验证版本信息
        version_info = self.version_manager.get_version_info(new_version)
        self.assertIn('rollback', version_info['tags'])
        self.assertIn(f'from_{version1}', version_info['tags'])

    def test_compare_versions(self):
        """测试比较版本"""
        # 创建多个版本
        version1 = self.version_manager.create_version(
            self.test_model1,
            description="Test version 1",
            tags=["test", "v1"],
            creator="tester1"
        )

        version2 = self.version_manager.create_version(
            self.test_model2,
            description="Test version 2",
            tags=["test", "v2"],
            creator="tester2"
        )

        # 比较版本
        diff = self.version_manager.compare_versions(version1, version2)

        # 验证元数据差异
        self.assertIn('metadata_diff', diff)
        self.assertIn('added', diff['metadata_diff'])
        self.assertIn('updated_by', diff['metadata_diff']['added'])
        self.assertIn('changed', diff['metadata_diff'])
        self.assertIn('source', diff['metadata_diff']['changed'])

        # 验证数据差异
        self.assertIn('data_diff', diff)
        self.assertIn('shape_diff', diff['data_diff'])
        self.assertEqual(diff['data_diff']['shape_diff']['rows'], 1)  # 增加了1行
        self.assertEqual(diff['data_diff']['shape_diff']['columns'], 1)  # 增加了1列

        self.assertIn('columns_diff', diff['data_diff'])
        self.assertEqual(diff['data_diff']['columns_diff']['added'], ['gender'])
        self.assertEqual(diff['data_diff']['columns_diff']['removed'], [])

        self.assertIn('value_diff', diff['data_diff'])
        # 没有共同列的值变化，因为我们的测试数据没有修改现有行


if __name__ == '__main__':
    unittest.main()
