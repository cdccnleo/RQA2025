#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征版本管理测试用例

测试范围：
1. 特征版本创建
2. 特征版本切换
3. 特征版本回滚
4. 特征版本删除
5. 特征版本兼容性检查
"""

import pytest
import unittest
from datetime import datetime
from typing import Dict, Any, List
import json
import tempfile
import os

# 导入被测试模块
from src.features.core.engine import FeatureEngine
from src.features.core.config import FeatureConfig


class TestFeatureVersionManagement(unittest.TestCase):
    """特征版本管理测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = FeatureConfig()
        self.engine = FeatureEngine(self.config)
        
        # 创建测试用的临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.config.data_dir = self.temp_dir
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    # ==================== 测试用例 1: 特征版本创建 ====================
    
    def test_create_feature_version_success(self):
        """测试成功创建特征版本"""
        # 准备测试数据
        version_name = "v1.0.0"
        features = ["SMA", "EMA", "RSI"]
        description = "测试版本"
        
        # 执行创建操作
        result = self.engine.create_feature_version(
            version_name=version_name,
            features=features,
            description=description
        )
        
        # 验证结果
        self.assertTrue(result['success'])
        self.assertEqual(result['version'], version_name)
        self.assertIn('created_at', result)
        self.assertEqual(result['feature_count'], len(features))
    
    def test_create_feature_version_duplicate(self):
        """测试创建重复版本号应失败"""
        version_name = "v1.0.0"
        features = ["SMA", "EMA"]
        
        # 第一次创建
        result1 = self.engine.create_feature_version(
            version_name=version_name,
            features=features
        )
        self.assertTrue(result1['success'])
        
        # 第二次创建相同版本应失败
        result2 = self.engine.create_feature_version(
            version_name=version_name,
            features=features
        )
        self.assertFalse(result2['success'])
        self.assertIn('error', result2)
    
    def test_create_feature_version_invalid_name(self):
        """测试创建无效版本名称应失败"""
        invalid_names = ["", "   ", "v 1.0", "version@1"]
        features = ["SMA"]
        
        for name in invalid_names:
            result = self.engine.create_feature_version(
                version_name=name,
                features=features
            )
            self.assertFalse(result['success'], f"版本名 '{name}' 应该无效")
    
    def test_create_feature_version_empty_features(self):
        """测试创建空特征列表应失败"""
        result = self.engine.create_feature_version(
            version_name="v1.0.0",
            features=[]
        )
        self.assertFalse(result['success'])
    
    # ==================== 测试用例 2: 特征版本切换 ====================
    
    def test_switch_feature_version_success(self):
        """测试成功切换特征版本"""
        # 创建两个版本
        self.engine.create_feature_version("v1.0.0", ["SMA", "EMA"], "版本1")
        self.engine.create_feature_version("v2.0.0", ["SMA", "EMA", "RSI"], "版本2")
        
        # 切换到 v2.0.0
        result = self.engine.switch_feature_version("v2.0.0")
        
        # 验证结果
        self.assertTrue(result['success'])
        self.assertEqual(self.engine.current_version, "v2.0.0")
        self.assertEqual(len(self.engine.active_features), 3)
    
    def test_switch_feature_version_not_exist(self):
        """测试切换到不存在的版本应失败"""
        result = self.engine.switch_feature_version("v999.0.0")
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_switch_feature_version_same_version(self):
        """测试切换到相同版本应提示"""
        self.engine.create_feature_version("v1.0.0", ["SMA"])
        
        result = self.engine.switch_feature_version("v1.0.0")
        
        self.assertTrue(result['success'])
        self.assertIn('message', result)
        self.assertIn('already active', result['message'].lower())
    
    # ==================== 测试用例 3: 特征版本回滚 ====================
    
    def test_rollback_feature_version_success(self):
        """测试成功回滚特征版本"""
        # 创建历史版本
        self.engine.create_feature_version("v1.0.0", ["SMA"], "初始版本")
        self.engine.create_feature_version("v2.0.0", ["SMA", "EMA"], "更新版本")
        
        # 当前版本是 v2.0.0
        self.assertEqual(self.engine.current_version, "v2.0.0")
        
        # 回滚到 v1.0.0
        result = self.engine.rollback_feature_version("v1.0.0")
        
        # 验证结果
        self.assertTrue(result['success'])
        self.assertEqual(self.engine.current_version, "v1.0.0")
        self.assertEqual(len(self.engine.active_features), 1)
    
    def test_rollback_feature_version_with_backup(self):
        """测试回滚时创建当前版本备份"""
        self.engine.create_feature_version("v1.0.0", ["SMA"])
        self.engine.create_feature_version("v2.0.0", ["SMA", "EMA"])
        
        # 回滚并创建备份
        result = self.engine.rollback_feature_version("v1.0.0", create_backup=True)
        
        self.assertTrue(result['success'])
        # 验证备份已创建
        self.assertIn("backup_version", result)
    
    def test_rollback_feature_version_not_exist(self):
        """测试回滚到不存在的版本应失败"""
        result = self.engine.rollback_feature_version("v0.0.0")
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    # ==================== 测试用例 4: 特征版本删除 ====================
    
    def test_delete_feature_version_success(self):
        """测试成功删除特征版本"""
        # 创建版本
        self.engine.create_feature_version("v1.0.0", ["SMA"])
        
        # 删除版本
        result = self.engine.delete_feature_version("v1.0.0")
        
        # 验证结果
        self.assertTrue(result['success'])
        
        # 验证版本已删除
        versions = self.engine.list_feature_versions()
        version_names = [v['version'] for v in versions]
        self.assertNotIn("v1.0.0", version_names)
    
    def test_delete_active_feature_version(self):
        """测试删除当前活跃版本应失败或警告"""
        self.engine.create_feature_version("v1.0.0", ["SMA"])
        
        result = self.engine.delete_feature_version("v1.0.0")
        
        # 应该失败或需要确认
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_delete_nonexistent_feature_version(self):
        """测试删除不存在的版本应失败"""
        result = self.engine.delete_feature_version("v999.0.0")
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    # ==================== 测试用例 5: 特征版本兼容性检查 ====================
    
    def test_check_version_compatibility_compatible(self):
        """测试检查兼容的版本"""
        self.engine.create_feature_version("v1.0.0", ["SMA", "EMA"])
        self.engine.create_feature_version("v1.1.0", ["SMA", "EMA", "RSI"])
        
        result = self.engine.check_version_compatibility("v1.0.0", "v1.1.0")
        
        self.assertTrue(result['compatible'])
        self.assertEqual(result['common_features'], ["SMA", "EMA"])
    
    def test_check_version_compatibility_incompatible(self):
        """测试检查不兼容的版本"""
        self.engine.create_feature_version("v1.0.0", ["SMA", "EMA"])
        self.engine.create_feature_version("v2.0.0", ["RSI", "MACD"])
        
        result = self.engine.check_version_compatibility("v1.0.0", "v2.0.0")
        
        self.assertFalse(result['compatible'])
        self.assertEqual(len(result['common_features']), 0)
    
    def test_check_version_compatibility_same_version(self):
        """测试检查相同版本"""
        self.engine.create_feature_version("v1.0.0", ["SMA", "EMA"])
        
        result = self.engine.check_version_compatibility("v1.0.0", "v1.0.0")
        
        self.assertTrue(result['compatible'])
        self.assertEqual(result['common_features'], ["SMA", "EMA"])
    
    # ==================== 测试用例 6: 版本列表和查询 ====================
    
    def test_list_feature_versions(self):
        """测试列出所有特征版本"""
        # 创建多个版本
        self.engine.create_feature_version("v1.0.0", ["SMA"], "版本1")
        self.engine.create_feature_version("v1.1.0", ["SMA", "EMA"], "版本2")
        self.engine.create_feature_version("v2.0.0", ["SMA", "EMA", "RSI"], "版本3")
        
        versions = self.engine.list_feature_versions()
        
        self.assertEqual(len(versions), 3)
        
        # 验证版本信息完整性
        for version in versions:
            self.assertIn('version', version)
            self.assertIn('feature_count', version)
            self.assertIn('created_at', version)
            self.assertIn('description', version)
    
    def test_get_feature_version_details(self):
        """测试获取特征版本详细信息"""
        self.engine.create_feature_version(
            version_name="v1.0.0",
            features=["SMA", "EMA", "RSI"],
            description="测试版本"
        )
        
        details = self.engine.get_feature_version_details("v1.0.0")
        
        self.assertIsNotNone(details)
        self.assertEqual(details['version'], "v1.0.0")
        self.assertEqual(details['features'], ["SMA", "EMA", "RSI"])
        self.assertEqual(details['description'], "测试版本")
    
    def test_get_feature_version_details_not_exist(self):
        """测试获取不存在的版本详情"""
        details = self.engine.get_feature_version_details("v999.0.0")
        
        self.assertIsNone(details)
    
    # ==================== 测试用例 7: 版本比较 ====================
    
    def test_compare_feature_versions(self):
        """测试比较两个特征版本"""
        self.engine.create_feature_version("v1.0.0", ["SMA", "EMA"])
        self.engine.create_feature_version("v2.0.0", ["SMA", "EMA", "RSI", "MACD"])
        
        comparison = self.engine.compare_feature_versions("v1.0.0", "v2.0.0")
        
        self.assertIn('added_features', comparison)
        self.assertIn('removed_features', comparison)
        self.assertIn('common_features', comparison)
        
        self.assertEqual(comparison['added_features'], ["RSI", "MACD"])
        self.assertEqual(comparison['removed_features'], [])
        self.assertEqual(comparison['common_features'], ["SMA", "EMA"])
    
    # ==================== 测试用例 8: 版本导出导入 ====================
    
    def test_export_feature_version(self):
        """测试导出特征版本"""
        self.engine.create_feature_version("v1.0.0", ["SMA", "EMA"], "测试版本")
        
        export_data = self.engine.export_feature_version("v1.0.0")
        
        self.assertIsNotNone(export_data)
        self.assertEqual(export_data['version'], "v1.0.0")
        self.assertEqual(export_data['features'], ["SMA", "EMA"])
        self.assertIn('export_timestamp', export_data)
    
    def test_import_feature_version(self):
        """测试导入特征版本"""
        # 准备导入数据
        import_data = {
            'version': 'v1.0.0',
            'features': ['SMA', 'EMA', 'RSI'],
            'description': '导入的测试版本',
            'created_at': datetime.now().isoformat()
        }
        
        result = self.engine.import_feature_version(import_data)
        
        self.assertTrue(result['success'])
        
        # 验证导入成功
        versions = self.engine.list_feature_versions()
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]['version'], 'v1.0.0')
    
    def test_import_duplicate_version(self):
        """测试导入重复版本应失败"""
        # 先创建版本
        self.engine.create_feature_version("v1.0.0", ["SMA"])
        
        # 尝试导入相同版本
        import_data = {
            'version': 'v1.0.0',
            'features': ['SMA', 'EMA'],
            'description': '导入版本'
        }
        
        result = self.engine.import_feature_version(import_data)
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)


class TestFeatureVersionManagementIntegration(unittest.TestCase):
    """特征版本管理集成测试类"""
    
    def setUp(self):
        """集成测试准备"""
        self.config = FeatureConfig()
        self.engine = FeatureEngine(self.config)
        self.temp_dir = tempfile.mkdtemp()
        self.config.data_dir = self.temp_dir
    
    def tearDown(self):
        """集成测试清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_version_workflow(self):
        """测试完整的版本管理工作流"""
        # 1. 创建初始版本
        result1 = self.engine.create_feature_version(
            "v1.0.0",
            ["SMA", "EMA"],
            "初始版本"
        )
        self.assertTrue(result1['success'])
        
        # 2. 创建更新版本
        result2 = self.engine.create_feature_version(
            "v2.0.0",
            ["SMA", "EMA", "RSI", "MACD"],
            "增加技术指标"
        )
        self.assertTrue(result2['success'])
        
        # 3. 切换到新版本
        result3 = self.engine.switch_feature_version("v2.0.0")
        self.assertTrue(result3['success'])
        self.assertEqual(self.engine.current_version, "v2.0.0")
        
        # 4. 比较版本差异
        comparison = self.engine.compare_feature_versions("v1.0.0", "v2.0.0")
        self.assertEqual(len(comparison['added_features']), 2)
        
        # 5. 回滚到旧版本
        result4 = self.engine.rollback_feature_version("v1.0.0", create_backup=True)
        self.assertTrue(result4['success'])
        self.assertEqual(self.engine.current_version, "v1.0.0")
        
        # 6. 验证版本列表
        versions = self.engine.list_feature_versions()
        self.assertEqual(len(versions), 3)  # v1.0.0, v2.0.0, backup
        
        # 7. 导出版本
        export_data = self.engine.export_feature_version("v1.0.0")
        self.assertIsNotNone(export_data)
        
        # 8. 删除版本
        result5 = self.engine.delete_feature_version("v2.0.0")
        self.assertTrue(result5['success'])
        
        versions = self.engine.list_feature_versions()
        self.assertEqual(len(versions), 2)  # v1.0.0, backup


# ==================== 性能测试 ====================

class TestFeatureVersionManagementPerformance(unittest.TestCase):
    """特征版本管理性能测试类"""
    
    def setUp(self):
        self.config = FeatureConfig()
        self.engine = FeatureEngine(self.config)
        self.temp_dir = tempfile.mkdtemp()
        self.config.data_dir = self.temp_dir
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_version_performance(self):
        """测试创建版本的性能"""
        import time
        
        start_time = time.time()
        
        # 创建100个版本
        for i in range(100):
            self.engine.create_feature_version(
                f"v{i}.0.0",
                ["SMA", "EMA", "RSI"],
                f"版本{i}"
            )
        
        elapsed_time = time.time() - start_time
        
        # 应该在5秒内完成
        self.assertLess(elapsed_time, 5.0)
        print(f"创建100个版本耗时: {elapsed_time:.2f}秒")
    
    def test_list_versions_performance(self):
        """测试列出版本的性能"""
        import time
        
        # 先创建100个版本
        for i in range(100):
            self.engine.create_feature_version(
                f"v{i}.0.0",
                ["SMA", "EMA"],
                f"版本{i}"
            )
        
        start_time = time.time()
        
        # 列出所有版本
        versions = self.engine.list_feature_versions()
        
        elapsed_time = time.time() - start_time
        
        # 应该在1秒内完成
        self.assertLess(elapsed_time, 1.0)
        self.assertEqual(len(versions), 100)
        print(f"列出100个版本耗时: {elapsed_time:.2f}秒")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
