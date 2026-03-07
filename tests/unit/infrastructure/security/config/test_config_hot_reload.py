#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 配置热更新测试

测试配置管理器的热更新功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import json
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from src.infrastructure.security.access.components.config_manager import ConfigManager


class TestConfigHotReload(unittest.TestCase):
    """配置热更新测试"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_config.json"

        # 创建测试配置
        self.test_config = {
            "version": "1.0",
            "cache": {"enabled": True, "max_size": 100},
            "audit": {"enabled": True, "log_level": "INFO"}
        }

        # 保存初始配置
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """测试后清理"""
        # 停止所有监控线程
        import psutil
        import os

        try:
            current_process = psutil.Process(os.getpid())
            for thread in current_process.threads():
                # 这里可以添加线程清理逻辑
                pass
        except:
            pass

    def test_hot_reload_initialization(self):
        """测试热更新初始化"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=True
        )

        # 验证初始化状态
        self.assertTrue(config_mgr.enable_hot_reload)
        self.assertIsNotNone(config_mgr._monitor_thread)
        self.assertTrue(config_mgr._monitor_thread.is_alive())

        config_mgr.shutdown()

    def test_config_hash_calculation(self):
        """测试配置哈希计算"""
        # 创建一个临时的配置文件
        temp_config_file = self.temp_dir / "hash_test_config.json"

        # 写入初始配置
        initial_config = {"test": "value1"}
        with open(temp_config_file, 'w') as f:
            json.dump(initial_config, f)

        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=False
        )

        # 修改ConfigManager的config_file指向临时文件
        config_mgr.config_file = temp_config_file

        # 计算初始哈希
        hash1 = config_mgr._calculate_config_hash()
        self.assertNotEqual(hash1, "")

        # 修改配置文件内容
        modified_config = {"test": "value2", "new_field": "added"}
        with open(temp_config_file, 'w') as f:
            json.dump(modified_config, f)

        # 再次计算哈希，应该不同
        hash2 = config_mgr._calculate_config_hash()
        self.assertNotEqual(hash1, hash2, f"Hashes should be different: {hash1} vs {hash2}")

    def test_config_change_detection(self):
        """测试配置变更检测"""
        # 创建一个新的配置文件用于测试
        test_config_file = self.temp_dir / "change_test_config.json"

        # 写入初始配置
        initial_config = {"test": "initial"}
        with open(test_config_file, 'w') as f:
            json.dump(initial_config, f)

        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=False
        )

        # 设置测试配置文件
        config_mgr.config_file = test_config_file
        config_mgr._config_hash = config_mgr._calculate_config_hash()

        # 初始状态，不应该检测到变化
        self.assertFalse(config_mgr._check_config_file_changed())

        # 修改配置文件
        modified_config = {"test": "modified"}
        with open(test_config_file, 'w') as f:
            json.dump(modified_config, f)

        # 应该检测到变化
        self.assertTrue(config_mgr._check_config_file_changed())

    def test_manual_reload(self):
        """测试手动重新加载"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=False
        )

        # 修改配置文件
        self.test_config["audit"]["log_level"] = "DEBUG"
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)

        # 手动重新加载
        success = config_mgr.trigger_manual_reload()
        self.assertTrue(success)

        # 验证配置已更新
        audit_config = config_mgr.get_config("audit")
        self.assertEqual(audit_config["log_level"], "DEBUG")

    def test_config_callbacks(self):
        """测试配置变更回调"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=False
        )

        # 添加回调
        callback_called = False
        new_config = None

        def test_callback(config):
            nonlocal callback_called, new_config
            callback_called = True
            new_config = config

        config_mgr.add_config_change_callback(test_callback)

        # 修改配置并手动重新加载
        self.test_config["cache"]["max_size"] = 500
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)

        config_mgr.trigger_manual_reload()

        # 验证回调被调用
        self.assertTrue(callback_called)
        self.assertIsNotNone(new_config)
        self.assertEqual(new_config["cache"]["max_size"], 500)

        # 移除回调
        config_mgr.remove_config_change_callback(test_callback)

    def test_hot_reload_with_invalid_config(self):
        """测试热更新时配置验证失败的处理"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=False
        )

        original_config = config_mgr._config.copy()

        # 创建无效配置（缺少必需字段）
        invalid_config = {"invalid": "config"}
        with open(self.config_file, 'w') as f:
            json.dump(invalid_config, f)

        # 手动重新加载应该失败
        success = config_mgr.trigger_manual_reload()
        self.assertFalse(success)

        # 配置应该回滚到原始状态
        self.assertEqual(config_mgr._config, original_config)

    def test_hot_reload_disabled(self):
        """测试禁用热更新"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=False
        )

        # 验证热更新被禁用
        self.assertFalse(config_mgr.enable_hot_reload)
        self.assertIsNone(config_mgr._monitor_thread)

        # 验证配置摘要
        summary = config_mgr.get_config_summary()
        self.assertFalse(summary["hot_reload_enabled"])

    def test_config_summary_with_hot_reload(self):
        """测试包含热更新信息的配置摘要"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=True
        )

        summary = config_mgr.get_config_summary()
        self.assertTrue(summary["hot_reload_enabled"])

        config_mgr.shutdown()

    def test_multiple_callbacks(self):
        """测试多个配置变更回调"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=False
        )

        callback_count = 0

        def callback1(config):
            nonlocal callback_count
            callback_count += 1

        def callback2(config):
            nonlocal callback_count
            callback_count += 1

        config_mgr.add_config_change_callback(callback1)
        config_mgr.add_config_change_callback(callback2)

        # 修改配置并手动重新加载
        self.test_config["cache"]["max_size"] = 300
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)

        config_mgr.trigger_manual_reload()

        # 验证两个回调都被调用
        self.assertEqual(callback_count, 2)

    def test_shutdown_with_hot_reload(self):
        """测试启用热更新时的关闭操作"""
        config_mgr = ConfigManager(
            config_path=self.temp_dir,
            enable_hot_reload=True
        )

        # 验证监控线程正在运行
        self.assertIsNotNone(config_mgr._monitor_thread)
        self.assertTrue(config_mgr._monitor_thread.is_alive())

        # 关闭配置管理器
        config_mgr.shutdown()

        # 验证监控线程已停止
        self.assertTrue(config_mgr._stop_monitoring.is_set())


if __name__ == '__main__':
    unittest.main()
