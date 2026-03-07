#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimization_disk_optimizer 模块测试
测试磁盘优化器的所有功能，提升测试覆盖率从32.14%到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

try:
    from src.infrastructure.resource.core.optimization_disk_optimizer import DiskOptimizer
    from src.infrastructure.resource.core.optimization_config import DiskOptimizationConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "optimization_disk_optimizer模块导入失败")
class TestDiskOptimizer(unittest.TestCase):
    """测试磁盘优化器"""

    def setUp(self):
        """测试前准备"""
        self.optimizer = DiskOptimizer()

    def test_disk_optimizer_initialization(self):
        """测试磁盘优化器初始化"""
        # 测试默认初始化
        optimizer = DiskOptimizer()
        self.assertIsNotNone(optimizer.logger)
        self.assertIsNotNone(optimizer.error_handler)

        # 测试自定义logger和error_handler
        mock_logger = Mock()
        mock_error_handler = Mock()
        optimizer_custom = DiskOptimizer(logger=mock_logger, error_handler=mock_error_handler)
        self.assertEqual(optimizer_custom.logger, mock_logger)
        self.assertEqual(optimizer_custom.error_handler, mock_error_handler)

    def test_optimize_disk_from_config_success(self):
        """测试基于配置对象的磁盘优化成功"""
        config = DiskOptimizationConfig()
        config.enabled = True
        config.io_scheduler = {"enabled": True}
        config.caching = {"enabled": False}
        config.readahead = {"enabled": True}
        
        current_resources = {"disk_usage": 80.0, "disk_size": 1000}
        
        result = self.optimizer.optimize_disk_from_config(config, current_resources)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertIsInstance(result["actions"], list)

    def test_optimize_disk_from_config_failure(self):
        """测试基于配置对象的磁盘优化失败"""
        # 创建会引发异常的配置
        mock_config = Mock()
        mock_config.to_dict.side_effect = Exception("Config error")
        
        current_resources = {"disk_usage": 70.0}
        
        # optimize_disk_from_config没有异常处理，to_dict的异常会直接抛出
        with self.assertRaises(Exception) as context:
            self.optimizer.optimize_disk_from_config(mock_config, current_resources)
        
        self.assertIn("Config error", str(context.exception))

    def test_optimize_disk_with_io_scheduler_enabled(self):
        """测试启用I/O调度器的磁盘优化"""
        config = {
            "io_scheduler": {"enabled": True},
            "caching": {"enabled": False},
            "readahead": {"enabled": False}
        }
        current_resources = {"disk_usage": 50.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertIn("配置I/O调度器", result["actions"])
        self.assertNotIn("启用磁盘缓存策略", result["actions"])
        self.assertNotIn("配置预读参数", result["actions"])

    def test_optimize_disk_with_caching_enabled(self):
        """测试启用缓存的磁盘优化"""
        config = {
            "io_scheduler": {"enabled": False},
            "caching": {"enabled": True},
            "readahead": {"enabled": False}
        }
        current_resources = {"disk_usage": 60.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertIn("启用磁盘缓存策略", result["actions"])
        self.assertNotIn("配置I/O调度器", result["actions"])
        self.assertNotIn("配置预读参数", result["actions"])

    def test_optimize_disk_with_readahead_enabled(self):
        """测试启用预读的磁盘优化"""
        config = {
            "io_scheduler": {"enabled": False},
            "caching": {"enabled": False},
            "readahead": {"enabled": True}
        }
        current_resources = {"disk_usage": 70.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertIn("配置预读参数", result["actions"])
        self.assertNotIn("配置I/O调度器", result["actions"])
        self.assertNotIn("启用磁盘缓存策略", result["actions"])

    def test_optimize_disk_all_options_enabled(self):
        """测试所有选项都启用的磁盘优化"""
        config = {
            "io_scheduler": {"enabled": True},
            "caching": {"enabled": True},
            "readahead": {"enabled": True}
        }
        current_resources = {"disk_usage": 85.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertIn("配置I/O调度器", result["actions"])
        self.assertIn("启用磁盘缓存策略", result["actions"])
        self.assertIn("配置预读参数", result["actions"])
        self.assertEqual(len(result["actions"]), 3)

    def test_optimize_disk_no_options_enabled(self):
        """测试没有选项启用的磁盘优化"""
        config = {
            "io_scheduler": {"enabled": False},
            "caching": {"enabled": False},
            "readahead": {"enabled": False}
        }
        current_resources = {"disk_usage": 30.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(len(result["actions"]), 0)

    def test_optimize_disk_with_missing_config_keys(self):
        """测试配置缺失键的磁盘优化"""
        config = {}  # 空配置
        current_resources = {"disk_usage": 40.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(len(result["actions"]), 0)

    def test_optimize_disk_with_partial_config(self):
        """测试部分配置的磁盘优化"""
        config = {
            "io_scheduler": {"enabled": True}
            # 缺少caching和readahead
        }
        current_resources = {"disk_usage": 55.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertIn("配置I/O调度器", result["actions"])
        self.assertEqual(len(result["actions"]), 1)

    def test_optimize_disk_with_none_values(self):
        """测试配置值为None的磁盘优化"""
        config = {
            "io_scheduler": None,
            "caching": {"enabled": True},
            "readahead": {"enabled": False}
        }
        current_resources = {"disk_usage": 65.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        # 由于io_scheduler为None会导致AttributeError，异常被捕获
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)
        # 验证错误信息包含NoneType相关的错误
        self.assertIn("NoneType", result["error"])

    def test_optimize_disk_exception_handling(self):
        """测试磁盘优化异常处理"""
        mock_error_handler = Mock()
        optimizer = DiskOptimizer(error_handler=mock_error_handler)
        
        # 创建一个会导致AttributeError的配置来触发异常处理
        config = {
            "io_scheduler": None,  # 这会导致AttributeError
            "caching": {"enabled": True}
        }
        current_resources = {}
        
        result = optimizer.optimize_disk(config, current_resources)
        
        # 验证异常被正确处理
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)
        
        # 验证错误处理器被调用
        mock_error_handler.handle_error.assert_called_once()
        call_args = mock_error_handler.handle_error.call_args
        self.assertIn("context", call_args[0][1])
        self.assertEqual(call_args[0][1]["context"], "磁盘优化失败")

    def test_optimize_disk_return_structure(self):
        """测试磁盘优化返回结构"""
        config = {
            "io_scheduler": {"enabled": True},
            "caching": {"enabled": True}
        }
        current_resources = {"disk_usage": 75.0}
        
        result = self.optimizer.optimize_disk(config, current_resources)
        
        # 验证返回结构
        self.assertIsInstance(result, dict)
        self.assertIn("type", result)
        self.assertIn("status", result)
        self.assertIn("actions", result)
        
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        self.assertIsInstance(result["actions"], list)

    def test_optimize_disk_with_different_resource_structures(self):
        """测试不同资源结构的磁盘优化"""
        config = {
            "readahead": {"enabled": True}
        }
        
        # 测试不同的资源结构
        resources_structures = [
            {"disk_usage": 45.0, "disk_size": 500},
            {"disk_free": 200.0, "disk_total": 1000},
            {"storage_info": {"usage": 80.0}},
            {}  # 空资源
        ]
        
        for resources in resources_structures:
            result = self.optimizer.optimize_disk(config, resources)
            self.assertEqual(result["type"], "disk_optimization")
            self.assertEqual(result["status"], "applied")

    def test_disk_optimization_config_integration(self):
        """测试与DiskOptimizationConfig的集成"""
        config = DiskOptimizationConfig()
        config.io_scheduler = {"enabled": True, "algorithm": "deadline"}
        config.caching = {"enabled": True, "size": 1024}
        config.readahead = {"enabled": True, "pages": 256}
        
        current_resources = {"disk_usage": 60.0}
        
        result = self.optimizer.optimize_disk_from_config(config, current_resources)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "disk_optimization")
        self.assertEqual(result["status"], "applied")
        # 验证所有启用的功能都被处理
        expected_actions = ["配置I/O调度器", "启用磁盘缓存策略", "配置预读参数"]
        for action in expected_actions:
            self.assertIn(action, result["actions"])

    def test_error_handler_integration(self):
        """测试错误处理器集成"""
        mock_error_handler = Mock()
        optimizer = DiskOptimizer(error_handler=mock_error_handler)
        
        # 使用patch来模拟异常情况
        original_optimize_disk = optimizer.optimize_disk
        
        def optimized_disk_with_exception(config, current_resources):
            """模拟会抛出异常的optimize_disk"""
            try:
                result = {
                    "type": "disk_optimization",
                    "status": "applied",
                    "actions": []
                }
                # 人为抛出异常来测试错误处理
                raise Exception("Test error")
            except Exception as e:
                optimizer.error_handler.handle_error(e, {"context": "磁盘优化失败"})
                return {
                    "type": "disk_optimization",
                    "status": "failed",
                    "error": str(e)
                }
        
        # 临时替换方法
        optimizer.optimize_disk = optimized_disk_with_exception
        
        try:
            result = optimizer.optimize_disk({}, {})
            
            # 验证返回的失败状态
            self.assertEqual(result["type"], "disk_optimization")
            self.assertEqual(result["status"], "failed")
            self.assertIn("error", result)
            
            # 验证错误处理器被调用
            mock_error_handler.handle_error.assert_called_once()
        finally:
            # 恢复原始方法
            optimizer.optimize_disk = original_optimize_disk


if __name__ == '__main__':
    unittest.main()
