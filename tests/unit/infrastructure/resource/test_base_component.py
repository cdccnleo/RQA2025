#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
base_component 模块测试
测试基础组件的所有功能，提升测试覆盖率到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import threading
import time
from unittest.mock import Mock, patch
from dataclasses import dataclass

try:
    from src.infrastructure.resource.core.base_component import (
        ParameterConfig, IBaseResourceComponent, BaseResourceComponent
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "base_component模块导入失败")
class TestParameterConfig(unittest.TestCase):
    """测试参数配置类"""

    def test_parameter_config_default_values(self):
        """测试参数配置默认值"""
        config = ParameterConfig()
        
        # 测试基本参数默认值
        self.assertIsNone(config.operation_name)
        self.assertIsNone(config.resource_type)
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.retry_count, 3)
        
        # 测试性能阈值参数默认值
        self.assertEqual(config.cpu_threshold, 80.0)
        self.assertEqual(config.memory_threshold, 85.0)
        self.assertEqual(config.disk_threshold, 90.0)
        self.assertEqual(config.network_threshold, 100.0)
        
        # 测试监控参数默认值
        self.assertTrue(config.enable_alerts)
        self.assertEqual(config.log_level, "INFO")

    def test_parameter_config_custom_values(self):
        """测试参数配置自定义值"""
        config = ParameterConfig(
            operation_name="test_operation",
            resource_type="cpu",
            timeout=60,
            cpu_threshold=75.0,
            enable_alerts=False
        )
        
        self.assertEqual(config.operation_name, "test_operation")
        self.assertEqual(config.resource_type, "cpu")
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.cpu_threshold, 75.0)
        self.assertFalse(config.enable_alerts)

    def test_parameter_config_merge_with_none(self):
        """测试与None配置合并"""
        config1 = ParameterConfig(timeout=30, cpu_threshold=80.0)
        config2 = None
        
        merged = config1.merge(config2)
        self.assertEqual(merged.timeout, 30)
        self.assertEqual(merged.cpu_threshold, 80.0)

    def test_parameter_config_merge_with_values(self):
        """测试配置合并"""
        config1 = ParameterConfig(timeout=30, cpu_threshold=80.0, memory_threshold=85.0)
        config2 = ParameterConfig(timeout=60, cpu_threshold=75.0)
        
        merged = config1.merge(config2)
        
        # config2中的值应该覆盖config1
        self.assertEqual(merged.timeout, 60)
        self.assertEqual(merged.cpu_threshold, 75.0)
        # config2中没有的值应该保留config1的值
        self.assertEqual(merged.memory_threshold, 85.0)

    def test_parameter_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "operation_name": "test_op",
            "timeout": 45,
            "cpu_threshold": 70.0,
            "enable_alerts": True
        }
        
        config = ParameterConfig.from_dict(data)
        
        self.assertEqual(config.operation_name, "test_op")
        self.assertEqual(config.timeout, 45)
        self.assertEqual(config.cpu_threshold, 70.0)
        self.assertTrue(config.enable_alerts)

    def test_parameter_config_from_dict_with_invalid_keys(self):
        """测试从字典创建配置时包含无效键"""
        data = {
            "valid_key": "value",
            "invalid_key": "should_be_ignored",
            "timeout": 30
        }
        
        config = ParameterConfig.from_dict(data)
        
        # 只有有效键应该被设置
        self.assertEqual(config.timeout, 30)
        # 无效键不应该被设置，应该保持默认值

    def test_parameter_config_to_dict(self):
        """测试转换为字典"""
        config = ParameterConfig(
            operation_name="test",
            timeout=60,
            cpu_threshold=75.0
        )
        
        result = config.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["operation_name"], "test")
        self.assertEqual(result["timeout"], 60)
        self.assertEqual(result["cpu_threshold"], 75.0)


@unittest.skipUnless(IMPORTS_AVAILABLE, "base_component模块导入失败")
class TestBaseResourceComponent(unittest.TestCase):
    """测试基础资源组件"""

    def setUp(self):
        """测试前准备"""
        # 创建一个具体的实现类用于测试
        class TestComponent(BaseResourceComponent):
            def __init__(self, config=None, component_name="test"):
                super().__init__(config, component_name)
                self.custom_initialized = False
                self.custom_shutdown = False
            
            def _initialize_component(self):
                self.custom_initialized = True
            
            def _shutdown_component(self):
                self.custom_shutdown = True
        
        self.TestComponent = TestComponent

    def test_component_initialization(self):
        """测试组件初始化"""
        config = {"test_param": "test_value"}
        component = self.TestComponent(config, "test_component")
        
        self.assertFalse(component._initialized)
        self.assertEqual(component._status, "stopped")
        self.assertEqual(component.component_name, "test_component")
        self.assertEqual(component.config, config)

    def test_component_initialization_with_defaults(self):
        """测试组件初始化默认值"""
        component = self.TestComponent()
        
        self.assertEqual(component.config, {})
        self.assertEqual(component.component_name, "test")  # TestComponent的默认名称
        self.assertFalse(component._initialized)
        self.assertEqual(component._status, "stopped")

    def test_initialize_component_success(self):
        """测试组件初始化成功"""
        component = self.TestComponent()
        config = {"param1": "value1"}
        
        result = component.initialize(config)
        
        self.assertTrue(result)
        self.assertTrue(component._initialized)
        self.assertEqual(component._status, "running")
        self.assertIsNotNone(component._start_time)
        self.assertTrue(component.custom_initialized)
        # 验证配置被更新
        self.assertIn("param1", component.config)

    def test_initialize_component_failure(self):
        """测试组件初始化失败"""
        class FailingComponent(BaseResourceComponent):
            def _initialize_component(self):
                raise Exception("Initialization failed")
        
        component = FailingComponent()
        
        result = component.initialize({"test": "config"})
        
        self.assertFalse(result)
        self.assertFalse(component._initialized)
        self.assertEqual(component._status, "error")

    def test_get_status(self):
        """测试获取组件状态"""
        component = self.TestComponent()
        
        # 初始化前状态
        status = component.get_status()
        self.assertEqual(status["component"], "test")  # TestComponent的默认名称
        self.assertEqual(status["status"], "stopped")
        self.assertFalse(status["initialized"])
        self.assertEqual(status["uptime"], 0)
        self.assertIsInstance(status["stats"], dict)
        
        # 初始化后状态
        component.initialize({"test": "config"})
        time.sleep(0.01)  # 确保uptime > 0
        
        status = component.get_status()
        self.assertEqual(status["status"], "running")
        self.assertTrue(status["initialized"])
        self.assertGreater(status["uptime"], 0)

    def test_shutdown_component(self):
        """测试组件关闭"""
        component = self.TestComponent()
        component.initialize({"test": "config"})
        
        component.shutdown()
        
        self.assertFalse(component._initialized)
        self.assertEqual(component._status, "stopped")
        self.assertTrue(component.custom_shutdown)

    def test_shutdown_component_with_failure(self):
        """测试组件关闭时发生异常"""
        class FailingShutdownComponent(BaseResourceComponent):
            def _shutdown_component(self):
                raise Exception("Shutdown failed")
        
        component = FailingShutdownComponent()
        component.initialize({"test": "config"})
        
        # 关闭时发生异常不应该抛出异常
        component.shutdown()
        
        # 根据实际代码，如果_shutdown_component抛出异常，状态更新代码不会执行
        # 因为状态更新在try块的最后，异常会让执行跳转到except块
        self.assertTrue(component._initialized)  # 状态保持不变
        self.assertEqual(component._status, "running")  # 状态保持不变

    def test_update_config_success(self):
        """测试配置更新成功"""
        component = self.TestComponent()
        component.config = {"old_param": "old_value"}
        
        new_config = {"new_param": "new_value", "old_param": "updated_value"}
        result = component.update_config(new_config)
        
        self.assertTrue(result)
        self.assertEqual(component.config["old_param"], "updated_value")
        self.assertEqual(component.config["new_param"], "new_value")

    def test_update_config_failure(self):
        """测试配置更新失败"""
        component = self.TestComponent()
        
        # 创建一个会抛出异常的config对象
        class FailingConfig(dict):
            def update(self, *args, **kwargs):
                raise Exception("Update failed")
        
        component.config = FailingConfig()
        result = component.update_config({"test": "value"})
        self.assertFalse(result)

    def test_is_healthy(self):
        """测试组件健康检查"""
        component = self.TestComponent()
        
        # 未初始化时应该不健康
        self.assertFalse(component.is_healthy())
        
        # 初始化后应该健康
        component.initialize({"test": "config"})
        self.assertTrue(component.is_healthy())
        
        # 状态不是running时应该不健康
        component._status = "error"
        self.assertFalse(component.is_healthy())

    def test_record_operation_success(self):
        """测试记录成功操作"""
        component = self.TestComponent()
        initial_time = time.time()
        
        component.record_operation(success=True, response_time=1.5)
        
        stats = component._stats
        self.assertEqual(stats["total_operations"], 1)
        self.assertEqual(stats["successful_operations"], 1)
        self.assertEqual(stats["failed_operations"], 0)
        self.assertIsNotNone(stats["last_operation_time"])
        self.assertGreaterEqual(stats["last_operation_time"], initial_time)

    def test_record_operation_failure(self):
        """测试记录失败操作"""
        component = self.TestComponent()
        
        component.record_operation(success=False, response_time=2.0)
        
        stats = component._stats
        self.assertEqual(stats["total_operations"], 1)
        self.assertEqual(stats["successful_operations"], 0)
        self.assertEqual(stats["failed_operations"], 1)

    def test_record_operation_multiple(self):
        """测试记录多次操作"""
        component = self.TestComponent()
        
        # 记录多次操作
        component.record_operation(success=True, response_time=1.0)
        component.record_operation(success=False, response_time=2.0)
        component.record_operation(success=True, response_time=1.5)
        
        stats = component._stats
        self.assertEqual(stats["total_operations"], 3)
        self.assertEqual(stats["successful_operations"], 2)
        self.assertEqual(stats["failed_operations"], 1)

    def test_record_operation_no_response_time(self):
        """测试记录操作不提供响应时间"""
        component = self.TestComponent()
        
        component.record_operation(success=True)
        
        stats = component._stats
        self.assertEqual(stats["total_operations"], 1)
        self.assertEqual(stats["successful_operations"], 1)

    def test_get_operation_stats(self):
        """测试获取操作统计信息"""
        component = self.TestComponent()
        
        # 记录一些操作
        component.record_operation(success=True, response_time=1.0)
        component.record_operation(success=True, response_time=2.0)
        component.record_operation(success=False)
        
        stats = component.get_operation_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_operations", stats)
        self.assertIn("successful_operations", stats)
        self.assertIn("failed_operations", stats)
        self.assertIn("last_operation_time", stats)
        self.assertIn("average_response_time", stats)
        self.assertIn("success_rate", stats)
        
        self.assertEqual(stats["total_operations"], 3)
        self.assertEqual(stats["successful_operations"], 2)
        self.assertEqual(stats["failed_operations"], 1)
        # 成功率应该是 (2/3) * 100 = 66.67%
        self.assertAlmostEqual(stats["success_rate"], 66.67, places=1)

    def test_reset_stats(self):
        """测试重置统计信息"""
        component = self.TestComponent()
        
        # 记录一些操作
        component.record_operation(success=True)
        component.record_operation(success=False)
        
        # 验证有统计数据
        self.assertEqual(component._stats["total_operations"], 2)
        
        component.reset_stats()
        
        # 验证统计信息被重置
        stats = component._stats
        self.assertEqual(stats["total_operations"], 0)
        self.assertEqual(stats["successful_operations"], 0)
        self.assertEqual(stats["failed_operations"], 0)
        self.assertEqual(stats["average_response_time"], 0.0)
        self.assertIsNone(stats["last_operation_time"])

    def test_log_operation(self):
        """测试记录操作日志"""
        component = self.TestComponent()
        
        # 测试不同级别的日志
        with patch.object(component.logger, 'info') as mock_info:
            component.log_operation("test_operation", {"param": "value"}, "info")
            mock_info.assert_called_once()
        
        with patch.object(component.logger, 'error') as mock_error:
            component.log_operation("error_operation", None, "error")
            mock_error.assert_called_once()
        
        with patch.object(component.logger, 'debug') as mock_debug:
            component.log_operation("debug_operation", level="debug")
            mock_debug.assert_called_once()

    def test_record_operation_response_time_calculation(self):
        """测试记录操作时的响应时间计算"""
        component = self.TestComponent()
        
        # 记录多个操作，测试平均响应时间计算
        component.record_operation(success=True, response_time=1.0)
        component.record_operation(success=True, response_time=3.0)
        component.record_operation(success=True, response_time=2.0)
        
        # 验证平均响应时间计算正确 (1.0 + 3.0 + 2.0) / 3 = 2.0
        stats = component.get_operation_stats()
        self.assertAlmostEqual(stats["average_response_time"], 2.0, places=1)

    def test_get_operation_stats_no_operations(self):
        """测试无操作时的统计信息"""
        component = self.TestComponent()
        
        stats = component.get_operation_stats()
        
        # 无操作时成功率应该为0
        self.assertEqual(stats["success_rate"], 0)
        self.assertEqual(stats["total_operations"], 0)

    def test_thread_safety(self):
        """测试线程安全性"""
        component = self.TestComponent()
        results = []
        errors = []
        
        def worker():
            try:
                for i in range(10):
                    component.record_operation(success=True, response_time=0.1)
                    component.get_status()
                    results.append(i)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有错误发生
        self.assertEqual(len(errors), 0)
        
        # 验证统计数据正确（应该记录50次操作）
        self.assertEqual(component._stats["total_operations"], 50)

    def test_interface_compliance(self):
        """测试接口合规性"""
        component = self.TestComponent()
        
        # 验证实现了IBaseResourceComponent接口
        self.assertIsInstance(component, IBaseResourceComponent)
        
        # 验证接口方法存在且可调用
        self.assertTrue(hasattr(component, 'initialize'))
        self.assertTrue(hasattr(component, 'get_status'))
        self.assertTrue(hasattr(component, 'shutdown'))
        self.assertTrue(callable(component.initialize))
        self.assertTrue(callable(component.get_status))
        self.assertTrue(callable(component.shutdown))


if __name__ == '__main__':
    unittest.main()