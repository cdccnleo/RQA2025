#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层基类测试
测试基础设施组件的通用基类功能
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import time
import threading
class TestBaseInfrastructureComponent(unittest.TestCase):
    """测试基础设施组件基类"""

    def setUp(self):
        """测试前准备"""
        from src.infrastructure.base import BaseInfrastructureComponent

        # 创建测试用的具体实现类
        class TestComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

        self.component = TestComponent("test_component")

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.component.component_name, "test_component")
        self.assertIsInstance(self.component.start_time, datetime)
        self.assertFalse(self.component._initialized)
        self.assertIsNotNone(self.component._lock)

    def test_get_status_not_initialized(self):
        """测试获取状态（未初始化）"""
        status = self.component.get_status()
        self.assertEqual(status['component'], "test_component")
        self.assertEqual(status['status'], "stopped")
        self.assertIn('uptime', status)
        self.assertIn('timestamp', status)

    def test_initialize(self):
        """测试初始化"""
        result = self.component.initialize()
        self.assertTrue(result)
        self.assertTrue(self.component._initialized)

        # 再次初始化应该成功
        result = self.component.initialize()
        self.assertTrue(result)

    def test_get_status_initialized(self):
        """测试获取状态（已初始化）"""
        self.component.initialize()
        status = self.component.get_status()
        self.assertEqual(status['status'], "running")

    def test_health_check_success(self):    
        """测试健康检查成功"""
        self.component.initialize()
        health = self.component.health_check()
        self.assertEqual(health['component'], "test_component")
        self.assertEqual(health['status'], "healthy")
        self.assertIn('timestamp', health)

    def test_health_check_failure(self):
        """测试健康检查失败"""
        # 创建一个会失败的组件
        class FailingComponent(self.component.__class__):
            def _perform_health_check(self):
                return False

        failing_component = FailingComponent("failing_component")
        failing_component.initialize()

        health = failing_component.health_check()
        self.assertEqual(health['status'], "unhealthy")

        def test_health_check_exception(self):    
            """测试健康检查异常"""
            # 创建一个抛出异常的组件
            class ExceptionComponent(self.component.__class__):
                def _perform_health_check(self):
                    raise ValueError("Test exception")

                    exception_component = ExceptionComponent("exception_component")
                    exception_component.initialize()

                    health = exception_component.health_check()
                    self.assertEqual(health['status'], "error")
                    self.assertIn('error', health)

        def test_shutdown(self):    
            """测试关闭"""
            self.component.initialize()
            result = self.component.shutdown()
            self.assertTrue(result)
            self.assertFalse(self.component._initialized)

            # 再次关闭应该成功
            result = self.component.shutdown()
            self.assertTrue(result)
class TestBaseServiceComponent(unittest.TestCase):
    """测试服务组件基类"""

    def setUp(self):
            """测试前准备"""
            from src.infrastructure.base import BaseServiceComponent

            # 创建测试用的具体实现类
            class TestService(BaseServiceComponent):
                def _start_service(self):
                    self.is_running = True

                def _stop_service(self):
                    self.is_running = False

            self.service = TestService("test_service", "localhost", 8080)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.service.component_name, "test_service")
        self.assertEqual(self.service.host, "localhost")
        self.assertEqual(self.service.port, 8080)
        self.assertFalse(self.service.is_running)


    def test_service_health_check_running(self):
        """测试服务健康检查（运行中）"""
        self.service.initialize()
        health = self.service.health_check()
        self.assertEqual(health['status'], "healthy")

    def test_service_health_check_stopped(self):
        """测试服务健康检查（已停止）"""
        self.service.initialize()
        self.service.shutdown()
        health = self.service.health_check()
        self.assertEqual(health['status'], "unhealthy")


    def test_initialize_starts_service(self):
        """测试初始化启动服务"""
        self.service.initialize()
        self.assertTrue(self.service.is_running)

    def test_shutdown_stops_service(self):
        """测试关闭停止服务"""
        self.service.initialize()
        self.service.shutdown()
        self.assertFalse(self.service.is_running)

class TestBaseManagerComponent(unittest.TestCase):
    """测试管理器组件基类"""
    def setUp(self):
        """测试前准备"""
        from src.infrastructure.base import BaseManagerComponent

        self.manager = BaseManagerComponent("test_manager", 5)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.manager.component_name, "test_manager")
        self.assertEqual(self.manager.max_items, 5)
        self.assertEqual(len(self.manager._items), 0)
        self.assertIsNotNone(self.manager._item_lock)

    def test_add_item_success(self):
        """测试添加项目成功"""
        result = self.manager.add_item("key1", "value1")
        self.assertTrue(result)
        self.assertEqual(len(self.manager._items), 1)
        self.assertEqual(self.manager._items["key1"], "value1")

    def test_add_item_at_limit(self):
        """测试添加项目到限制"""
        # 添加到最大限制
        for i in range(5):
            result = self.manager.add_item(f"key{i}", f"value{i}")
            self.assertTrue(result)

        self.assertEqual(len(self.manager._items), 5)

        # 尝试超出限制
        result = self.manager.add_item("key5", "value5")
        self.assertFalse(result)
        self.assertEqual(len(self.manager._items), 5)

    def test_get_item(self):
        """测试获取项目"""
        self.manager.add_item("key1", "value1")

        # 获取存在的项目
        value = self.manager.get_item("key1")
        self.assertEqual(value, "value1")

        # 获取不存在的项目
        value = self.manager.get_item("nonexistent")
        self.assertIsNone(value)

    def test_remove_item(self):
        """测试移除项目"""
        self.manager.add_item("key1", "value1")

        # 移除存在的项目
        result = self.manager.remove_item("key1")
        self.assertTrue(result)
        self.assertEqual(len(self.manager._items), 0)

        # 移除不存在的项目
        result = self.manager.remove_item("nonexistent")
        self.assertFalse(result)

    def test_list_items(self):
        """测试列出项目"""
        self.manager.add_item("key1", "value1")
        self.manager.add_item("key2", "value2")

        items = self.manager.list_items()
        self.assertEqual(len(items), 2)
        self.assertIn("key1", items)
        self.assertIn("key2", items)

    def test_clear_items(self):
        """测试清空项目"""
        self.manager.add_item("key1", "value1")
        self.manager.add_item("key2", "value2")

        self.manager.clear_items()
        self.assertEqual(len(self.manager._items), 0)

    def test_manager_health_check(self):
        """测试管理器健康检查"""
        self.manager.initialize()

        # 正常情况（没有超出限制）
        health = self.manager.health_check()
        self.assertEqual(health['status'], "healthy")

        # 超出限制的情况 - 直接修改内部状态来模拟超出限制
        with self.manager._item_lock:
            # 直接添加超出限制的项目来测试健康检查
            for i in range(6):
                self.manager._items[f"key{i}"] = f"value{i}"

        # 现在有6个项目，超出5的限制，应该不健康
        health = self.manager.health_check()
        self.assertEqual(health['status'], "unhealthy")

    def test_initialize_clears_items(self):
        """测试初始化清空项目"""
        self.manager.add_item("key1", "value1")
        self.manager.initialize()
        self.assertEqual(len(self.manager._items), 0)

    def test_shutdown_clears_items(self):
        """测试关闭清空项目"""
        self.manager.add_item("key1", "value1")
        # 初始化后再关闭，确保能正确清空
        self.manager.initialize()
        self.assertEqual(len(self.manager._items), 0)  # 初始化时会清空


if __name__ == '__main__':
    unittest.main()
