"""
基础设施层 - BaseResourceComponent 单元测试

测试基础资源组件的核心功能。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch

from src.infrastructure.resource.core.base import BaseResourceComponent
from src.infrastructure.resource.core.base_component import IResourceComponent


class TestBaseResourceComponent(unittest.TestCase):
    """BaseResourceComponent 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.component = BaseResourceComponent()

    def tearDown(self):
        """测试后清理"""
        pass

    def test_initialization(self):
        """测试初始化"""
        # 测试无参数初始化
        component = BaseResourceComponent()
        self.assertIsInstance(component, BaseResourceComponent)
        self.assertIsInstance(component, IResourceComponent)
        self.assertEqual(component.config, {})  # 默认为空字典
        self.assertFalse(component._initialized)
        self.assertEqual(component._status, "stopped")

        # 测试有参数初始化
        config = {"test": "value"}
        component_with_config = BaseResourceComponent(config)
        self.assertEqual(component_with_config.config, config)

    def test_initialize_success(self):
        """测试成功初始化"""
        config = {"component": "test", "enabled": True}

        result = self.component.initialize(config)

        self.assertTrue(result)
        self.assertTrue(self.component._initialized)
        self.assertEqual(self.component._status, "running")
        self.assertEqual(self.component.config, config)

    def test_initialize_failure(self):
        """测试初始化失败"""
        # 创建一个会抛出异常的config对象
        class FailingDict(dict):
            def update(self, *args, **kwargs):
                raise Exception("初始化失败")

        failing_config = FailingDict()
        component = BaseResourceComponent(failing_config)

        # 直接修改component的config引用使其失败
        component.config = failing_config
        result = component.initialize({})

        self.assertFalse(result)
        self.assertFalse(component._initialized)
        self.assertEqual(component._status, "error")

    def test_get_status(self):
        """测试获取状态"""
        # 默认状态
        status = self.component.get_status()
        expected_status = {
            "component": "resource",
            "status": "stopped",
            "initialized": False,
            "uptime": 0,
            "config": {},
            "stats": {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "last_operation_time": None,
                "average_response_time": 0.0
            }
        }
        self.assertEqual(status, expected_status)

        # 初始化后状态
        self.component.initialize({"test": "config"})
        import time
        time.sleep(0.01)  # 添加小延迟确保uptime > 0
        status = self.component.get_status()

        # 检查关键字段
        self.assertEqual(status["component"], "resource")
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["initialized"], True)
        self.assertEqual(status["config"], {"test": "config"})
        self.assertGreaterEqual(status["uptime"], 0)  # uptime应该>=0
        self.assertIn("stats", status)  # 应该包含stats字段

    def test_shutdown(self):
        """测试关闭组件"""
        # 先初始化
        self.component.initialize({"test": "config"})
        self.assertTrue(self.component._initialized)
        self.assertEqual(self.component._status, "running")

        # 关闭
        self.component.shutdown()
        self.assertFalse(self.component._initialized)
        self.assertEqual(self.component._status, "stopped")

    def test_config_update(self):
        """测试配置更新"""
        initial_config = {"initial": "value"}
        component = BaseResourceComponent(initial_config)

        # 初始化时更新配置
        new_config = {"new": "value", "updated": True}
        component.initialize(new_config)

        # 验证配置已更新
        self.assertIn("initial", component.config)
        self.assertIn("new", component.config)
        self.assertIn("updated", component.config)
        self.assertEqual(component.config["initial"], "value")
        self.assertEqual(component.config["new"], "value")

    def test_interface_compliance(self):
        """测试接口合规性"""
        # 验证实现必要的接口方法
        self.assertTrue(hasattr(self.component, 'initialize'))
        self.assertTrue(hasattr(self.component, 'get_status'))
        self.assertTrue(hasattr(self.component, 'shutdown'))

        # 验证方法签名
        self.assertTrue(callable(self.component.initialize))
        self.assertTrue(callable(self.component.get_status))
        self.assertTrue(callable(self.component.shutdown))

    def test_status_transitions(self):
        """测试状态转换"""
        # stopped -> running
        self.assertEqual(self.component._status, "stopped")
        self.component.initialize({})
        self.assertEqual(self.component._status, "running")

        # running -> stopped
        self.component.shutdown()
        self.assertEqual(self.component._status, "stopped")

    def test_config_reference(self):
        """测试配置引用行为"""
        config = {"mutable": ["item"]}
        component = BaseResourceComponent(config)

        # 验证初始配置正确
        self.assertEqual(component.config["mutable"], ["item"])

        # 修改原始配置会影响组件配置（引用关系）
        config["mutable"].append("new_item")
        self.assertEqual(len(component.config["mutable"]), 2)
        self.assertEqual(component.config["mutable"], ["item", "new_item"])


if __name__ == '__main__':
    unittest.main()


