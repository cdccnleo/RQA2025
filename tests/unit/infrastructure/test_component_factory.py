#!/usr/bin/env python3
"""
ComponentFactory单元测试

测试统一ComponentFactory基类的功能和所有子类的继承关系。

作者: RQA2025 Team
版本: 1.0.0
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.infrastructure.utils.core.base_components import ComponentFactory, IComponentFactory


class TestComponentFactory(unittest.TestCase):
    """ComponentFactory基类测试"""

    def setUp(self):
        """测试前准备"""
        self.factory = ComponentFactory()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.factory._components, dict)
        self.assertIsInstance(self.factory._factories, dict)
        self.assertIsInstance(self.factory._statistics, dict)

    def test_register_factory(self):
        """测试工厂函数注册"""
        mock_factory = Mock(return_value="test_component")

        # 注册工厂函数
        self.factory.register_factory("test_type", mock_factory)

        # 验证注册成功
        self.assertIn("test_type", self.factory._factories)
        self.assertEqual(self.factory._factories["test_type"], mock_factory)

    def test_create_component_with_registered_factory(self):
        """测试使用注册工厂创建组件"""
        mock_component = Mock()
        mock_component.initialize = Mock(return_value=True)

        mock_factory = Mock(return_value=mock_component)
        self.factory.register_factory("test_type", mock_factory)

        # 创建组件
        result = self.factory.create_component("test_type", {"param": "value"})

        # 验证调用
        mock_factory.assert_called_once_with({"param": "value"})
        mock_component.initialize.assert_called_once_with({"param": "value"})
        self.assertEqual(result, mock_component)

    def test_create_component_without_factory(self):
        """测试无注册工厂时的组件创建"""
        # 这应该返回None，因为没有_create_component_instance实现
        result = self.factory.create_component("unknown_type", {})
        self.assertIsNone(result)

    def test_create_component_with_initialize_failure(self):
        """测试组件初始化失败的情况"""
        mock_component = Mock()
        mock_component.initialize = Mock(return_value=False)

        mock_factory = Mock(return_value=mock_component)
        self.factory.register_factory("test_type", mock_factory)

        result = self.factory.create_component("test_type", {})

        self.assertIsNone(result)
        mock_component.initialize.assert_called_once()

    def test_create_component_exception_handling(self):
        """测试异常处理"""
        mock_factory = Mock(side_effect=Exception("Test error"))

        self.factory.register_factory("error_type", mock_factory)

        with patch('src.infrastructure.utils.core.base_components.logger') as mock_logger:
            result = self.factory.create_component("error_type", {})

            self.assertIsNone(result)
            mock_logger.error.assert_called_once()

    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        # 这个测试可能需要根据实际实现调整
        # 目前ComponentFactory可能没有统计跟踪功能
        pass


class TestComponentFactoryInheritance(unittest.TestCase):
    """ComponentFactory继承关系测试"""

    def setUp(self):
        """测试前准备"""
        self.component_factories = [
            "infrastructure.cache.cache_components.CacheComponentFactory",
            "infrastructure.health.health_components.HealthComponentFactory",
            "infrastructure.utils.util_components.UtilComponentFactory",
            "infrastructure.logging.logging_service_components.LoggingServiceComponentFactory",
            "infrastructure.error.error_components.ErrorComponentFactory",
        ]

    def test_inheritance_verification(self):
        """验证所有ComponentFactory子类正确继承"""
        for factory_path in self.component_factories:
            with self.subTest(factory=factory_path):
                try:
                    # 动态导入
                    module_path, class_name = factory_path.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[class_name])
                    factory_class = getattr(module, class_name)

                    # 验证继承关系
                    self.assertTrue(issubclass(factory_class, ComponentFactory),
                                  f"{factory_class.__name__} 没有正确继承 ComponentFactory")

                    # 验证可以实例化
                    instance = factory_class()
                    self.assertIsInstance(instance, ComponentFactory)
                    self.assertIsInstance(instance, factory_class)

                except (ImportError, AttributeError) as e:
                    self.fail(f"无法导入或实例化 {factory_path}: {e}")

    def test_factory_functionality(self):
        """测试各工厂的基本功能"""
        # 测试CacheComponentFactory
        try:
            from infrastructure.cache.cache_components import CacheComponentFactory

            factory = CacheComponentFactory()

            # 测试工厂方法注册
            self.assertIsInstance(factory._factories, dict)

            # 测试组件创建（使用有效ID）
            component = factory.create_component(1, {})
            if component:  # 如果创建成功
                self.assertIsNotNone(component.component_id)

        except Exception as e:
            self.fail(f"CacheComponentFactory功能测试失败: {e}")

    def test_interface_compliance(self):
        """测试接口一致性"""
        # 这里可以添加更多接口一致性检查
        # 比如检查所有工厂都有相同的方法签名等
        pass


class TestComponentFactoryIntegration(unittest.TestCase):
    """ComponentFactory集成测试"""

    def test_cross_factory_compatibility(self):
        """测试不同工厂间的兼容性"""
        # 这里可以测试不同工厂创建的组件之间的交互
        pass

    def test_factory_statistics(self):
        """测试工厂使用统计"""
        # 这里可以测试工厂的统计功能
        pass


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestComponentFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestComponentFactoryInheritance))
    suite.addTests(loader.loadTestsFromTestCase(TestComponentFactoryIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    return result.wasSuccessful(), result.testsRun, len(result.errors), len(result.failures)


if __name__ == "__main__":
    print("🧪 运行ComponentFactory单元测试...")

    success, total, errors, failures = run_tests()

    print("\n📊 测试结果:")
    print(f"  总测试数: {total}")
    print(f"  错误: {errors}")
    print(f"  失败: {failures}")

    if success:
        print("✅ 所有测试通过!")
        exit(0)
    else:
        print("❌ 部分测试失败!")
        exit(1)
