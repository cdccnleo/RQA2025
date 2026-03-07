#!/usr/bin/env python3
"""
RQA2025 基础设施层服务注册器测试

测试基础设施服务注册器的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.config.services.service_registry import InfrastructureServiceRegistry


class TestInfrastructureServiceRegistry(unittest.TestCase):
    """测试基础设施服务注册器"""

    def setUp(self):
        """测试前准备"""
        self.registry = InfrastructureServiceRegistry()

    def tearDown(self):
        """测试后清理"""
        # 重置注册状态
        self.registry._registered = False

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.registry, InfrastructureServiceRegistry)
        self.assertFalse(self.registry._registered)

    @patch('src.infrastructure.config.services.service_registry.get_container')
    def test_register_all_services_without_container(self, mock_get_container):
        """测试在没有容器的情况下注册服务"""
        mock_get_container.return_value = None

        # 创建新的注册器实例
        registry = InfrastructureServiceRegistry()

        # 注册服务
        registry.register_all_services()

        # 验证没有抛出异常且注册状态为False
        self.assertFalse(registry._registered)

    @patch('src.infrastructure.config.services.service_registry.get_container')
    def test_register_all_services_with_container(self, mock_get_container):
        """测试在有容器的情况下注册服务"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container

        # 创建新的注册器实例
        registry = InfrastructureServiceRegistry()

        # 注册服务
        registry.register_all_services()

        # 验证注册状态为True
        self.assertTrue(registry._registered)

        # 验证容器注册方法被调用
        self.assertTrue(mock_container.register_singleton.called)

    @patch('src.infrastructure.config.services.service_registry.get_container')
    def test_register_config_services_success(self, mock_get_container):
        """测试配置服务注册成功"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_config_services()

        # 验证UnifiedConfigManager被注册为单例
        mock_container.register_singleton.assert_called()
        call_args = mock_container.register_singleton.call_args
        self.assertEqual(call_args[0][0].__name__, 'UnifiedConfigManager')

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.UnifiedDatabaseManager')
    def test_register_database_services_success(self, mock_db_manager, mock_get_container):
        """测试数据库服务注册成功"""
        mock_container = Mock()
        mock_config_manager = Mock()
        mock_container.resolve.return_value = mock_config_manager
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_database_services()

        # 验证UnifiedDatabaseManager被注册
        mock_container.register_singleton.assert_called_once()
        call_args = mock_container.register_singleton.call_args
        # 检查第一个参数是mock对象本身
        self.assertEqual(call_args[0][0], mock_db_manager)

        # 验证factory函数被调用时会解析配置管理器
        factory = call_args[1]['factory']
        factory(mock_container)
        mock_container.resolve.assert_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.MemoryCacheManager')
    def test_register_cache_services_success(self, mock_memory_cache, mock_get_container):
        """测试缓存服务注册成功"""
        mock_container = Mock()
        mock_config_manager = Mock()
        mock_config_manager.get.side_effect = lambda key, default: {
            'cache.memory.max_size': 1000,
            'cache.memory.ttl': 600
        }.get(key, default)
        mock_container.resolve.return_value = mock_config_manager
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_cache_services()

        # 验证MemoryCacheManager被注册
        mock_container.register_singleton.assert_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.AutomationMonitor')
    def test_register_monitoring_services_success(self, mock_monitor, mock_get_container):
        """测试监控服务注册成功"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_monitoring_services()

        # 验证AutomationMonitor被注册
        mock_container.register_singleton.assert_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.ErrorHandler')
    def test_register_error_services_success(self, mock_error_handler, mock_get_container):
        """测试错误处理服务注册成功"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_error_services()

        # 验证ErrorHandler被注册
        mock_container.register_singleton.assert_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.Logger')
    def test_register_logging_services_success(self, mock_logger, mock_get_container):
        """测试日志服务注册成功"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_logging_services()

        # 验证Logger被注册
        mock_container.register_singleton.assert_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.HealthChecker')
    def test_register_health_services_success(self, mock_health_checker, mock_get_container):
        """测试健康检查服务注册成功"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_health_services()

        # 验证HealthChecker被注册
        mock_container.register_singleton.assert_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.DeploymentValidator')
    def test_register_deployment_services_success(self, mock_deployment_validator, mock_get_container):
        """测试部署服务注册成功"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()
        registry._register_deployment_services()

        # 验证DeploymentValidator被注册
        mock_container.register_singleton.assert_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.logger')
    def test_register_services_exception_handling(self, mock_logger, mock_get_container):
        """测试服务注册异常处理"""
        mock_container = Mock()
        mock_container.register_singleton.side_effect = Exception("Registration failed")
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()

        # 测试配置服务注册异常 - 直接调用实际方法，因为它有异常处理
        registry._register_config_services()

        # 验证警告日志被记录
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Failed to register config services", warning_call)

    def test_idempotent_registration(self):
        """测试重复注册的幂等性"""
        # 第一次注册
        self.registry._registered = False
        with patch.object(self.registry, 'container', None):
            self.registry.register_all_services()
            self.assertFalse(self.registry._registered)

        # 第二次注册应该直接返回
        self.registry._registered = True
        with patch.object(self.registry, '_register_config_services') as mock_register:
            self.registry.register_all_services()
            # 验证注册方法没有被调用
            mock_register.assert_not_called()

    @patch('src.infrastructure.config.services.service_registry.get_container')
    @patch('src.infrastructure.config.services.service_registry.UnifiedDatabaseManager')
    def test_service_registration_with_dependencies(self, mock_db_manager_class, mock_get_container):
        """测试带依赖关系的服务注册"""
        mock_container = Mock()
        mock_config_manager = Mock()
        mock_container.resolve.return_value = mock_config_manager
        mock_get_container.return_value = mock_container

        registry = InfrastructureServiceRegistry()

        # 测试数据库服务的依赖注入
        registry._register_database_services()

        # 获取factory函数并执行
        call_args = mock_container.register_singleton.call_args
        factory = call_args[1]['factory']
        result = factory(mock_container)

        # 验证依赖解析被调用
        mock_container.resolve.assert_called()
        # 验证UnifiedDatabaseManager被正确实例化
        mock_db_manager_class.assert_called_once_with(config_manager=mock_config_manager)


if __name__ == '__main__':
    unittest.main()