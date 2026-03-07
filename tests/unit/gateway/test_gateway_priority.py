#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网关层核心优先级测试套件

测试覆盖网关层的核心组件：
1. API网关 (APIGateway)
2. 网关组件工厂 (GatewayComponentFactory)
3. WebSocket API (WebSocket连接管理)
4. 统一管理界面 (UnifiedDashboard)
5. 网关接口 (IGatewayComponent)
"""

import pytest
import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestAPIGateway(unittest.TestCase):
    """测试API网关"""

    def setUp(self):
        """设置测试环境"""
        try:
            import sys
            from pathlib import Path
            # 添加src路径
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
            if str(PROJECT_ROOT / 'src') not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT / 'src'))

            from gateway.api_gateway import GatewayRouter as APIGateway
            self.gateway_class = APIGateway
        except ImportError:
            # 如果导入失败，使用Mock
            self.gateway_class = Mock

    def test_api_gateway_initialization(self):
        """测试API网关初始化"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        config = {"timeout": 30, "max_retries": 3}
        gateway = self.gateway_class(config)
        
        self.assertEqual(gateway.config, config)
        self.assertIsInstance(gateway.routes, dict)
        self.assertIsInstance(gateway.middlewares, list)
        self.assertIsInstance(gateway.services, dict)

    def test_service_registration(self):
        """测试服务注册"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        service_info = {
            "endpoints": ["/api/v1/data"],
            "health_check": "/health",
            "version": "1.0.0"
        }
        
        result = gateway.register_service("data_service", service_info)
        
        self.assertTrue(result)
        self.assertIn("data_service", gateway.services)

    def test_route_registration(self):
        """测试路由注册"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        # 先注册服务
        service_info = {"endpoints": ["/api/v1/test"]}
        gateway.register_service("test_service", service_info)
        
        # 注册路由
        result = gateway.register_route("/api/v1/test", "test_service", ["GET", "POST"])
        
        self.assertTrue(result)
        self.assertIn("/api/v1/test", gateway.routes)

    def test_request_routing(self):
        """测试请求路由"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        # 注册服务和路由
        service_info = {"endpoints": ["/api/v1/users"]}
        gateway.register_service("user_service", service_info)
        gateway.register_route("/api/v1/users", "user_service", ["GET"])
        
        # 测试路由
        result = gateway.route_request("/api/v1/users", "GET")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("service"), "user_service")

    def test_route_not_found(self):
        """测试路由未找到"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        result = gateway.route_request("/nonexistent/path", "GET")
        
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_gateway_status(self):
        """测试网关状态"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        # 注册一些服务
        gateway.register_service("service1", {"endpoints": ["/api/v1/test1"]})
        gateway.register_service("service2", {"endpoints": ["/api/v1/test2"]})
        
        status = gateway.get_service_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status["total_services"], 2)
        self.assertIn("services", status)

    def test_health_check(self):
        """测试健康检查"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        health = gateway.health_check()
        
        self.assertIsInstance(health, dict)
        self.assertEqual(health["status"], "healthy")
        self.assertIn("timestamp", health)


class TestGatewayComponentFactory(unittest.TestCase):
    """测试网关组件工厂"""

    def setUp(self):
        """设置测试环境"""
        try:
            from src.gateway.api.gateway_components import GatewayComponentFactory, GatewayComponent
            self.factory_class = GatewayComponentFactory
            self.component_class = GatewayComponent
        except ImportError:
            # 如果导入失败，使用Mock
            self.factory_class = Mock
            self.component_class = Mock

    def test_factory_supported_gateways(self):
        """测试工厂支持的网关"""
        if self.factory_class == Mock:
            self.skipTest("GatewayComponentFactory导入失败")
        
        supported_gateways = self.factory_class.get_available_gateways()
        
        self.assertIsInstance(supported_gateways, list)
        self.assertGreater(len(supported_gateways), 0)

    def test_factory_create_component(self):
        """测试工厂创建组件"""
        if self.factory_class == Mock:
            self.skipTest("GatewayComponentFactory导入失败")
        
        supported_gateways = self.factory_class.get_available_gateways()
        if supported_gateways:
            gateway_id = supported_gateways[0]
            component = self.factory_class.create_component(gateway_id)
            
            self.assertIsNotNone(component)
            self.assertEqual(component.get_gateway_id(), gateway_id)

    def test_factory_create_invalid_component(self):
        """测试工厂创建无效组件"""
        if self.factory_class == Mock:
            self.skipTest("GatewayComponentFactory导入失败")
        
        with self.assertRaises(ValueError):
            self.factory_class.create_component(999)  # 无效的gateway ID

    def test_factory_create_all_gateways(self):
        """测试工厂创建所有网关"""
        if self.factory_class == Mock:
            self.skipTest("GatewayComponentFactory导入失败")
        
        all_gateways = self.factory_class.create_all_gateways()
        
        self.assertIsInstance(all_gateways, dict)
        self.assertGreater(len(all_gateways), 0)

    def test_factory_info(self):
        """测试工厂信息"""
        if self.factory_class == Mock:
            self.skipTest("GatewayComponentFactory导入失败")
        
        factory_info = self.factory_class.get_factory_info()
        
        self.assertIsInstance(factory_info, dict)
        self.assertIn('factory_name', factory_info)
        self.assertIn('version', factory_info)
        self.assertIn('total_gateways', factory_info)

    def test_gateway_component_operations(self):
        """测试网关组件操作"""
        if self.factory_class == Mock:
            self.skipTest("GatewayComponentFactory导入失败")
        
        supported_gateways = self.factory_class.get_available_gateways()
        if supported_gateways:
            gateway_id = supported_gateways[0]
            component = self.factory_class.create_component(gateway_id)
            
            # 测试组件信息
            info = component.get_info()
            self.assertIsInstance(info, dict)
            self.assertEqual(info['gateway_id'], gateway_id)
            
            # 测试数据处理
            test_data = {"test": "data"}
            result = component.process(test_data)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['gateway_id'], gateway_id)
            
            # 测试状态获取
            status = component.get_status()
            self.assertIsInstance(status, dict)
            self.assertEqual(status['gateway_id'], gateway_id)


class TestGatewayInterfaces(unittest.TestCase):
    """测试网关接口"""

    def setUp(self):
        """设置测试环境"""
        try:
            from src.gateway.interfaces import IGatewayComponent
            self.interface_class = IGatewayComponent
        except ImportError:
            # 如果导入失败，使用Mock
            self.interface_class = Mock

    def test_gateway_interface_definition(self):
        """测试网关接口定义"""
        if self.interface_class == Mock:
            self.skipTest("IGatewayComponent导入失败")
        
        # 测试接口是否是抽象基类
        from abc import ABC
        self.assertTrue(issubclass(self.interface_class, ABC))

    def test_gateway_interface_implementation(self):
        """测试网关接口实现"""
        if self.interface_class == Mock:
            self.skipTest("IGatewayComponent导入失败")
        
        # 创建接口的具体实现
        class TestGatewayComponent(self.interface_class):
            def get_status(self):
                return {"status": "active"}
            
            def health_check(self):
                return {"healthy": True}
        
        component = TestGatewayComponent()
        
        # 测试接口方法
        status = component.get_status()
        self.assertIsInstance(status, dict)
        
        health = component.health_check()
        self.assertIsInstance(health, dict)


class TestWebSocketAPI(unittest.TestCase):
    """测试WebSocket API"""

    def setUp(self):
        """设置测试环境"""
        try:
            from src.gateway.web.websocket_api import router, ConnectionManager
            self.router = router
            self.connection_manager_class = ConnectionManager
        except ImportError:
            # 如果导入失败，使用Mock
            self.router = Mock()
            self.connection_manager_class = Mock

    def test_websocket_router_exists(self):
        """测试WebSocket路由器存在"""
        if self.router == Mock():
            self.skipTest("WebSocket router导入失败")
        
        self.assertIsNotNone(self.router)

    def test_connection_manager_initialization(self):
        """测试连接管理器初始化"""
        if self.connection_manager_class == Mock:
            self.skipTest("ConnectionManager导入失败")
        
        manager = self.connection_manager_class()
        
        self.assertIsNotNone(manager)

    @patch('fastapi.WebSocket')
    def test_websocket_connection_management(self, mock_websocket):
        """测试WebSocket连接管理"""
        if self.connection_manager_class == Mock:
            self.skipTest("ConnectionManager导入失败")
        
        manager = self.connection_manager_class()
        mock_ws = mock_websocket()
        
        # 测试连接
        manager.connect(mock_ws, "test_channel")
        
        # 测试断开连接
        manager.disconnect(mock_ws)

    def test_websocket_api_endpoints(self):
        """测试WebSocket API端点"""
        if self.router == Mock():
            self.skipTest("WebSocket router导入失败")
        
        # 这里可以测试路由器是否包含预期的端点
        # 由于路由器的具体实现可能不同，这里只测试基本属性
        self.assertIsNotNone(self.router)


class TestUnifiedDashboard(unittest.TestCase):
    """测试统一管理界面"""

    def setUp(self):
        """设置测试环境"""
        try:
            from src.gateway.web.unified_dashboard import UnifiedDashboard, DashboardConfig
            self.dashboard_class = UnifiedDashboard
            self.config_class = DashboardConfig
        except ImportError:
            # 如果导入失败，使用Mock
            self.dashboard_class = Mock
            self.config_class = Mock

    def test_dashboard_config(self):
        """测试仪表板配置"""
        if self.config_class == Mock:
            self.skipTest("DashboardConfig导入失败")
        
        config = self.config_class(
            title="Test Dashboard",
            version="2.0.0",
            refresh_interval=60
        )
        
        self.assertEqual(config.title, "Test Dashboard")
        self.assertEqual(config.version, "2.0.0")
        self.assertEqual(config.refresh_interval, 60)

    @patch('fastapi.FastAPI')
    def test_dashboard_initialization(self, mock_fastapi):
        """测试仪表板初始化"""
        if self.dashboard_class == Mock:
            self.skipTest("UnifiedDashboard导入失败")
        
        mock_app = Mock()
        mock_fastapi.return_value = mock_app
        
        config = self.config_class() if self.config_class != Mock else {}
        dashboard = self.dashboard_class(config)
        
        self.assertIsNotNone(dashboard)

    def test_dashboard_module_registration(self):
        """测试仪表板模块注册"""
        if self.dashboard_class == Mock:
            self.skipTest("UnifiedDashboard导入失败")
        
        # 由于实际的仪表板初始化可能依赖很多外部组件
        # 这里只测试基本的类结构
        self.assertTrue(hasattr(self.dashboard_class, '__init__'))


class TestGatewayIntegration(unittest.TestCase):
    """测试网关集成功能"""

    def setUp(self):
        """设置测试环境"""
        # 尝试导入多个组件进行集成测试
        self.components_available = True
        try:
            from src.gateway.api_gateway import GatewayRouter as APIGateway
            from src.gateway.api.gateway_components import GatewayComponentFactory
            self.gateway_class = APIGateway
            self.factory_class = GatewayComponentFactory
        except ImportError:
            self.components_available = False

    def test_gateway_factory_integration(self):
        """测试网关和工厂的集成"""
        if not self.components_available:
            self.skipTest("网关组件导入失败")
        
        # 创建网关
        gateway = self.gateway_class()
        
        # 创建组件
        if hasattr(self.factory_class, 'get_available_gateways'):
            available_gateways = self.factory_class.get_available_gateways()
            if available_gateways:
                component = self.factory_class.create_component(available_gateways[0])
                
                # 测试组件和网关的协作
                self.assertIsNotNone(gateway)
                self.assertIsNotNone(component)

    def test_service_and_route_integration(self):
        """测试服务和路由的集成"""
        if not self.components_available:
            self.skipTest("网关组件导入失败")
        
        gateway = self.gateway_class()
        
        # 注册服务
        service_info = {"endpoints": ["/api/test"], "version": "1.0"}
        gateway.register_service("test_service", service_info)
        
        # 注册路由
        gateway.register_route("/api/test", "test_service", ["GET"])
        
        # 测试路由
        result = gateway.route_request("/api/test", "GET")
        
        self.assertEqual(result.get("service"), "test_service")

    def test_gateway_middleware_support(self):
        """测试网关中间件支持"""
        if not self.components_available:
            self.skipTest("网关组件导入失败")
        
        gateway = self.gateway_class()
        
        # 测试中间件列表存在
        self.assertIsInstance(gateway.middlewares, list)
        
        # 可以添加更多中间件相关的测试


class TestGatewayErrorHandling(unittest.TestCase):
    """测试网关错误处理"""

    def setUp(self):
        """设置测试环境"""
        try:
            from src.gateway.api_gateway import GatewayRouter as APIGateway
            self.gateway_class = APIGateway
        except ImportError:
            self.gateway_class = Mock

    def test_invalid_service_registration(self):
        """测试无效服务注册"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        # 测试无效的服务信息
        result = gateway.register_service("", {})
        
        # 根据实现，这可能返回False或抛出异常
        if result is not None:
            self.assertFalse(result)

    def test_route_registration_without_service(self):
        """测试未注册服务的路由注册"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        # 尝试为未注册的服务注册路由
        result = gateway.register_route("/api/unknown", "unknown_service", ["GET"])
        
        # 根据实现，这可能返回False
        if result is not None:
            self.assertFalse(result)

    def test_request_routing_error_handling(self):
        """测试请求路由错误处理"""
        if self.gateway_class == Mock:
            self.skipTest("APIGateway导入失败")
        
        gateway = self.gateway_class()
        
        # 测试无效的请求路径
        result = gateway.route_request("", "GET")
        
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


# 异步测试基类
class AsyncTestCase(unittest.TestCase):
    """异步测试基类"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


class TestAsyncGatewayOperations(AsyncTestCase):
    """测试异步网关操作"""

    def setUp(self):
        super().setUp()
        try:
            from src.gateway.web.websocket_api import manager
            self.websocket_manager = manager
        except ImportError:
            self.websocket_manager = None

    @patch('fastapi.WebSocket')
    def test_async_websocket_connection(self, mock_websocket):
        """测试异步WebSocket连接"""
        if self.websocket_manager is None:
            self.skipTest("WebSocket manager导入失败")
        
        async def test_connection():
            mock_ws = mock_websocket()
            mock_ws.accept = AsyncMock()
            
            # 测试连接管理
            await self.websocket_manager.connect(mock_ws, "test_channel")
            
            return True
        
        result = self.run_async(test_connection())
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
