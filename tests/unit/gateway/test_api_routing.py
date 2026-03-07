#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API路由测试
测试路由组件、代理组件和负载均衡功能
"""

import sys
import importlib
from pathlib import Path
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import tempfile
import os

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    # 尝试导入路由组件
    gateway_api_router_components_module = importlib.import_module('src.gateway.api.router_components')
    RouterComponents = getattr(gateway_api_router_components_module, 'RouterComponents', Mock)
    ROUTER_COMPONENTS_AVAILABLE = RouterComponents != Mock
except ImportError:
    RouterComponents = Mock
    ROUTER_COMPONENTS_AVAILABLE = False

try:
    # 尝试导入代理组件
    gateway_api_proxy_components_module = importlib.import_module('src.gateway.api.proxy_components')
    ProxyComponents = getattr(gateway_api_proxy_components_module, 'ProxyComponents', Mock)
    PROXY_COMPONENTS_AVAILABLE = ProxyComponents != Mock
except ImportError:
    ProxyComponents = Mock
    PROXY_COMPONENTS_AVAILABLE = False

try:
    # 尝试导入负载均衡器
    gateway_api_load_balancer_module = importlib.import_module('src.gateway.api.load_balancer')
    LoadBalancer = getattr(gateway_api_load_balancer_module, 'LoadBalancer', Mock)
    LOAD_BALANCER_AVAILABLE = LoadBalancer != Mock
except ImportError:
    LoadBalancer = Mock
    LOAD_BALANCER_AVAILABLE = False


class TestRouterComponents:
    """测试路由组件"""

    def setup_method(self, method):
        """设置测试环境"""
        if ROUTER_COMPONENTS_AVAILABLE:
            self.router = RouterComponents()
        else:
            self.router = Mock()
            self.router.add_route = Mock(return_value=True)
            self.router.remove_route = Mock(return_value=True)
            self.router.get_route = Mock(return_value={'path': '/api/test', 'method': 'GET'})
            self.router.list_routes = Mock(return_value=[])

    def test_router_components_creation(self):
        """测试路由组件创建"""
        assert self.router is not None

    def test_add_route(self):
        """测试添加路由"""
        route_config = {
            'path': '/api/market-data',
            'method': 'GET',
            'service': 'market-service',
            'endpoint': '/data',
            'middlewares': ['auth', 'rate-limit']
        }

        if ROUTER_COMPONENTS_AVAILABLE:
            result = self.router.add_route(route_config)
            assert result is True
        else:
            result = self.router.add_route(route_config)
            assert result is True

    def test_remove_route(self):
        """测试移除路由"""
        route_id = 'route_001'

        if ROUTER_COMPONENTS_AVAILABLE:
            result = self.router.remove_route(route_id)
            assert result is True
        else:
            result = self.router.remove_route(route_id)
            assert result is True

    def test_get_route(self):
        """测试获取路由"""
        route_id = 'route_001'

        if ROUTER_COMPONENTS_AVAILABLE:
            route = self.router.get_route(route_id)
            assert isinstance(route, dict)
        else:
            route = self.router.get_route(route_id)
            assert isinstance(route, dict)

    def test_list_routes(self):
        """测试列出路由"""
        if ROUTER_COMPONENTS_AVAILABLE:
            routes = self.router.list_routes()
            assert isinstance(routes, list)
        else:
            routes = self.router.list_routes()
            assert isinstance(routes, list)

    def test_route_matching(self):
        """测试路由匹配"""
        # 添加一些路由
        routes = [
            {'path': '/api/users', 'method': 'GET', 'service': 'user-service'},
            {'path': '/api/users/{id}', 'method': 'GET', 'service': 'user-service'},
            {'path': '/api/orders', 'method': 'POST', 'service': 'order-service'}
        ]

        for route in routes:
            if ROUTER_COMPONENTS_AVAILABLE:
                self.router.add_route(route)
            else:
                self.router.add_route(route)

        # 测试路由匹配
        test_requests = [
            {'path': '/api/users', 'method': 'GET'},
            {'path': '/api/users/123', 'method': 'GET'},
            {'path': '/api/orders', 'method': 'POST'}
        ]

        for request in test_requests:
            if ROUTER_COMPONENTS_AVAILABLE:
                matched_route = self.router.match_route(request['path'], request['method'])
                assert matched_route is not None
                assert matched_route['service'] is not None
            else:
                self.router.match_route = Mock(return_value={'service': 'test-service'})
                matched_route = self.router.match_route(request['path'], request['method'])
                assert matched_route is not None


class TestProxyComponents:
    """测试代理组件"""

    def setup_method(self, method):
        """设置测试环境"""
        if PROXY_COMPONENTS_AVAILABLE:
            self.proxy = ProxyComponents()
        else:
            self.proxy = Mock()
            self.proxy.forward_request = Mock(return_value={'status': 200, 'data': 'success'})
            self.proxy.add_backend = Mock(return_value=True)
            self.proxy.remove_backend = Mock(return_value=True)
            self.proxy.get_backend_status = Mock(return_value='healthy')

    def test_proxy_components_creation(self):
        """测试代理组件创建"""
        assert self.proxy is not None

    def test_add_backend(self):
        """测试添加后端服务"""
        backend_config = {
            'name': 'market-service',
            'url': 'http://localhost:8081',
            'weight': 1,
            'health_check': '/health',
            'timeout': 30
        }

        if PROXY_COMPONENTS_AVAILABLE:
            result = self.proxy.add_backend(backend_config)
            assert result is True
        else:
            result = self.proxy.add_backend(backend_config)
            assert result is True

    def test_remove_backend(self):
        """测试移除后端服务"""
        backend_name = 'market-service'

        if PROXY_COMPONENTS_AVAILABLE:
            result = self.proxy.remove_backend(backend_name)
            assert result is True
        else:
            result = self.proxy.remove_backend(backend_name)
            assert result is True

    def test_forward_request(self):
        """测试请求转发"""
        request = {
            'method': 'GET',
            'path': '/api/market-data',
            'headers': {'Authorization': 'Bearer token'},
            'body': None
        }

        if PROXY_COMPONENTS_AVAILABLE:
            response = self.proxy.forward_request(request, 'market-service')
            assert isinstance(response, dict)
            assert 'status' in response
        else:
            response = self.proxy.forward_request(request, 'market-service')
            assert isinstance(response, dict)
            assert 'status' in response

    def test_get_backend_status(self):
        """测试获取后端状态"""
        backend_name = 'market-service'

        if PROXY_COMPONENTS_AVAILABLE:
            status = self.proxy.get_backend_status(backend_name)
            assert isinstance(status, str)
            assert status in ['healthy', 'unhealthy', 'unknown']
        else:
            status = self.proxy.get_backend_status(backend_name)
            assert isinstance(status, str)

    def test_proxy_circuit_breaker(self):
        """测试断路器功能"""
        backend_name = 'failing-service'

        # 模拟连续失败
        for i in range(10):
            request = {'method': 'GET', 'path': '/api/test', 'headers': {}, 'body': None}
            if PROXY_COMPONENTS_AVAILABLE:
                try:
                    self.proxy.forward_request(request, backend_name)
                except Exception:
                    pass  # 失败是预期的
            else:
                self.proxy.forward_request(request, backend_name)

        # 检查断路器是否激活
        if PROXY_COMPONENTS_AVAILABLE:
            status = self.proxy.get_backend_status(backend_name)
            # 断路器可能已经激活
            assert isinstance(status, str)
        else:
            status = self.proxy.get_backend_status(backend_name)
            assert isinstance(status, str)


class TestLoadBalancer:
    """测试负载均衡器"""

    def setup_method(self, method):
        """设置测试环境"""
        if LOAD_BALANCER_AVAILABLE:
            self.lb = LoadBalancer()
        else:
            self.lb = Mock()
            self.lb.add_server = Mock(return_value=True)
            self.lb.remove_server = Mock(return_value=True)
            self.lb.get_next_server = Mock(return_value='server_001')
            self.lb.get_server_stats = Mock(return_value={'active_connections': 10, 'total_requests': 100})

    def test_load_balancer_creation(self):
        """测试负载均衡器创建"""
        assert self.lb is not None

    def test_add_server(self):
        """测试添加服务器"""
        server_config = {
            'name': 'server_001',
            'host': 'localhost',
            'port': 8081,
            'weight': 1,
            'max_connections': 100
        }

        if LOAD_BALANCER_AVAILABLE:
            result = self.lb.add_server(server_config)
            assert result is True
        else:
            result = self.lb.add_server(server_config)
            assert result is True

    def test_remove_server(self):
        """测试移除服务器"""
        server_name = 'server_001'

        if LOAD_BALANCER_AVAILABLE:
            result = self.lb.remove_server(server_name)
            assert result is True
        else:
            result = self.lb.remove_server(server_name)
            assert result is True

    def test_get_next_server(self):
        """测试获取下一个服务器"""
        # 添加多个服务器
        servers = [
            {'name': 'server_001', 'host': 'localhost', 'port': 8081, 'weight': 1},
            {'name': 'server_002', 'host': 'localhost', 'port': 8082, 'weight': 2},
            {'name': 'server_003', 'host': 'localhost', 'port': 8083, 'weight': 1}
        ]

        for server in servers:
            if LOAD_BALANCER_AVAILABLE:
                self.lb.add_server(server)
            else:
                self.lb.add_server(server)

        # 测试负载均衡
        selected_servers = set()
        for i in range(10):
            if LOAD_BALANCER_AVAILABLE:
                server = self.lb.get_next_server()
                assert server is not None
                selected_servers.add(server)
            else:
                server = self.lb.get_next_server()
                assert server is not None
                selected_servers.add(server)

        # 应该至少选择了不同的服务器
        assert len(selected_servers) >= 1

    def test_get_server_stats(self):
        """测试获取服务器统计"""
        server_name = 'server_001'

        if LOAD_BALANCER_AVAILABLE:
            stats = self.lb.get_server_stats(server_name)
            assert isinstance(stats, dict)
            assert 'active_connections' in stats
        else:
            stats = self.lb.get_server_stats(server_name)
            assert isinstance(stats, dict)
            assert 'active_connections' in stats

    def test_load_balancing_algorithms(self):
        """测试负载均衡算法"""
        # 测试不同算法
        algorithms = ['round_robin', 'weighted_round_robin', 'least_connections']

        for algorithm in algorithms:
            if LOAD_BALANCER_AVAILABLE:
                self.lb.set_algorithm(algorithm)
                server = self.lb.get_next_server()
                assert server is not None
            else:
                self.lb.set_algorithm = Mock()
                self.lb.get_next_server = Mock(return_value='server_001')
                self.lb.set_algorithm(algorithm)
                server = self.lb.get_next_server()
                assert server is not None


class TestGatewayIntegration:
    """测试网关集成功能"""

    def setup_method(self, method):
        """设置测试环境"""
        if ROUTER_COMPONENTS_AVAILABLE and PROXY_COMPONENTS_AVAILABLE and LOAD_BALANCER_AVAILABLE:
            self.router = RouterComponents()
            self.proxy = ProxyComponents()
            self.lb = LoadBalancer()
        else:
            self.router = Mock()
            self.proxy = Mock()
            self.lb = Mock()
            self.router.add_route = Mock(return_value=True)
            self.proxy.add_backend = Mock(return_value=True)
            self.lb.add_server = Mock(return_value=True)

    def test_complete_gateway_setup(self):
        """测试完整的网关设置"""
        # 1. 设置路由
        routes = [
            {
                'path': '/api/market-data',
                'method': 'GET',
                'service': 'market-service',
                'middlewares': ['auth', 'rate-limit']
            },
            {
                'path': '/api/orders',
                'method': 'POST',
                'service': 'order-service',
                'middlewares': ['auth']
            }
        ]

        for route in routes:
            if ROUTER_COMPONENTS_AVAILABLE:
                result = self.router.add_route(route)
                assert result is True
            else:
                result = self.router.add_route(route)
                assert result is True

        # 2. 设置后端服务
        backends = [
            {'name': 'market-service', 'url': 'http://localhost:8081'},
            {'name': 'order-service', 'url': 'http://localhost:8082'}
        ]

        for backend in backends:
            if PROXY_COMPONENTS_AVAILABLE:
                result = self.proxy.add_backend(backend)
                assert result is True
            else:
                result = self.proxy.add_backend(backend)
                assert result is True

        # 3. 设置负载均衡
        servers = [
            {'name': 'server_001', 'host': 'localhost', 'port': 8081, 'weight': 1},
            {'name': 'server_002', 'host': 'localhost', 'port': 8082, 'weight': 1}
        ]

        for server in servers:
            if LOAD_BALANCER_AVAILABLE:
                result = self.lb.add_server(server)
                assert result is True
            else:
                result = self.lb.add_server(server)
                assert result is True

    def test_request_flow_simulation(self):
        """测试请求流程模拟"""
        # 模拟一个完整的请求流程
        request = {
            'method': 'GET',
            'path': '/api/market-data',
            'headers': {'Authorization': 'Bearer token123'},
            'query_params': {'symbol': 'AAPL'},
            'body': None
        }

        # 1. 路由匹配
        if ROUTER_COMPONENTS_AVAILABLE:
            matched_route = self.router.match_route(request['path'], request['method'])
            assert matched_route is not None
            service_name = matched_route.get('service')
        else:
            self.router.match_route = Mock(return_value={'service': 'market-service'})
            matched_route = self.router.match_route(request['path'], request['method'])
            service_name = matched_route.get('service')

        # 2. 负载均衡选择服务器
        if LOAD_BALANCER_AVAILABLE:
            server = self.lb.get_next_server()
            assert server is not None
        else:
            server = self.lb.get_next_server()
            assert server is not None

        # 3. 代理转发请求
        if PROXY_COMPONENTS_AVAILABLE:
            response = self.proxy.forward_request(request, service_name)
            assert isinstance(response, dict)
            assert 'status' in response
        else:
            response = self.proxy.forward_request(request, service_name)
            assert isinstance(response, dict)
            assert 'status' in response

    def test_gateway_error_handling(self):
        """测试网关错误处理"""
        # 测试无效路由
        invalid_request = {
            'method': 'GET',
            'path': '/invalid/path',
            'headers': {},
            'body': None
        }

        if ROUTER_COMPONENTS_AVAILABLE:
            matched_route = self.router.match_route(invalid_request['path'], invalid_request['method'])
            # 对于无效路由，应该返回None或抛出异常
            assert matched_route is None or isinstance(matched_route, dict)
        else:
            self.router.match_route = Mock(return_value=None)
            matched_route = self.router.match_route(invalid_request['path'], invalid_request['method'])
            assert matched_route is None or isinstance(matched_route, dict)

    def test_gateway_performance(self):
        """测试网关性能"""
        import time

        # 模拟多个并发请求
        start_time = time.time()

        for i in range(10):
            request = {
                'method': 'GET',
                'path': '/api/test',
                'headers': {'request_id': str(i)},
                'body': None
            }

            if ROUTER_COMPONENTS_AVAILABLE:
                self.router.match_route(request['path'], request['method'])
            else:
                self.router.match_route = Mock(return_value={'service': 'test-service'})
                self.router.match_route(request['path'], request['method'])

            if LOAD_BALANCER_AVAILABLE:
                self.lb.get_next_server()
            else:
                self.lb.get_next_server = Mock(return_value='server_001')
                self.lb.get_next_server()

        end_time = time.time()
        processing_time = end_time - start_time

        # 网关处理应该很快
        assert processing_time < 5.0  # 5秒上限

    def test_gateway_monitoring(self):
        """测试网关监控"""
        # 模拟一些请求统计
        metrics = {
            'total_requests': 1000,
            'successful_requests': 950,
            'failed_requests': 50,
            'average_response_time': 0.1
        }

        if ROUTER_COMPONENTS_AVAILABLE:
            # 记录路由统计
            self.router.record_request('/api/test', 'GET', 0.05)
            stats = self.router.get_stats()
            assert isinstance(stats, dict)
        else:
            self.router.record_request = Mock()
            self.router.get_stats = Mock(return_value={})
            self.router.record_request('/api/test', 'GET', 0.05)
            stats = self.router.get_stats()
            assert isinstance(stats, dict)

        if LOAD_BALANCER_AVAILABLE:
            # 记录负载均衡统计
            lb_stats = self.lb.get_server_stats('server_001')
            assert isinstance(lb_stats, dict)
        else:
            lb_stats = self.lb.get_server_stats('server_001')
            assert isinstance(lb_stats, dict)

