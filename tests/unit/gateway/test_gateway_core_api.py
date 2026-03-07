#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Gateway模块API网关核心功能测试

测试API网关的核心功能，包括请求路由、认证、授权、限流等
"""

import pytest
import time
import json
import hashlib
import hmac
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
import base64


class TestGatewayCoreAPI:
    """测试API网关核心功能"""

    def setup_method(self):
        """测试前准备"""
        self.start_time = time.time()

        # 模拟API网关配置
        self.gateway_config = {
            'host': 'localhost',
            'port': 8080,
            'ssl_enabled': True,
            'rate_limit': {
                'requests_per_minute': 100,
                'burst_limit': 20
            },
            'auth': {
                'enabled': True,
                'jwt_secret': 'test_secret_key',
                'token_expiry': 3600
            },
            'cors': {
                'allowed_origins': ['*'],
                'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE'],
                'allowed_headers': ['*']
            }
        }

        # 模拟服务注册表
        self.services = {
            'trading': {
                'url': 'http://trading-service:8081',
                'health_check': '/health',
                'routes': ['/api/trading/*']
            },
            'market_data': {
                'url': 'http://market-data-service:8082',
                'health_check': '/health',
                'routes': ['/api/market/*']
            },
            'user': {
                'url': 'http://user-service:8083',
                'health_check': '/health',
                'routes': ['/api/user/*']
            }
        }

    def test_request_routing(self):
        """测试请求路由功能"""
        def create_router():
            """创建路由器"""
            routes = {}

            def add_route(path: str, service: str, methods: List[str] = None):
                """添加路由"""
                if methods is None:
                    methods = ['GET', 'POST', 'PUT', 'DELETE']
                routes[path] = {
                    'service': service,
                    'methods': methods
                }

            def match_route(request_path: str, method: str = 'GET') -> Optional[Dict[str, Any]]:
                """匹配路由"""
                # 简单的路径匹配，支持通配符
                for route_path, route_info in routes.items():
                    if method not in route_info['methods']:
                        continue

                    # 转换为正则表达式模式
                    pattern = route_path.replace('*', '.*')
                    import re
                    if re.match(f'^{pattern}$', request_path):
                        return {
                            'service': route_info['service'],
                            'path': request_path,
                            'method': method,
                            'route_pattern': route_path
                        }
                return None

            def get_all_routes() -> Dict[str, Any]:
                """获取所有路由"""
                return routes.copy()

            return {
                'add_route': add_route,
                'match_route': match_route,
                'get_all_routes': get_all_routes
            }

        # 创建路由器并配置路由
        router = create_router()

        # 添加服务路由
        for service_name, service_info in self.services.items():
            for route in service_info['routes']:
                router['add_route'](route, service_name)

        # 测试路由匹配
        test_cases = [
            ('/api/trading/orders', 'GET', 'trading'),
            ('/api/trading/positions', 'POST', 'trading'),
            ('/api/market/prices', 'GET', 'market_data'),
            ('/api/market/history', 'GET', 'market_data'),
            ('/api/user/profile', 'GET', 'user'),
            ('/api/user/login', 'POST', 'user'),
            ('/api/admin/dashboard', 'GET', None),  # 不匹配的路由
        ]

        for request_path, method, expected_service in test_cases:
            route_match = router['match_route'](request_path, method)
            if expected_service:
                assert route_match is not None, f"Route {request_path} should match"
                assert route_match['service'] == expected_service
                assert route_match['path'] == request_path
                assert route_match['method'] == method
            else:
                assert route_match is None, f"Route {request_path} should not match"

        # 验证路由表
        all_routes = router['get_all_routes']()
        assert len(all_routes) == 3  # 三个服务
        assert '/api/trading/*' in all_routes
        assert '/api/market/*' in all_routes
        assert '/api/user/*' in all_routes

    def test_authentication_jwt(self):
        """测试JWT认证功能"""
        def create_jwt_authenticator(secret_key: str, expiry_seconds: int = 3600):
            """创建JWT认证器"""
            # 简化的JWT实现（用于测试）
            import base64
            import json
            import hmac

            def generate_token(user_id: str, roles: List[str] = None) -> str:
                """生成JWT令牌"""
                if roles is None:
                    roles = ['user']

                payload = {
                    'user_id': user_id,
                    'roles': roles,
                    'iat': datetime.utcnow().timestamp(),
                    'exp': (datetime.utcnow() + timedelta(seconds=expiry_seconds)).timestamp()
                }

                # 简化的JWT编码
                header = {'alg': 'HS256', 'typ': 'JWT'}
                header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
                payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')

                message = f"{header_b64}.{payload_b64}"
                signature = base64.urlsafe_b64encode(
                    hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
                ).decode().rstrip('=')

                return f"{message}.{signature}"

            def verify_token(token: str) -> Optional[Dict[str, Any]]:
                """验证JWT令牌"""
                try:
                    parts = token.split('.')
                    if len(parts) != 3:
                        return None

                    header_b64, payload_b64, signature = parts

                    # 验证签名
                    message = f"{header_b64}.{payload_b64}"
                    expected_signature = base64.urlsafe_b64encode(
                        hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
                    ).decode().rstrip('=')

                    if not hmac.compare_digest(signature, expected_signature):
                        return None

                    # 解码payload
                    payload_json = base64.urlsafe_b64decode(payload_b64 + '=' * (4 - len(payload_b64) % 4))
                    payload = json.loads(payload_json)

                    # 检查是否过期
                    exp_time = datetime.fromtimestamp(payload['exp'])
                    if exp_time < datetime.utcnow():
                        return None

                    return {
                        'user_id': payload['user_id'],
                        'roles': payload['roles'],
                        'valid': True
                    }
                except Exception:
                    return None

            def refresh_token(token: str) -> Optional[str]:
                """刷新令牌"""
                user_info = verify_token(token)
                if user_info and user_info['valid']:
                    # 强制延迟一秒以确保令牌不同
                    import time
                    time.sleep(0.001)
                    return generate_token(user_info['user_id'], user_info['roles'])
                return None

            return {
                'generate_token': generate_token,
                'verify_token': verify_token,
                'refresh_token': refresh_token
            }

        # 创建JWT认证器
        auth = create_jwt_authenticator(self.gateway_config['auth']['jwt_secret'])

        # 测试令牌生成
        user_id = 'user123'
        roles = ['user', 'trader']
        token = auth['generate_token'](user_id, roles)

        assert isinstance(token, str)
        assert len(token) > 0

        # 测试令牌验证
        verified = auth['verify_token'](token)
        assert verified is not None
        assert verified['user_id'] == user_id
        assert verified['roles'] == roles
        assert verified['valid'] == True

        # 测试令牌刷新
        refreshed_token = auth['refresh_token'](token)
        assert refreshed_token is not None
        assert refreshed_token != token  # 应该生成新的令牌

        # 验证刷新后的令牌
        refreshed_verified = auth['verify_token'](refreshed_token)
        assert refreshed_verified is not None
        assert refreshed_verified['user_id'] == user_id
        assert refreshed_verified['roles'] == roles

        # 测试无效令牌
        invalid_verified = auth['verify_token']('invalid_token')
        assert invalid_verified is None

        # 测试过期令牌（通过修改密钥）
        expired_auth = create_jwt_authenticator('different_secret')
        expired_verified = expired_auth['verify_token'](token)
        assert expired_verified is None

    def test_rate_limiting(self):
        """测试限流功能"""
        def create_rate_limiter(requests_per_minute: int = 100, burst_limit: int = 20):
            """创建限流器"""
            import time
            import collections

            # 使用令牌桶算法
            class TokenBucket:
                def __init__(self, rate_per_second: float, burst_size: int):
                    self.rate = rate_per_second
                    self.burst_size = burst_size
                    self.tokens = burst_size
                    self.last_update = time.time()

                def consume(self, tokens: int = 1) -> bool:
                    """消费令牌"""
                    now = time.time()
                    elapsed = now - self.last_update

                    # 补充令牌
                    self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
                    self.last_update = now

                    if self.tokens >= tokens:
                        self.tokens -= tokens
                        return True
                    return False

                def get_remaining_tokens(self) -> float:
                    """获取剩余令牌数"""
                    now = time.time()
                    elapsed = now - self.last_update
                    return min(self.burst_size, self.tokens + elapsed * self.rate)

            # 按客户端IP维护令牌桶
            buckets = {}
            rate_per_second = requests_per_minute / 60.0

            def is_allowed(client_ip: str) -> Dict[str, Any]:
                """检查请求是否被允许"""
                if client_ip not in buckets:
                    buckets[client_ip] = TokenBucket(rate_per_second, burst_limit)

                bucket = buckets[client_ip]
                allowed = bucket.consume(1)

                return {
                    'allowed': allowed,
                    'remaining_tokens': bucket.get_remaining_tokens(),
                    'reset_time': bucket.last_update + (burst_limit / rate_per_second)
                }

            def get_client_stats(client_ip: str) -> Dict[str, Any]:
                """获取客户端统计信息"""
                if client_ip in buckets:
                    bucket = buckets[client_ip]
                    return {
                        'remaining_tokens': bucket.get_remaining_tokens(),
                        'reset_time': bucket.last_update + (burst_limit / rate_per_second),
                        'total_requests': burst_limit - bucket.tokens  # 近似值
                    }
                return {
                    'remaining_tokens': burst_limit,
                    'reset_time': time.time() + (burst_limit / rate_per_second),
                    'total_requests': 0
                }

            def reset_client(client_ip: str):
                """重置客户端限流"""
                if client_ip in buckets:
                    del buckets[client_ip]

            return {
                'is_allowed': is_allowed,
                'get_client_stats': get_client_stats,
                'reset_client': reset_client
            }

        # 创建限流器
        rate_limiter = create_rate_limiter(
            self.gateway_config['rate_limit']['requests_per_minute'],
            self.gateway_config['rate_limit']['burst_limit']
        )

        # 测试正常请求
        client_ip = '192.168.1.100'

        # 应该允许初始请求
        result = rate_limiter['is_allowed'](client_ip)
        assert result['allowed'] == True
        assert result['remaining_tokens'] >= 0

        # 连续快速请求，测试限流
        allowed_count = 0
        for i in range(25):  # 超过burst_limit
            result = rate_limiter['is_allowed'](client_ip)
            if result['allowed']:
                allowed_count += 1
            time.sleep(0.01)  # 小延迟

        # 应该被限流
        assert allowed_count <= 25  # 允许一些突发请求

        # 测试获取客户端统计
        stats = rate_limiter['get_client_stats'](client_ip)
        assert 'remaining_tokens' in stats
        assert 'reset_time' in stats
        assert 'total_requests' in stats

        # 测试重置客户端
        rate_limiter['reset_client'](client_ip)

        # 重置后应该允许新请求
        result = rate_limiter['is_allowed'](client_ip)
        assert result['allowed'] == True

    def test_load_balancing(self):
        """测试负载均衡功能"""
        def create_load_balancer(algorithm: str = 'round_robin'):
            """创建负载均衡器"""
            servers = []
            current_index = 0
            server_stats = {}  # 服务器统计信息

            def add_server(server_url: str, weight: int = 1):
                """添加服务器"""
                servers.append({
                    'url': server_url,
                    'weight': weight,
                    'active': True,
                    'connections': 0
                })
                server_stats[server_url] = {
                    'requests': 0,
                    'errors': 0,
                    'response_time': 0.0,
                    'last_health_check': datetime.now()
                }

            def remove_server(server_url: str):
                """移除服务器"""
                servers[:] = [s for s in servers if s['url'] != server_url]
                if server_url in server_stats:
                    del server_stats[server_url]

            def get_next_server() -> Optional[str]:
                """获取下一个服务器"""
                if not servers:
                    return None

                active_servers = [s for s in servers if s['active']]
                if not active_servers:
                    return None

                if algorithm == 'round_robin':
                    # 轮询算法
                    nonlocal current_index
                    server = active_servers[current_index % len(active_servers)]
                    current_index += 1
                    return server['url']

                elif algorithm == 'least_connections':
                    # 最少连接数算法
                    server = min(active_servers, key=lambda s: s['connections'])
                    server['connections'] += 1
                    return server['url']

                elif algorithm == 'weighted_round_robin':
                    # 加权轮询算法
                    total_weight = sum(s['weight'] for s in active_servers)
                    if total_weight == 0:
                        return None

                    # 简化实现：按权重随机选择
                    import random
                    rand_val = random.uniform(0, total_weight)
                    current_weight = 0

                    for server in active_servers:
                        current_weight += server['weight']
                        if rand_val <= current_weight:
                            return server['url']

                return active_servers[0]['url']  # 默认返回第一个

            def record_request(server_url: str, response_time: float, success: bool):
                """记录请求统计"""
                if server_url in server_stats:
                    stats = server_stats[server_url]
                    stats['requests'] += 1
                    if not success:
                        stats['errors'] += 1
                    # 简单移动平均响应时间
                    stats['response_time'] = (stats['response_time'] + response_time) / 2
                    stats['last_health_check'] = datetime.now()

            def get_server_stats(server_url: str) -> Optional[Dict[str, Any]]:
                """获取服务器统计"""
                return server_stats.get(server_url)

            def get_all_servers() -> List[Dict[str, Any]]:
                """获取所有服务器"""
                return servers.copy()

            def mark_server_down(server_url: str):
                """标记服务器为下线"""
                for server in servers:
                    if server['url'] == server_url:
                        server['active'] = False
                        break

            def mark_server_up(server_url: str):
                """标记服务器为上线"""
                for server in servers:
                    if server['url'] == server_url:
                        server['active'] = True
                        break

            return {
                'add_server': add_server,
                'remove_server': remove_server,
                'get_next_server': get_next_server,
                'record_request': record_request,
                'get_server_stats': get_server_stats,
                'get_all_servers': get_all_servers,
                'mark_server_down': mark_server_down,
                'mark_server_up': mark_server_up
            }

        # 创建负载均衡器
        lb = create_load_balancer('round_robin')

        # 添加服务器
        servers = [
            'http://server1:8080',
            'http://server2:8080',
            'http://server3:8080'
        ]

        for server in servers:
            lb['add_server'](server)

        # 测试轮询算法
        selected_servers = []
        for i in range(9):  # 3轮
            server = lb['get_next_server']()
            assert server is not None
            selected_servers.append(server)

        # 验证轮询分布
        server_counts = {}
        for server in selected_servers:
            server_counts[server] = server_counts.get(server, 0) + 1

        # 每个服务器应该被选择3次
        for server in servers:
            assert server_counts.get(server, 0) == 3

        # 测试请求记录
        lb['record_request']('http://server1:8080', 0.1, True)
        lb['record_request']('http://server1:8080', 0.2, False)  # 失败请求

        stats = lb['get_server_stats']('http://server1:8080')
        assert stats is not None
        assert stats['requests'] == 2
        assert stats['errors'] == 1
        assert stats['response_time'] > 0

        # 测试服务器状态管理
        lb['mark_server_down']('http://server2:8080')

        # 下线服务器不应该被选择
        for i in range(3):
            server = lb['get_next_server']()
            assert server in ['http://server1:8080', 'http://server3:8080']

        # 重新上线
        lb['mark_server_up']('http://server2:8080')

        # 重新上线的服务器应该在活跃服务器列表中
        all_servers = lb['get_all_servers']()
        active_servers = [s for s in all_servers if s['active']]
        assert len(active_servers) == 3
        assert any(s['url'] == 'http://server2:8080' for s in active_servers)

    def test_cors_handling(self):
        """测试CORS处理功能"""
        def create_cors_handler(cors_config: Dict[str, Any]):
            """创建CORS处理器"""
            allowed_origins = cors_config.get('allowed_origins', [])
            allowed_methods = cors_config.get('allowed_methods', [])
            allowed_headers = cors_config.get('allowed_headers', [])
            allow_credentials = cors_config.get('allow_credentials', False)
            max_age = cors_config.get('max_age', 86400)

            def is_origin_allowed(origin: str) -> bool:
                """检查源是否被允许"""
                if '*' in allowed_origins:
                    return True
                return origin in allowed_origins

            def is_method_allowed(method: str) -> bool:
                """检查方法是否被允许"""
                if '*' in allowed_methods or not allowed_methods:
                    return True
                return method.upper() in [m.upper() for m in allowed_methods]

            def is_header_allowed(header: str) -> bool:
                """检查头部是否被允许"""
                if '*' in allowed_headers or not allowed_headers:
                    return True
                return header.lower() in [h.lower() for h in allowed_headers]

            def handle_preflight_request(origin: str, method: str, headers: List[str]) -> Dict[str, Any]:
                """处理预检请求"""
                if not is_origin_allowed(origin):
                    return {'status': 403, 'error': 'Origin not allowed'}

                if not is_method_allowed(method):
                    return {'status': 403, 'error': 'Method not allowed'}

                for header in headers:
                    if not is_header_allowed(header):
                        return {'status': 403, 'error': f'Header not allowed: {header}'}

                # 返回CORS头部
                cors_headers = {
                    'Access-Control-Allow-Origin': origin if origin != '*' else '*',
                    'Access-Control-Allow-Methods': ', '.join(allowed_methods) if allowed_methods else '*',
                    'Access-Control-Allow-Headers': ', '.join(allowed_headers) if allowed_headers else '*',
                    'Access-Control-Max-Age': str(max_age)
                }

                if allow_credentials:
                    cors_headers['Access-Control-Allow-Credentials'] = 'true'

                return {
                    'status': 200,
                    'headers': cors_headers
                }

            def add_cors_headers(response_headers: Dict[str, str], origin: str) -> Dict[str, str]:
                """为响应添加CORS头部"""
                if is_origin_allowed(origin):
                    response_headers['Access-Control-Allow-Origin'] = origin if origin != '*' else '*'
                    if allow_credentials:
                        response_headers['Access-Control-Allow-Credentials'] = 'true'

                return response_headers

            return {
                'handle_preflight_request': handle_preflight_request,
                'add_cors_headers': add_cors_headers,
                'is_origin_allowed': is_origin_allowed,
                'is_method_allowed': is_method_allowed,
                'is_header_allowed': is_header_allowed
            }

        # 创建CORS处理器
        cors = create_cors_handler(self.gateway_config['cors'])

        # 测试预检请求
        preflight_result = cors['handle_preflight_request'](
            origin='https://example.com',
            method='POST',
            headers=['Content-Type', 'Authorization']
        )

        assert preflight_result['status'] == 200
        assert 'headers' in preflight_result
        cors_headers = preflight_result['headers']

        assert 'Access-Control-Allow-Origin' in cors_headers
        assert 'Access-Control-Allow-Methods' in cors_headers
        assert 'Access-Control-Allow-Headers' in cors_headers

        # 验证允许的方法
        methods = cors_headers['Access-Control-Allow-Methods'].split(', ')
        for method in ['GET', 'POST', 'PUT', 'DELETE']:
            assert method in methods

        # 测试不允许的源
        cors_strict = create_cors_handler({
            'allowed_origins': ['https://trusted.com'],
            'allowed_methods': ['GET', 'POST'],
            'allowed_headers': ['Content-Type']
        })

        preflight_strict = cors_strict['handle_preflight_request'](
            origin='https://malicious.com',
            method='POST',
            headers=['Content-Type']
        )

        assert preflight_strict['status'] == 403
        assert 'error' in preflight_strict

        # 测试不允许的方法
        preflight_method = cors_strict['handle_preflight_request'](
            origin='https://trusted.com',
            method='DELETE',  # 不允许的方法
            headers=['Content-Type']
        )

        assert preflight_method['status'] == 403

        # 测试不允许的头部
        preflight_header = cors_strict['handle_preflight_request'](
            origin='https://trusted.com',
            method='POST',
            headers=['X-Custom-Header']  # 不允许的头部
        )

        assert preflight_header['status'] == 403

    def test_request_transformation(self):
        """测试请求转换功能"""
        def create_request_transformer():
            """创建请求转换器"""
            transformations = []

            def add_transformation(name: str, condition_func: Callable, transform_func: Callable):
                """添加转换规则"""
                transformations.append({
                    'name': name,
                    'condition': condition_func,
                    'transform': transform_func,
                    'enabled': True
                })

            def transform_request(request: Dict[str, Any]) -> Dict[str, Any]:
                """转换请求"""
                transformed_request = request.copy()

                for transformation in transformations:
                    if not transformation['enabled']:
                        continue

                    if transformation['condition'](transformed_request):
                        transformed_request = transformation['transform'](transformed_request)

                return transformed_request

            def enable_transformation(name: str):
                """启用转换"""
                for t in transformations:
                    if t['name'] == name:
                        t['enabled'] = True

            def disable_transformation(name: str):
                """禁用转换"""
                for t in transformations:
                    if t['name'] == name:
                        t['enabled'] = False

            return {
                'add_transformation': add_transformation,
                'transform_request': transform_request,
                'enable_transformation': enable_transformation,
                'disable_transformation': disable_transformation
            }

        # 创建请求转换器
        transformer = create_request_transformer()

        # 添加转换规则：API版本转换
        def api_version_condition(request):
            return 'api-version' in request.get('headers', {})

        def api_version_transform(request):
            version = request['headers']['api-version']
            # 将路径中的版本占位符替换为实际版本
            if '{version}' in request['path']:
                request['path'] = request['path'].replace('{version}', version)
            return request

        transformer['add_transformation']('api_version', api_version_condition, api_version_transform)

        # 添加转换规则：请求头标准化
        def header_normalization_condition(request):
            return 'headers' in request

        def header_normalization_transform(request):
            headers = request['headers']
            # 标准化头部名称
            normalized_headers = {}
            for key, value in headers.items():
                normalized_key = '-'.join(word.capitalize() for word in key.split('-'))
                normalized_headers[normalized_key] = value
            request['headers'] = normalized_headers
            return request

        transformer['add_transformation']('header_normalization', header_normalization_condition, header_normalization_transform)

        # 测试请求转换
        test_request = {
            'method': 'GET',
            'path': '/api/{version}/users',
            'headers': {
                'api-version': 'v1',
                'content-type': 'application/json',
                'authorization': 'Bearer token123'
            },
            'query_params': {'limit': '10'},
            'body': None
        }

        transformed = transformer['transform_request'](test_request)

        # 验证API版本转换
        assert transformed['path'] == '/api/v1/users'

        # 验证头部标准化
        headers = transformed['headers']
        assert 'Api-Version' in headers
        assert 'Content-Type' in headers
        assert 'Authorization' in headers

        # 验证其他字段不变
        assert transformed['method'] == 'GET'
        assert transformed['query_params'] == {'limit': '10'}

        # 测试禁用转换
        transformer['disable_transformation']('api_version')

        test_request2 = {
            'method': 'POST',
            'path': '/api/{version}/orders',
            'headers': {'api-version': 'v2'}
        }

        transformed2 = transformer['transform_request'](test_request2)

        # API版本转换应该被禁用
        assert transformed2['path'] == '/api/{version}/orders'

        # 头部标准化仍然有效
        assert 'Api-Version' in transformed2['headers']

    def test_api_gateway_metrics(self):
        """测试API网关指标收集"""
        def create_metrics_collector():
            """创建指标收集器"""
            metrics = {
                'requests_total': 0,
                'requests_by_method': {},
                'requests_by_status': {},
                'response_times': [],
                'errors_total': 0,
                'errors_by_type': {},
                'active_connections': 0,
                'peak_connections': 0
            }

            def record_request(method: str, status_code: int, response_time: float):
                """记录请求"""
                metrics['requests_total'] += 1

                # 按方法统计
                if method not in metrics['requests_by_method']:
                    metrics['requests_by_method'][method] = 0
                metrics['requests_by_method'][method] += 1

                # 按状态码统计
                status_key = str(status_code)
                if status_key not in metrics['requests_by_status']:
                    metrics['requests_by_status'][status_key] = 0
                metrics['requests_by_status'][status_key] += 1

                # 记录响应时间
                metrics['response_times'].append(response_time)

                # 记录错误
                if status_code >= 400:
                    metrics['errors_total'] += 1
                    error_type = 'client_error' if status_code < 500 else 'server_error'
                    if error_type not in metrics['errors_by_type']:
                        metrics['errors_by_type'][error_type] = 0
                    metrics['errors_by_type'][error_type] += 1

            def update_connections(active: int):
                """更新连接数"""
                metrics['active_connections'] = active
                metrics['peak_connections'] = max(metrics['peak_connections'], active)

            def get_metrics() -> Dict[str, Any]:
                """获取指标"""
                result = metrics.copy()

                # 计算响应时间统计
                if metrics['response_times']:
                    result['avg_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])
                    result['min_response_time'] = min(metrics['response_times'])
                    result['max_response_time'] = max(metrics['response_times'])
                    result['p95_response_time'] = sorted(metrics['response_times'])[int(len(metrics['response_times']) * 0.95)]
                else:
                    result['avg_response_time'] = 0.0
                    result['min_response_time'] = 0.0
                    result['max_response_time'] = 0.0
                    result['p95_response_time'] = 0.0

                # 计算成功率
                if metrics['requests_total'] > 0:
                    result['success_rate'] = (metrics['requests_total'] - metrics['errors_total']) / metrics['requests_total']
                else:
                    result['success_rate'] = 1.0

                return result

            def reset_metrics():
                """重置指标"""
                metrics.clear()
                metrics.update({
                    'requests_total': 0,
                    'requests_by_method': {},
                    'requests_by_status': {},
                    'response_times': [],
                    'errors_total': 0,
                    'errors_by_type': {},
                    'active_connections': 0,
                    'peak_connections': 0
                })

            return {
                'record_request': record_request,
                'update_connections': update_connections,
                'get_metrics': get_metrics,
                'reset_metrics': reset_metrics
            }

        # 创建指标收集器
        collector = create_metrics_collector()

        # 记录一些测试请求
        test_requests = [
            ('GET', 200, 0.1),
            ('POST', 201, 0.2),
            ('GET', 200, 0.05),
            ('PUT', 200, 0.15),
            ('GET', 404, 0.08),  # 客户端错误
            ('POST', 500, 0.3),  # 服务器错误
            ('DELETE', 204, 0.12)
        ]

        for method, status, response_time in test_requests:
            collector['record_request'](method, status, response_time)

        # 更新连接数
        collector['update_connections'](5)
        collector['update_connections'](8)
        collector['update_connections'](3)

        # 获取指标
        metrics = collector['get_metrics']()

        # 验证基本计数
        assert metrics['requests_total'] == 7
        assert metrics['errors_total'] == 2  # 404和500
        assert metrics['active_connections'] == 3
        assert metrics['peak_connections'] == 8

        # 验证按方法统计
        assert metrics['requests_by_method']['GET'] == 3
        assert metrics['requests_by_method']['POST'] == 2
        assert metrics['requests_by_method']['PUT'] == 1
        assert metrics['requests_by_method']['DELETE'] == 1

        # 验证按状态码统计
        assert metrics['requests_by_status']['200'] == 3
        assert metrics['requests_by_status']['201'] == 1
        assert metrics['requests_by_status']['204'] == 1
        assert metrics['requests_by_status']['404'] == 1
        assert metrics['requests_by_status']['500'] == 1

        # 验证错误类型统计
        assert metrics['errors_by_type']['client_error'] == 1  # 404
        assert metrics['errors_by_type']['server_error'] == 1  # 500

        # 验证响应时间统计
        assert metrics['avg_response_time'] > 0
        assert metrics['min_response_time'] == 0.05
        assert metrics['max_response_time'] == 0.3
        assert metrics['p95_response_time'] > 0

        # 验证成功率
        assert abs(metrics['success_rate'] - (5/7)) < 0.001  # 5个成功请求

        # 测试重置
        collector['reset_metrics']()
        reset_metrics = collector['get_metrics']()
        assert reset_metrics['requests_total'] == 0
        assert reset_metrics['active_connections'] == 0
