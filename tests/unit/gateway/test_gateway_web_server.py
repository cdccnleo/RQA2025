#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Gateway模块Web服务器组件测试

测试Web服务器的核心功能，包括HTTP处理、WebSocket支持、静态文件服务等
"""

import pytest
import time
import json
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import tempfile
import os


class TestGatewayWebServer:
    """测试Web服务器组件"""

    def setup_method(self):
        """测试前准备"""
        self.start_time = time.time()

        # 模拟Web服务器配置
        self.web_config = {
            'host': 'localhost',
            'port': 8080,
            'ssl_enabled': True,
            'ssl_cert': '/path/to/cert.pem',
            'ssl_key': '/path/to/key.pem',
            'static_dir': '/path/to/static',
            'template_dir': '/path/to/templates',
            'max_connections': 1000,
            'timeout': 30,
            'cors_enabled': True,
            'websocket_enabled': True,
            'compression_enabled': True
        }

        # 模拟路由表
        self.routes = {
            'GET /api/health': 'health_handler',
            'POST /api/users': 'create_user_handler',
            'GET /api/users/{id}': 'get_user_handler',
            'PUT /api/users/{id}': 'update_user_handler',
            'DELETE /api/users/{id}': 'delete_user_handler'
        }

    def test_http_request_handling(self):
        """测试HTTP请求处理"""
        def create_http_server():
            """创建HTTP服务器"""
            request_log = []
            response_cache = {}

            def handle_request(method: str, path: str, headers: Dict[str, str],
                             body: Optional[str] = None) -> Dict[str, Any]:
                """处理HTTP请求"""
                request_info = {
                    'method': method,
                    'path': path,
                    'headers': headers,
                    'body': body,
                    'timestamp': datetime.now().isoformat()
                }
                request_log.append(request_info)

                # 路由匹配和处理
                route_key = f"{method} {path}"
                if route_key in self.routes:
                    handler_name = self.routes[route_key]

                    # 模拟不同处理器的响应
                    if handler_name == 'health_handler':
                        return {
                            'status': 200,
                            'headers': {'Content-Type': 'application/json'},
                            'body': json.dumps({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
                        }
                    elif handler_name == 'create_user_handler':
                        # 模拟用户创建
                        user_data = json.loads(body) if body else {}
                        user_id = f"user_{len(request_log)}"
                        return {
                            'status': 201,
                            'headers': {'Content-Type': 'application/json', 'Location': f'/api/users/{user_id}'},
                            'body': json.dumps({'id': user_id, 'created': True, **user_data})
                        }
                    elif 'get_user_handler' in handler_name:
                        # 从路径中提取用户ID
                        user_id = path.split('/')[-1]
                        return {
                            'status': 200,
                            'headers': {'Content-Type': 'application/json'},
                            'body': json.dumps({'id': user_id, 'name': f'User {user_id}', 'active': True})
                        }
                    elif 'update_user_handler' in handler_name:
                        user_id = path.split('/')[-1]
                        return {
                            'status': 200,
                            'headers': {'Content-Type': 'application/json'},
                            'body': json.dumps({'id': user_id, 'updated': True})
                        }
                    elif 'delete_user_handler' in handler_name:
                        user_id = path.split('/')[-1]
                        return {
                            'status': 204,
                            'headers': {},
                            'body': ''
                        }

                # 处理静态文件请求
                if path.startswith('/static/'):
                    filename = path[len('/static/'):]
                    if filename in ['style.css', 'app.js', 'logo.png']:
                        content_type = {
                            'css': 'text/css',
                            'js': 'application/javascript',
                            'png': 'image/png'
                        }.get(filename.split('.')[-1], 'text/plain')

                        return {
                            'status': 200,
                            'headers': {'Content-Type': content_type, 'Cache-Control': 'max-age=3600'},
                            'body': f'/* Content of {filename} */'
                        }

                # 404处理
                return {
                    'status': 404,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({'error': 'Not Found', 'path': path})
                }

            def get_request_log() -> List[Dict[str, Any]]:
                """获取请求日志"""
                return request_log.copy()

            def clear_cache():
                """清空缓存"""
                response_cache.clear()

            def get_server_stats() -> Dict[str, Any]:
                """获取服务器统计"""
                return {
                    'total_requests': len(request_log),
                    'unique_paths': len(set(r['path'] for r in request_log)),
                    'methods_used': list(set(r['method'] for r in request_log)),
                    'avg_response_time': 0.05,  # 模拟值
                    'uptime': time.time() - self.start_time
                }

            return {
                'handle_request': handle_request,
                'get_request_log': get_request_log,
                'clear_cache': clear_cache,
                'get_server_stats': get_server_stats
            }

        # 创建HTTP服务器
        server = create_http_server()

        # 测试健康检查请求
        response = server['handle_request']('GET', '/api/health', {'Accept': 'application/json'})
        assert response['status'] == 200
        assert 'Content-Type' in response['headers']
        health_data = json.loads(response['body'])
        assert health_data['status'] == 'healthy'

        # 测试用户创建请求
        user_data = {'name': 'John Doe', 'email': 'john@example.com'}
        response = server['handle_request']('POST', '/api/users',
                                          {'Content-Type': 'application/json'},
                                          json.dumps(user_data))
        assert response['status'] == 201
        assert 'Location' in response['headers']
        created_user = json.loads(response['body'])
        assert created_user['created'] == True
        assert created_user['name'] == 'John Doe'

        # 测试获取用户请求
        response = server['handle_request']('GET', '/api/users/user_123',
                                          {'Accept': 'application/json'})
        assert response['status'] == 200
        user_info = json.loads(response['body'])
        assert user_info['id'] == 'user_123'
        assert user_info['active'] == True

        # 测试静态文件请求
        response = server['handle_request']('GET', '/static/style.css',
                                          {'Accept': 'text/css'})
        assert response['status'] == 200
        assert response['headers']['Content-Type'] == 'text/css'
        assert 'Cache-Control' in response['headers']

        # 测试404请求
        response = server['handle_request']('GET', '/nonexistent/path',
                                          {'Accept': 'application/json'})
        assert response['status'] == 404
        error_data = json.loads(response['body'])
        assert error_data['error'] == 'Not Found'

        # 验证请求日志
        request_log = server['get_request_log']()
        assert len(request_log) == 5  # 5个请求

        # 验证服务器统计
        stats = server['get_server_stats']()
        assert stats['total_requests'] == 5
        assert stats['unique_paths'] == 5
        assert 'GET' in stats['methods_used']
        assert 'POST' in stats['methods_used']

    def test_websocket_connection_management(self):
        """测试WebSocket连接管理"""
        def create_websocket_manager():
            """创建WebSocket连接管理器"""
            connections = {}
            message_log = []
            connection_id_counter = 1

            def handle_connection(client_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
                """处理WebSocket连接"""
                nonlocal connection_id_counter

                # 验证WebSocket握手
                if 'Upgrade' not in headers or headers['Upgrade'].lower() != 'websocket':
                    return {'status': 400, 'error': 'Not a WebSocket request'}

                if 'Sec-WebSocket-Key' not in headers:
                    return {'status': 400, 'error': 'Missing WebSocket key'}

                # 生成连接ID
                connection_id = f"conn_{connection_id_counter}"
                connection_id_counter += 1

                # 存储连接信息
                connections[connection_id] = {
                    'client_id': client_id,
                    'connected_at': datetime.now(),
                    'last_activity': datetime.now(),
                    'subscribed_channels': set(),
                    'message_count': 0
                }

                return {
                    'status': 101,
                    'connection_id': connection_id,
                    'headers': {
                        'Upgrade': 'websocket',
                        'Connection': 'Upgrade',
                        'Sec-WebSocket-Accept': 'mock_accept_key'
                    }
                }

            def handle_message(connection_id: str, message_type: str, payload: Any) -> Dict[str, Any]:
                """处理WebSocket消息"""
                if connection_id not in connections:
                    return {'status': 'error', 'error': 'Connection not found'}

                conn = connections[connection_id]
                conn['last_activity'] = datetime.now()
                conn['message_count'] += 1

                # 记录消息
                message_log.append({
                    'connection_id': connection_id,
                    'type': message_type,
                    'payload': payload,
                    'timestamp': datetime.now().isoformat()
                })

                # 处理不同类型的消息
                if message_type == 'subscribe':
                    channel = payload.get('channel')
                    if channel:
                        conn['subscribed_channels'].add(channel)
                        return {'status': 'success', 'action': 'subscribed', 'channel': channel}

                elif message_type == 'unsubscribe':
                    channel = payload.get('channel')
                    if channel and channel in conn['subscribed_channels']:
                        conn['subscribed_channels'].remove(channel)
                        return {'status': 'success', 'action': 'unsubscribed', 'channel': channel}

                elif message_type == 'ping':
                    return {'status': 'success', 'action': 'pong', 'timestamp': datetime.now().isoformat()}

                elif message_type == 'broadcast':
                    # 广播消息给所有订阅者
                    channel = payload.get('channel')
                    message = payload.get('message')
                    if channel and message:
                        broadcast_count = 0
                        for conn_id, conn_info in connections.items():
                            if channel in conn_info['subscribed_channels']:
                                # 这里应该发送消息，但为了测试简化
                                broadcast_count += 1
                        return {'status': 'success', 'action': 'broadcast', 'recipients': broadcast_count}

                return {'status': 'success', 'action': 'received'}

            def disconnect(connection_id: str) -> bool:
                """断开连接"""
                if connection_id in connections:
                    del connections[connection_id]
                    return True
                return False

            def get_connection_stats() -> Dict[str, Any]:
                """获取连接统计"""
                return {
                    'total_connections': len(connections),
                    'active_connections': len([c for c in connections.values() if (datetime.now() - c['last_activity']).seconds < 300]),
                    'total_messages': sum(c['message_count'] for c in connections.values()),
                    'subscribed_channels': len(set(channel for c in connections.values() for channel in c['subscribed_channels']))
                }

            def broadcast_to_channel(channel: str, message: Any) -> int:
                """广播消息到频道"""
                recipients = 0
                for conn_id, conn_info in connections.items():
                    if channel in conn_info['subscribed_channels']:
                        # 模拟发送消息
                        message_log.append({
                            'connection_id': conn_id,
                            'type': 'broadcast',
                            'payload': {'channel': channel, 'message': message},
                            'timestamp': datetime.now().isoformat()
                        })
                        recipients += 1
                return recipients

            return {
                'handle_connection': handle_connection,
                'handle_message': handle_message,
                'disconnect': disconnect,
                'get_connection_stats': get_connection_stats,
                'broadcast_to_channel': broadcast_to_channel,
                'get_message_log': lambda: message_log.copy()
            }

        # 创建WebSocket管理器
        ws_manager = create_websocket_manager()

        # 测试WebSocket连接
        headers = {
            'Upgrade': 'websocket',
            'Connection': 'Upgrade',
            'Sec-WebSocket-Key': 'dGhlIHNhbXBsZSBub25jZQ==',
            'Sec-WebSocket-Version': '13'
        }

        result = ws_manager['handle_connection']('client_123', headers)
        assert result['status'] == 101
        assert 'connection_id' in result
        connection_id = result['connection_id']

        # 测试订阅消息
        subscribe_msg = {'channel': 'market_data'}
        response = ws_manager['handle_message'](connection_id, 'subscribe', subscribe_msg)
        assert response['status'] == 'success'
        assert response['action'] == 'subscribed'
        assert response['channel'] == 'market_data'

        # 测试ping消息
        ping_response = ws_manager['handle_message'](connection_id, 'ping', {})
        assert ping_response['status'] == 'success'
        assert ping_response['action'] == 'pong'

        # 测试广播消息
        broadcast_payload = {'channel': 'market_data', 'message': {'price': 150.0}}
        broadcast_response = ws_manager['handle_message'](connection_id, 'broadcast', broadcast_payload)
        assert broadcast_response['status'] == 'success'
        assert broadcast_response['action'] == 'broadcast'
        assert broadcast_response['recipients'] == 1

        # 测试频道广播
        recipients = ws_manager['broadcast_to_channel']('market_data', {'update': 'price changed'})
        assert recipients == 1

        # 测试取消订阅
        unsubscribe_msg = {'channel': 'market_data'}
        unsub_response = ws_manager['handle_message'](connection_id, 'unsubscribe', unsubscribe_msg)
        assert unsub_response['status'] == 'success'
        assert unsub_response['action'] == 'unsubscribed'

        # 验证连接统计
        stats = ws_manager['get_connection_stats']()
        assert stats['total_connections'] == 1
        assert stats['active_connections'] == 1
        assert stats['total_messages'] >= 4  # subscribe, ping, broadcast, unsubscribe

        # 测试断开连接
        disconnected = ws_manager['disconnect'](connection_id)
        assert disconnected == True

        # 验证连接已断开
        final_stats = ws_manager['get_connection_stats']()
        assert final_stats['total_connections'] == 0

    def test_static_file_serving(self):
        """测试静态文件服务"""
        def create_static_file_server(static_dir: str):
            """创建静态文件服务器"""
            served_files = {}
            cache_headers = {}

            # 模拟文件系统
            mock_files = {
                'index.html': '<html><body>Hello World</body></html>',
                'style.css': 'body { font-family: Arial; }',
                'app.js': 'console.log("Hello from JS");',
                'logo.png': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01',  # PNG header
            }

            def serve_file(filepath: str, headers: Dict[str, str]) -> Dict[str, Any]:
                """服务静态文件"""
                if not filepath.startswith('/'):
                    filepath = '/' + filepath

                # 移除查询参数
                filepath = filepath.split('?')[0]

                # 安全检查：防止目录遍历
                if '..' in filepath or filepath.startswith('/../'):
                    return {
                        'status': 403,
                        'headers': {'Content-Type': 'text/plain'},
                        'body': 'Forbidden: Directory traversal attempt'
                    }

                # 获取文件名
                filename = filepath.lstrip('/')
                if filename in mock_files:
                    content = mock_files[filename]
                    content_type = get_content_type(filename)

                    # 检查缓存
                    etag = generate_etag(content)
                    if 'If-None-Match' in headers and headers['If-None-Match'] == etag:
                        return {
                            'status': 304,
                            'headers': {'ETag': etag},
                            'body': ''
                        }

                    # 记录服务统计
                    served_files[filename] = served_files.get(filename, 0) + 1

                    # 设置缓存头
                    cache_control = cache_headers.get(filename, 'no-cache')

                    return {
                        'status': 200,
                        'headers': {
                            'Content-Type': content_type,
                            'Content-Length': str(len(content) if isinstance(content, str) else len(content)),
                            'ETag': etag,
                            'Cache-Control': cache_control,
                            'Last-Modified': datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
                        },
                        'body': content
                    }
                else:
                    return {
                        'status': 404,
                        'headers': {'Content-Type': 'text/plain'},
                        'body': 'File not found'
                    }

            def get_content_type(filename: str) -> str:
                """获取文件内容类型"""
                ext = filename.split('.')[-1].lower()
                content_types = {
                    'html': 'text/html',
                    'css': 'text/css',
                    'js': 'application/javascript',
                    'json': 'application/json',
                    'png': 'image/png',
                    'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg',
                    'gi': 'image/gi',
                    'txt': 'text/plain',
                    'xml': 'application/xml'
                }
                return content_types.get(ext, 'application/octet-stream')

            def generate_etag(content) -> str:
                """生成ETag"""
                import hashlib
                if isinstance(content, str):
                    content = content.encode()
                return f'"{hashlib.md5(content).hexdigest()}"'

            def set_cache_policy(filename: str, policy: str):
                """设置缓存策略"""
                cache_headers[filename] = policy

            def get_serving_stats() -> Dict[str, Any]:
                """获取服务统计"""
                return {
                    'total_files_served': sum(served_files.values()),
                    'unique_files_served': len(served_files),
                    'most_served_file': max(served_files.items(), key=lambda x: x[1], default=(None, 0))[0],
                    'files_by_type': {}
                }

            def compress_response(content: str, accept_encoding: str) -> tuple:
                """压缩响应（简化实现）"""
                if 'gzip' in accept_encoding:
                    # 模拟gzip压缩
                    compressed = f"compressed_{content[:50]}"  # 简化
                    return compressed, 'gzip'
                return content, None

            return {
                'serve_file': serve_file,
                'set_cache_policy': set_cache_policy,
                'get_serving_stats': get_serving_stats,
                'compress_response': compress_response
            }

        # 创建静态文件服务器
        static_server = create_static_file_server('/var/www/static')

        # 设置缓存策略
        static_server['set_cache_policy']('style.css', 'max-age=3600')
        static_server['set_cache_policy']('app.js', 'max-age=86400')

        # 测试HTML文件服务
        response = static_server['serve_file']('/index.html', {})
        assert response['status'] == 200
        assert response['headers']['Content-Type'] == 'text/html'
        assert 'ETag' in response['headers']
        assert 'Cache-Control' in response['headers']
        assert '<html>' in response['body']

        # 测试CSS文件服务
        response = static_server['serve_file']('/style.css', {})
        assert response['status'] == 200
        assert response['headers']['Content-Type'] == 'text/css'
        assert response['headers']['Cache-Control'] == 'max-age=3600'

        # 测试JavaScript文件服务
        response = static_server['serve_file']('/app.js', {})
        assert response['status'] == 200
        assert response['headers']['Content-Type'] == 'application/javascript'
        assert response['headers']['Cache-Control'] == 'max-age=86400'

        # 测试缓存验证（304 Not Modified）
        etag = response['headers']['ETag']
        response_cached = static_server['serve_file']('/app.js', {'If-None-Match': etag})
        assert response_cached['status'] == 304

        # 测试不存在的文件
        response_404 = static_server['serve_file']('/nonexistent.txt', {})
        assert response_404['status'] == 404

        # 测试目录遍历防护
        response_forbidden = static_server['serve_file']('/../etc/passwd', {})
        assert response_forbidden['status'] == 403

        # 验证服务统计
        stats = static_server['get_serving_stats']()
        assert stats['total_files_served'] >= 3  # index.html, style.css, app.js
        assert stats['unique_files_served'] >= 3

    def test_middleware_integration(self):
        """测试中间件集成"""
        def create_middleware_stack():
            """创建中间件栈"""
            middlewares = []
            execution_log = []

            def add_middleware(name: str, middleware_func: Callable):
                """添加中间件"""
                middlewares.append({
                    'name': name,
                    'func': middleware_func,
                    'enabled': True
                })

            def execute_middleware_stack(request: Dict[str, Any]) -> Dict[str, Any]:
                """执行中间件栈"""
                execution_log.clear()

                def next_middleware(index: int, req: Dict[str, Any]) -> Dict[str, Any]:
                    if index >= len(middlewares):
                        # 最终处理器
                        execution_log.append('final_handler')
                        return {
                            'status': 200,
                            'body': f'Processed request for {req["path"]}',
                            'processed_by': execution_log.copy()
                        }

                    middleware = middlewares[index]
                    if not middleware['enabled']:
                        return next_middleware(index + 1, req)

                    execution_log.append(f"middleware_{middleware['name']}_start")

                    def next_func(modified_req=None):
                        execution_log.append(f"middleware_{middleware['name']}_end")
                        return next_middleware(index + 1, modified_req or req)

                    result = middleware['func'](req, next_func)
                    return result

                return execute_middleware_stack(0, request)

            def enable_middleware(name: str):
                """启用中间件"""
                for m in middlewares:
                    if m['name'] == name:
                        m['enabled'] = True

            def disable_middleware(name: str):
                """禁用中间件"""
                for m in middlewares:
                    if m['name'] == name:
                        m['enabled'] = False

            def get_execution_log() -> List[str]:
                """获取执行日志"""
                return execution_log.copy()

            return {
                'add_middleware': add_middleware,
                'execute_middleware_stack': execute_middleware_stack,
                'enable_middleware': enable_middleware,
                'disable_middleware': disable_middleware,
                'get_execution_log': get_execution_log
            }

        # 创建中间件栈
        middleware_stack = create_middleware_stack()

        # 添加认证中间件
        def auth_middleware(request, next_func):
            if 'authorization' not in request.get('headers', {}):
                return {'status': 401, 'error': 'Unauthorized'}
            request['user'] = 'authenticated_user'
            return next_func(request)

        # 添加日志中间件
        def logging_middleware(request, next_func):
            request['logged'] = True
            result = next_func(request)
            result['logged'] = True
            return result

        # 添加CORS中间件
        def cors_middleware(request, next_func):
            result = next_func(request)
            result['headers'] = result.get('headers', {})
            result['headers']['Access-Control-Allow-Origin'] = '*'
            return result

        # 添加限流中间件
        def rate_limit_middleware(request, next_func):
            client_ip = request.get('client_ip', 'unknown')
            # 简化的限流逻辑
            if client_ip == 'blocked_ip':
                return {'status': 429, 'error': 'Too Many Requests'}
            return next_func(request)

        middleware_stack['add_middleware']('auth', auth_middleware)
        middleware_stack['add_middleware']('logging', logging_middleware)
        middleware_stack['add_middleware']('cors', cors_middleware)
        middleware_stack['add_middleware']('rate_limit', rate_limit_middleware)

        # 测试完整中间件链
        request = {
            'method': 'GET',
            'path': '/api/data',
            'headers': {'Authorization': 'Bearer token123'},
            'client_ip': '192.168.1.100'
        }

        response = middleware_stack['execute_middleware_stack'](request)

        # 验证响应
        assert response['status'] == 200
        assert 'Processed request for /api/data' in response['body']
        assert response['logged'] == True
        assert 'Access-Control-Allow-Origin' in response['headers']
        assert 'processed_by' in response

        # 验证执行顺序
        execution_log = middleware_stack['get_execution_log']()
        expected_sequence = [
            'middleware_auth_start',
            'middleware_auth_end',
            'middleware_logging_start',
            'middleware_logging_end',
            'middleware_cors_start',
            'middleware_cors_end',
            'middleware_rate_limit_start',
            'middleware_rate_limit_end',
            'final_handler'
        ]
        assert execution_log == expected_sequence

        # 测试认证失败
        unauth_request = {
            'method': 'GET',
            'path': '/api/data',
            'headers': {},
            'client_ip': '192.168.1.100'
        }

        auth_response = middleware_stack['execute_middleware_stack'](unauth_request)
        assert auth_response['status'] == 401
        assert auth_response['error'] == 'Unauthorized'

        # 测试限流
        blocked_request = {
            'method': 'GET',
            'path': '/api/data',
            'headers': {'Authorization': 'Bearer token123'},
            'client_ip': 'blocked_ip'
        }

        rate_limit_response = middleware_stack['execute_middleware_stack'](blocked_request)
        assert rate_limit_response['status'] == 429

        # 测试禁用中间件
        middleware_stack['disable_middleware']('auth')

        # 现在没有认证要求的请求应该成功
        no_auth_request = {
            'method': 'GET',
            'path': '/api/data',
            'client_ip': '192.168.1.100'
        }

        no_auth_response = middleware_stack['execute_middleware_stack'](no_auth_request)
        assert no_auth_response['status'] == 200
        assert 'user' not in no_auth_request  # 认证中间件未执行

    def test_error_handling_and_logging(self):
        """测试错误处理和日志记录"""
        def create_error_handler():
            """创建错误处理器"""
            error_log = []
            error_stats = {
                'total_errors': 0,
                'errors_by_type': {},
                'errors_by_status': {},
                'recent_errors': []
            }

            def handle_error(error_type: str, error_details: Dict[str, Any],
                           request_context: Dict[str, Any]) -> Dict[str, Any]:
                """处理错误"""
                error_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'type': error_type,
                    'details': error_details,
                    'request': request_context,
                    'stack_trace': error_details.get('stack_trace', 'N/A')
                }

                error_log.append(error_entry)

                # 更新统计
                error_stats['total_errors'] += 1

                if error_type not in error_stats['errors_by_type']:
                    error_stats['errors_by_type'][error_type] = 0
                error_stats['errors_by_type'][error_type] += 1

                # 生成错误响应
                status_code = get_status_for_error(error_type)
                if status_code not in error_stats['errors_by_status']:
                    error_stats['errors_by_status'][status_code] = 0
                error_stats['errors_by_status'][status_code] += 1

                # 保留最近的错误
                error_stats['recent_errors'].append(error_entry)
                if len(error_stats['recent_errors']) > 10:
                    error_stats['recent_errors'].pop(0)

                # 记录错误日志
                log_error(error_entry)

                return {
                    'status': status_code,
                    'error': {
                        'type': error_type,
                        'message': error_details.get('message', 'An error occurred'),
                        'request_id': request_context.get('request_id', 'unknown'),
                        'timestamp': error_entry['timestamp']
                    },
                    'headers': {'Content-Type': 'application/json'}
                }

            def get_status_for_error(error_type: str) -> int:
                """根据错误类型获取HTTP状态码"""
                status_map = {
                    'validation_error': 400,
                    'authentication_error': 401,
                    'authorization_error': 403,
                    'not_found_error': 404,
                    'rate_limit_error': 429,
                    'server_error': 500,
                    'service_unavailable': 503,
                    'timeout_error': 504
                }
                return status_map.get(error_type, 500)

            def log_error(error_entry: Dict[str, Any]):
                """记录错误日志"""
                # 在实际实现中，这里会写入日志文件或发送到日志服务
                print(f"[ERROR] {error_entry['timestamp']} - {error_entry['type']}: {error_entry['details'].get('message', 'Unknown error')}")

            def get_error_stats() -> Dict[str, Any]:
                """获取错误统计"""
                return error_stats.copy()

            def clear_error_log():
                """清空错误日志"""
                error_log.clear()
                error_stats['total_errors'] = 0
                error_stats['errors_by_type'].clear()
                error_stats['errors_by_status'].clear()
                error_stats['recent_errors'].clear()

            return {
                'handle_error': handle_error,
                'get_error_stats': get_error_stats,
                'clear_error_log': clear_error_log,
                'get_error_log': lambda: error_log.copy()
            }

        # 创建错误处理器
        error_handler = create_error_handler()

        # 测试各种错误类型
        test_errors = [
            {
                'type': 'validation_error',
                'details': {'message': 'Invalid input data', 'field': 'email'},
                'request': {'method': 'POST', 'path': '/api/users', 'request_id': 'req_123'}
            },
            {
                'type': 'authentication_error',
                'details': {'message': 'Invalid token'},
                'request': {'method': 'GET', 'path': '/api/data', 'request_id': 'req_124'}
            },
            {
                'type': 'not_found_error',
                'details': {'message': 'Resource not found'},
                'request': {'method': 'GET', 'path': '/api/users/999', 'request_id': 'req_125'}
            },
            {
                'type': 'server_error',
                'details': {'message': 'Database connection failed', 'stack_trace': '...'},
                'request': {'method': 'GET', 'path': '/api/data', 'request_id': 'req_126'}
            }
        ]

        for error_info in test_errors:
            response = error_handler['handle_error'](
                error_info['type'],
                error_info['details'],
                error_info['request']
            )

            # 验证错误响应
            assert 'status' in response
            assert 'error' in response
            assert response['error']['type'] == error_info['type']
            assert response['error']['message'] == error_info['details']['message']

        # 验证错误统计
        stats = error_handler['get_error_stats']()
        assert stats['total_errors'] == 4
        assert stats['errors_by_type']['validation_error'] == 1
        assert stats['errors_by_type']['authentication_error'] == 1
        assert stats['errors_by_type']['not_found_error'] == 1
        assert stats['errors_by_type']['server_error'] == 1

        # 验证状态码统计
        assert stats['errors_by_status'][400] == 1  # validation_error
        assert stats['errors_by_status'][401] == 1  # authentication_error
        assert stats['errors_by_status'][404] == 1  # not_found_error
        assert stats['errors_by_status'][500] == 1  # server_error

        # 验证最近错误记录
        assert len(stats['recent_errors']) == 4

        # 验证错误日志
        error_log = error_handler['get_error_log']()
        assert len(error_log) == 4

        for i, log_entry in enumerate(error_log):
            assert log_entry['type'] == test_errors[i]['type']
            assert log_entry['details']['message'] == test_errors[i]['details']['message']

        # 测试清空日志
        error_handler['clear_error_log']()
        cleared_stats = error_handler['get_error_stats']()
        assert cleared_stats['total_errors'] == 0
        assert len(cleared_stats['recent_errors']) == 0
