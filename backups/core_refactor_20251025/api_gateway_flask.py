"""
RQA2025 API网关模块

提供统一的API入口、路由管理、负载均衡、认证授权等功能
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import jwt
from functools import wraps
import re

from flask import Flask, request, jsonify, g
import requests
from concurrent.futures import ThreadPoolExecutor

from .service_communicator import get_service_communicator
from .service_discovery import get_discovery_client

logger = logging.getLogger(__name__)


@dataclass
class RouteConfig:

    """路由配置"""
    path: str
    method: str
    service: str
    endpoint: str
    auth_required: bool = True
    rate_limit: Optional[int] = None  # 请求 / 分钟限制
    timeout: int = 30
    retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def route_key(self) -> str:
        """路由键"""
        return f"{self.method}:{self.path}"


@dataclass
class GatewayMetrics:

    """网关指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    active_connections: int = 0
    rate_limited_requests: int = 0
    auth_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:

        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(1, self.total_requests),
            'avg_response_time': self.avg_response_time,
            'active_connections': self.active_connections,
            'rate_limited_requests': self.rate_limited_requests,
            'auth_failures': self.auth_failures
        }


class RateLimiter:

    """速率限制器"""

    def __init__(self):

        self.requests: Dict[str, List[float]] = {}
        self.lock = threading.RLock()

    def is_allowed(self, client_id: str, limit: int, window: int = 60) -> bool:
        """检查是否允许请求"""
        with self.lock:
            now = time.time()
            if client_id not in self.requests:
                self.requests[client_id] = []

            # 清理过期请求
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < window
            ]

            # 检查是否超过限制
            if len(self.requests[client_id]) >= limit:
                return False

            # 添加新请求
            self.requests[client_id].append(now)
            return True

    def get_remaining(self, client_id: str, limit: int, window: int = 60) -> int:
        """获取剩余请求次数"""
        with self.lock:
            if client_id not in self.requests:
                return limit

            now = time.time()
            valid_requests = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < window
            ]
            self.requests[client_id] = valid_requests

            return max(0, limit - len(valid_requests))


class JWTAuthenticator:

    """JWT认证器"""

    def __init__(self, secret_key: str):

        self.secret_key = secret_key
        self.algorithm = "HS256"

    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """生成JWT token"""
        payload_copy = payload.copy()
        payload_copy['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
        payload_copy['iat'] = datetime.utcnow()

        return jwt.encode(payload_copy, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token已过期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("无效的JWT token")
            return None

    def get_token_from_request(self, request) -> Optional[str]:
        """从请求中获取token"""
        # 从Authorization header获取
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            return auth_header[7:]

        # 从cookie获取
        return request.cookies.get('access_token')


class IntegrationProxy:

    """API网关"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):

        self.host = host
        self.port = port
        self.app = Flask(__name__)

        # 组件初始化
        self.communicator = get_service_communicator()
        self.discovery_client = get_discovery_client()

        # 配置
        self.routes: Dict[str, RouteConfig] = {}
        self.middlewares: List[Callable] = []
        self.rate_limiter = RateLimiter()
        self.authenticator = JWTAuthenticator("your - secret - key - change - in - production")

        # 指标
        self.metrics = GatewayMetrics()
        self.metrics_lock = threading.RLock()

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=20)

        # 设置路由
        self._setup_routes()

        logger.info(f"API网关初始化完成: http://{host}:{port}")

    def _setup_routes(self):
        """设置路由"""
        @self.app.before_request
        def before_request():
            """请求前处理"""
            g.start_time = time.time()
            with self.metrics_lock:
                self.metrics.total_requests += 1
                self.metrics.active_connections += 1

        @self.app.after_request
        def after_request(response):
            """请求后处理"""
            if hasattr(g, 'start_time'):
                response_time = time.time() - g.start_time
                with self.metrics_lock:
                    self.metrics.avg_response_time = (
                        (self.metrics.avg_response_time * (self.metrics.total_requests - 1))
                        + response_time
                    ) / self.metrics.total_requests
                    self.metrics.active_connections -= 1

            return response

        @self.app.route('/')
        def index():
            """网关首页"""
            return jsonify({
                'service': 'RQA2025 API Gateway',
                'version': '1.0.0',
                'status': 'running',
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/health')
        def health():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': len(self.communicator.list_services())
            })

        @self.app.route('/metrics')
        def gateway_metrics():
            """网关指标"""
            with self.metrics_lock:
                return jsonify(self.metrics.to_dict())

        @self.app.route('/services')
        def list_services():
            """服务列表"""
            services = {}
            for service_name in self.communicator.list_services():
                instance = self.discovery_client.discover_service(service_name)
                if instance:
                    services[service_name] = {
                        'endpoint': f"{instance.host}:{instance.port}",
                        'healthy': self.communicator.is_service_healthy(service_name)
                    }

            return jsonify({
                'services': services,
                'total': len(services)
            })

        # 动态路由处理
        @self.app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        def api_proxy(path):
            """API代理"""
            return self._handle_api_request(path)

    def add_route(self, config: RouteConfig):
        """添加路由"""
        self.routes[config.route_key] = config
        logger.info(f"路由已添加: {config.route_key} -> {config.service}")

    def add_middleware(self, middleware: Callable):
        """添加中间件"""
        self.middlewares.append(middleware)

    def _handle_api_request(self, path: str):
        """处理API请求"""
        try:
            # 查找匹配的路由
            route_config = self._find_matching_route(request.method, f"/api/{path}")
            if not route_config:
                return jsonify({'error': 'Route not found'}), 404

            # 速率限制检查
            if route_config.rate_limit:
                client_id = self._get_client_id(request)
                if not self.rate_limiter.is_allowed(client_id, route_config.rate_limit):
                    with self.metrics_lock:
                        self.metrics.rate_limited_requests += 1
                    return jsonify({'error': 'Rate limit exceeded'}), 429

            # 认证检查
            if route_config.auth_required:
                user = self._authenticate_request(request)
                if not user:
                    with self.metrics_lock:
                        self.metrics.auth_failures += 1
                    return jsonify({'error': 'Authentication required'}), 401

            # 执行中间件
            for middleware in self.middlewares:
                result = middleware(request)
                if result:
                    return result

            # 转发请求
            response = self._forward_request(route_config, request)

            # 更新成功指标
            with self.metrics_lock:
                self.metrics.successful_requests += 1

            return response

        except Exception as e:
            logger.error(f"API请求处理出错: {e}")
            with self.metrics_lock:
                self.metrics.failed_requests += 1
            return jsonify({'error': 'Internal server error'}), 500

    def _find_matching_route(self, method: str, path: str) -> Optional[RouteConfig]:
        """查找匹配的路由"""
        route_key = f"{method}:{path}"

        # 精确匹配
        if route_key in self.routes:
            return self.routes[route_key]

        # 模式匹配
        for config in self.routes.values():
            if config.method == method:
                # 将路由路径转换为正则表达式
                pattern = config.path.replace('{', '(?P<').replace('}', '>[^/]+)')
                if re.match(f"^{pattern}$", path):
                    return config

        return None

    def _authenticate_request(self, request) -> Optional[Dict[str, Any]]:
        """认证请求"""
        token = self.authenticator.get_token_from_request(request)
        if not token:
            return None

        return self.authenticator.verify_token(token)

    def _get_client_id(self, request) -> str:
        """获取客户端ID"""
        # 使用IP地址作为客户端ID
        return request.remote_addr or "unknown"

    def _forward_request(self, route_config: RouteConfig, request):
        """转发请求"""
        try:
            # 获取服务实例
            service_instance = self.discovery_client.discover_service(route_config.service)
            if not service_instance:
                return jsonify({'error': 'Service unavailable'}), 503

            # 构建目标URL
            target_url = f"{service_instance.endpoint.base_url}{route_config.endpoint}"

            # 替换路径参数
            if '{' in route_config.endpoint and '}' in route_config.endpoint:
                path_params = self._extract_path_params(route_config.path, request.path)
                for param, value in path_params.items():
                    target_url = target_url.replace(f"{{{param}}}", value)

            # 准备请求头
            headers = dict(request.headers)
            headers.update(route_config.headers)
            headers.pop('Host', None)  # 移除Host头

            # 转发请求
            response = requests.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=request.get_data(),
                params=request.args,
                timeout=route_config.timeout,
                allow_redirects=False
            )

            # 返回响应
            return response.content, response.status_code, dict(response.headers)

        except requests.Timeout:
            return jsonify({'error': 'Request timeout'}), 504
        except Exception as e:
            logger.error(f"请求转发失败: {e}")
            return jsonify({'error': 'Service error'}), 502

    def _extract_path_params(self, route_pattern: str, actual_path: str) -> Dict[str, str]:
        """提取路径参数"""
        params = {}

        # 移除 / api前缀进行匹配
        if actual_path.startswith('/api/'):
            actual_path = actual_path[4:]

        route_parts = route_pattern.strip('/').split('/')
        path_parts = actual_path.strip('/').split('/')

        for route_part, path_part in zip(route_parts, path_parts):
            if route_part.startswith('{') and route_part.endswith('}'):
                param_name = route_part[1:-1]
                params[param_name] = path_part

        return params

    def register_service_routes(self, service_name: str, routes: List[Dict[str, Any]]):
        """注册服务路由"""
        for route_data in routes:
            config = RouteConfig(
                path=route_data['path'],
                method=route_data.get('method', 'GET'),
                service=service_name,
                endpoint=route_data['endpoint'],
                auth_required=route_data.get('auth_required', True),
                rate_limit=route_data.get('rate_limit'),
                timeout=route_data.get('timeout', 30),
                retries=route_data.get('retries', 3),
                headers=route_data.get('headers', {})
            )
            self.add_route(config)

    def start(self):
        """启动网关"""
        logger.info(f"API网关启动: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)

    def get_status(self) -> Dict[str, Any]:
        """获取网关状态"""
        return {
            'host': self.host,
            'port': self.port,
            'routes': len(self.routes),
            'services': len(self.communicator.list_services()),
            'metrics': self.metrics.to_dict()
        }


# 装饰器

def require_auth(f):
    """需要认证的装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):

        # 这里可以添加认证逻辑
        return f(*args, **kwargs)
    return decorated_function


def rate_limit(limit: int, window: int = 60):
    """速率限制装饰器"""

    def decorator(f):

        @wraps(f)
        def decorated_function(*args, **kwargs):

            # 这里可以添加速率限制逻辑
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# 全局网关实例
_gateway_instance: Optional[IntegrationProxy] = None
_gateway_lock = threading.Lock()


def get_api_gateway(host: str = "0.0.0.0", port: int = 8080) -> IntegrationProxy:
    """获取API网关实例（单例模式）"""
    global _gateway_instance

    if _gateway_instance is None:
        with _gateway_lock:
            if _gateway_instance is None:
                _gateway_instance = IntegrationProxy(host, port)

    return _gateway_instance


# 便捷函数

def register_gateway_routes(service_name: str, routes: List[Dict[str, Any]]):
    """注册网关路由"""
    gateway = get_api_gateway()
    gateway.register_service_routes(service_name, routes)


def start_gateway():
    """启动网关"""
    gateway = get_api_gateway()
    gateway.start()


if __name__ == "__main__":
    # 测试代码
    print("API网关模块测试")

    # 创建网关
    gateway = get_api_gateway()

    # 注册测试路由
    test_routes = [
        {
            'path': '/alerts',
            'method': 'GET',
            'endpoint': '/api / dashboard / overview',
            'auth_required': False
        },
        {
            'path': '/trading / status',
            'method': 'GET',
            'endpoint': '/api / trading / status',
            'auth_required': False
        }
    ]

    gateway.register_service_routes('alert - intelligence', test_routes)
    gateway.register_service_routes('trading - monitor', test_routes)

    print("网关状态:", gateway.get_status())
    print("已注册路由:", list(gateway.routes.keys()))

    print("API网关模块测试完成")
