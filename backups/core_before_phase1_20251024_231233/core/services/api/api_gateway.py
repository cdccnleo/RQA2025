import os
#!/usr/bin/env python3
"""
API网关

提供统一的微服务入口、路由管理、负载均衡、认证授权和流量控制
"""

import logging
import json
import hashlib
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import asyncio
import aiohttp
from aiohttp import web
import jwt
import redis
import uuid

logger = logging.getLogger(__name__)

# 尝试导入必要的库
try:
    import aiohttp_cors
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    logger.warning("aiohttp-cors不可用，CORS支持将被禁用")

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis不可用，分布式限流将被禁用")


class HttpMethod(Enum):

    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class ServiceStatus(Enum):

    """服务状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    DOWN = "down"


class RateLimitType(Enum):

    """限流类型"""
    IP = "ip"
    USER = "user"
    GLOBAL = "global"


API_KEY = os.getenv("API_KEY", "")


@dataclass
@dataclass
class ServiceEndpoint:

    """服务端点"""
    service_name: str
    path: str = ""
    method: HttpMethod = HttpMethod.GET
    upstream_url: str = ""
    timeout: int = 30
    retries: int = 3
    weight: int = 1
    health_check_url: Optional[str] = None
    status: ServiceStatus = ServiceStatus.HEALTHY
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0


@dataclass
class RateLimitRule:

    """限流规则"""
    limit_type: RateLimitType
    limit: int  # 请求数限制
    window: int  # 时间窗口(秒)
    key: str = ""  # 限流键


@dataclass
class RouteRule:

    """路由规则"""
    path: str
    method: HttpMethod
    service_name: str
    strip_prefix: bool = True
    rate_limits: List[RateLimitRule] = field(default_factory=list)
    auth_required: bool = True
    cors_enabled: bool = True
    cache_enabled: bool = False
    cache_ttl: int = 300


@dataclass
class ApiRequest:

    """API请求"""
    id: str
    method: HttpMethod
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    body: Optional[bytes] = None
    client_ip: str = ""
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ApiResponse:

    """API响应"""
    status_code: int
    headers: Dict[str, str]
    body: bytes
    processing_time: float
    upstream_url: str
    cached: bool = False


class CircuitBreaker:

    """熔断器"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,


                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
        self.lock = threading.Lock()

    def call(self, func: Callable) -> Any:
        """执行带熔断器的函数"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func()
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        time_diff = datetime.now() - self.last_failure_time
        return time_diff.total_seconds() >= self.recovery_timeout

    def _on_success(self):
        """成功回调"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")

    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def record_failure(self):
        """记录失败"""
        with self.lock:
            self._on_failure()

    def record_success(self):
        """记录成功"""
        with self.lock:
            self._on_success()

    def can_attempt(self) -> bool:
        """检查是否可以尝试请求"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker attempting recovery - state changed to HALF_OPEN")
                    return True
                return False
            return True

    def _invalidate_cache(self, key: str):
        """使缓存失效"""
        with self.cache_lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_expiration:
                del self.cache_expiration[key]


class LoadBalancer:

    """负载均衡器"""

    def __init__(self, algorithm: str = "round_robin"):

        self.algorithm = algorithm
        self.endpoints: List[ServiceEndpoint] = []
        self.current_index = 0
        self.lock = threading.Lock()

    def add_endpoint(self, endpoint: ServiceEndpoint):
        """添加服务端点"""
        self.endpoints.append(endpoint)

    def select_endpoint(self) -> Optional[ServiceEndpoint]:
        """选择服务端点（别名）"""
        return self.get_endpoint()

    def get_endpoint(self) -> Optional[ServiceEndpoint]:
        """获取服务端点"""
        if not self.endpoints:
            return None

        healthy_endpoints = [ep for ep in self.endpoints if ep.status == ServiceStatus.HEALTHY]
        if not healthy_endpoints:
            return None

        with self.lock:
            if self.algorithm == "round_robin":
                endpoint = healthy_endpoints[self.current_index % len(healthy_endpoints)]
                self.current_index += 1
                return endpoint
            elif self.algorithm == "weighted":
                return self._weighted_selection(healthy_endpoints)
            elif self.algorithm == "random":
                import secrets
                return secrets.choice(healthy_endpoints)
            else:
                return healthy_endpoints[0]

    def _weighted_selection(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """加权选择"""
        total_weight = sum(ep.weight for ep in endpoints)
        import secrets
        r = secrets.uniform(0, total_weight)
        current_weight = 0

        for endpoint in endpoints:
            current_weight += endpoint.weight
            if r <= current_weight:
                return endpoint

        return endpoints[0]


class RateLimiter:

    """限流器"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):

        self.redis_client = redis_client
        self.local_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def is_allowed(self, rule: RateLimitRule, key: str) -> bool:
        """检查是否允许请求"""
        current_time = int(time.time())

        if self.redis_client and REDIS_AVAILABLE:
            return self._check_redis_limit(rule, key, current_time)
        else:
            return self._check_local_limit(rule, key, current_time)

    def _check_redis_limit(self, rule: RateLimitRule, key: str, current_time: int) -> bool:
        """检查Redis限流"""
        try:
            redis_key = f"ratelimit:{rule.limit_type.value}:{key}:{current_time // rule.window}"

            # 使用Redis管道执行原子操作
            with self.redis_client.pipeline() as pipe:
                pipe.incr(redis_key)
                pipe.expire(redis_key, rule.window)
                count = pipe.execute()[0]

            return count <= rule.limit

        except Exception as e:
            logger.error(f"Redis限流检查失败: {e}")
            return True  # 出错时允许请求

    def _check_local_limit(self, rule: RateLimitRule, key: str, current_time: int) -> bool:
        """检查本地限流"""
        request_times = self.local_limits[key]

        # 清理过期请求
        while request_times and request_times[0] < current_time - rule.window:
            request_times.popleft()

        # 检查是否超过限制
        if len(request_times) >= rule.limit:
            return False

        # 添加当前请求
        request_times.append(current_time)
        return True


class AuthenticationManager:

    """认证管理器"""

    def __init__(self, jwt_secret: str, jwt_algorithm: str = "HS256"):

        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm

    def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # 检查令牌是否过期
            if 'exp' in payload:
                if datetime.fromtimestamp(payload['exp']) < datetime.now():
                    return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("JWT令牌已过期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("无效的JWT令牌")
            return None
        except Exception as e:
            logger.error(f"JWT验证失败: {e}")
            return None

    def authorize(self, user_info: Dict[str, Any], required_permissions: List[str]) -> bool:
        """授权检查"""
        user_permissions = user_info.get('permissions', [])
        return any(perm in user_permissions for perm in required_permissions)

    def generate_token(self, user_info: Dict[str, Any], expires_in: int = 3600) -> str:
        """生成JWT令牌"""
        payload = {
            **user_info,
            'iat': datetime.now().timestamp(),
            'exp': (datetime.now() + timedelta(seconds=expires_in)).timestamp()
        }

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)


class ApiGateway:

    """API网关"""

    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self.routes: Dict[str, RouteRule] = {}
        self.services: Dict[str, LoadBalancer] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiter = RateLimiter()

        # 配置
        self.port = config.get('port', 8080)
        self.host = config.get('host', '0.0.0.0')

        # 认证
        self.auth_manager = AuthenticationManager(
            jwt_secret=config.get('jwt_secret', 'default_secret'),
            jwt_algorithm=config.get('jwt_algorithm', 'HS256')
        )

        # Redis客户端
        if REDIS_AVAILABLE and config.get('redis_enabled', False):
            self.redis_client = redis.Redis(
                host=config.get('redis_host', 'localhost'),
                port=config.get('redis_port', 6379),
                db=config.get('redis_db', 0),
                password=config.get('redis_password')
            )
            self.rate_limiter.redis_client = self.redis_client
        else:
            self.redis_client = None

        # 缓存
        self.response_cache: Dict[str, Tuple[ApiResponse, float]] = {}
        self.cache_lock = threading.Lock()

        # 速率限制存储
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.rate_limit_lock = threading.Lock()

        # API密钥存储
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.api_key_lock = threading.Lock()

        # 缓存对象
        self.cache: Dict[str, Any] = {}
        self.cache_expiration: Dict[str, float] = {}

        # 请求去重
        self.request_cache: Dict[str, float] = {}
        self.request_cache_lock = threading.Lock()

        # 最大并发请求数
        self.max_concurrent_requests = config.get('max_concurrent_requests', 100)
        self.current_requests = 0
        self.requests_lock = threading.Lock()

        # 请求超时设置
        self.request_timeout = config.get('request_timeout', 30)

        # 功能开关
        self.rate_limit_enabled = config.get('rate_limit_enabled', True)
        self.cache_enabled = config.get('cache_enabled', True)
        self.cors_enabled = config.get('cors_enabled', False)

        # 指标收集
        self.metrics = {
            'total_requests': 0,
            'total_responses': 0,
            'avg_response_time': 0.0,
            'requests_by_method': {},
            'requests_by_endpoint': {},
            'error_rate': 0.0
        }

        # 统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'rate_limited_requests': 0,
            'auth_failed_requests': 0
        }

        # aiohttp应用
        self.app = web.Application()
        self.setup_routes()

        # CORS
        if CORS_AVAILABLE:
            aiohttp_cors.setup(self.app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
                )
            })

        logger.info("API网关初始化完成")

    def setup_routes(self):
        """设置路由"""
        self.app.router.add_route('*', '/{path:.*}', self.handle_request)

        # 健康检查端点
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.get_metrics)

    def add_route(self, rule: RouteRule):
        """添加路由规则"""
        key = f"{rule.method.value}:{rule.path}"
        self.routes[key] = rule

        # 创建服务负载均衡器
        if rule.service_name not in self.services:
            self.services[rule.service_name] = LoadBalancer()

        # 创建熔断器
        if rule.service_name not in self.circuit_breakers:
            self.circuit_breakers[rule.service_name] = CircuitBreaker()

        logger.info(f"路由已添加: {key} -> {rule.service_name}")

    def register_service(self, service_name: str, endpoints: List[str], health_check_url: Optional[str] = None):
        """注册服务"""
        if service_name not in self.services:
            self.services[service_name] = LoadBalancer()

        load_balancer = self.services[service_name]
        for endpoint in endpoints:
            service_endpoint = ServiceEndpoint(
                service_name=service_name,
                upstream_url=endpoint,
                health_check_url=health_check_url or f"{endpoint}/health",
                weight=1
            )
            load_balancer.add_endpoint(service_endpoint)

        # 创建熔断器
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()

        logger.info(f"服务已注册: {service_name}, 端点数量: {len(endpoints)}")

    def _match_route(self, method: str, path: str) -> Optional[Tuple[RouteRule, Dict[str, str]]]:
        """匹配路由规则"""
        # 如果method是HttpMethod枚举，转换为字符串
        if hasattr(method, 'value'):
            method_str = method.value
        else:
            method_str = str(method)

        key = f"{method_str}:{path}"
        if key in self.routes:
            return self.routes[key], {}

        # 支持通配符匹配和路径参数提取
        for route_key, rule in self.routes.items():
            route_method, route_path = route_key.split(':', 1)
            if route_method != method_str:
                continue

            # 精确匹配
            if route_path == path:
                return rule, {}

            # 支持路径参数匹配（如 /users/{id}）
            if '{' in route_path and '}' in route_path:
                import re
                # 将路由路径转换为正则表达式
                pattern = re.sub(r'\{([^}]+)\}', r'(?P<\1>[^/]+)', route_path)
                pattern = f'^{pattern}$'
                match = re.match(pattern, path)
                if match:
                    path_params = match.groupdict()
                    return rule, path_params

            # 简单通配符匹配
            if route_path.endswith('/*'):
                prefix = route_path[:-2]
                if path.startswith(prefix):
                    return rule, {}

        return None, {}

    def _generate_jwt_token(self, payload: Dict[str, Any], expiration_hours: int = 24) -> str:
        """生成JWT令牌"""
        import jwt
        import datetime

        payload_copy = payload.copy()
        # 只有在payload中没有设置exp时才设置默认过期时间
        if 'exp' not in payload_copy:
            payload_copy['exp'] = datetime.datetime.utcnow(
            ) + datetime.timedelta(hours=expiration_hours)
        payload_copy['iat'] = datetime.datetime.utcnow()

        return jwt.encode(payload_copy, self.auth_manager.jwt_secret, algorithm=self.auth_manager.jwt_algorithm)

    def _validate_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """验证JWT令牌"""
        try:
            import jwt
            payload = jwt.decode(token, self.auth_manager.jwt_secret,
                                 algorithms=[self.auth_manager.jwt_algorithm])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, None
        except jwt.InvalidTokenError:
            return False, None

    def _validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """验证API密钥"""
        with self.api_key_lock:
            if api_key in self.api_keys and self.api_keys[api_key].get('active', True):
                return True, self.api_keys[api_key]
            return False, None

    def _check_permissions(self, user_id: str, resource: str, action: str) -> bool:
        """检查用户权限"""
        # 简化权限检查逻辑
        with self.api_key_lock:
            for api_key_info in self.api_keys.values():
                if api_key_info.get('user_id') == user_id:
                    permissions = api_key_info.get('permissions', [])
                    # 检查完整权限格式 (resource:action)
                    if f"{resource}:{action}" in permissions:
                        return True
                    # 检查简单权限格式 (action only)
                    if action in permissions:
                        return True
        return False

    def add_service_endpoint(self, endpoint: ServiceEndpoint):
        """添加服务端点"""
        if endpoint.service_name not in self.services:
            self.services[endpoint.service_name] = LoadBalancer()

        self.services[endpoint.service_name].add_endpoint(endpoint)
        logger.info(f"服务端点已添加: {endpoint.service_name} -> {endpoint.upstream_url}")

    async def handle_request(self, request: web.Request) -> web.Response:
        """处理请求"""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # 创建API请求对象
            api_request = ApiRequest(
                id=request_id,
                method=HttpMethod(request.method),
                path=request.path,
                headers=dict(request.headers),
                query_params=dict(request.query),
                client_ip=self._get_client_ip(request)
            )

            # 读取请求体
            if request.method in ['POST', 'PUT', 'PATCH']:
                api_request.body = await request.read()

            logger.info(f"处理请求: {request_id} {request.method} {request.path}")

            # 查找路由规则
            route_rule = self._find_route_rule(api_request)
            if not route_rule:
                return web.json_response(
                    {'error': 'Route not found', 'request_id': request_id},
                    status=404
                )

            # 认证和授权
            if route_rule.auth_required:
                auth_result = await self._authenticate_request(api_request)
                if not auth_result['success']:
                    self.stats['auth_failed_requests'] += 1
                    return web.json_response(
                        {'error': auth_result['error'], 'request_id': request_id},
                        status=401
                    )
                api_request.user_id = auth_result['user_id']

            # 限流检查
            for rate_limit in route_rule.rate_limits:
                key = self._get_rate_limit_key(rate_limit, api_request)
                if not self.rate_limiter.is_allowed(rate_limit, key):
                    self.stats['rate_limited_requests'] += 1
                    return web.json_response(
                        {'error': 'Rate limit exceeded', 'request_id': request_id},
                        status=429
                    )

            # 缓存检查
            if route_rule.cache_enabled:
                cache_key = self._generate_cache_key(api_request)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    cached_response.cached = True
                    return self._create_web_response(cached_response)

            # 路由到上游服务
            response = await self._route_to_service(api_request, route_rule)

            # 缓存响应
            if route_rule.cache_enabled and response.status_code == 200:
                self._cache_response(self._generate_cache_key(
                    api_request), response, route_rule.cache_ttl)

            # 更新统计
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            processing_time = time.time() - start_time
            self._update_avg_response_time(processing_time)

            return self._create_web_response(response)

        except Exception as e:
            logger.error(f"请求处理失败: {e}")
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1

            return web.json_response(
                {'error': 'Internal server error', 'request_id': request_id},
                status=500
            )

    def _find_route_rule(self, api_request: ApiRequest) -> Optional[RouteRule]:
        """查找路由规则"""
        # 精确匹配
        key = f"{api_request.method.value}:{api_request.path}"
        if key in self.routes:
            return self.routes[key]

        # 模式匹配（TODO: 支持路径参数）
        for route_key, rule in self.routes.items():
            if route_key.split(':')[0] == api_request.method.value:
                # 简单的路径匹配逻辑
                route_path = route_key.split(':', 1)[1]
                if self._path_matches(route_path, api_request.path):
                    return rule

        return None

    def _path_matches(self, route_path: str, request_path: str) -> bool:
        """路径匹配（简化版本）"""
        # TODO: 实现更复杂的路径匹配逻辑
        return route_path == request_path

    async def _authenticate_request(self, api_request: ApiRequest) -> Dict[str, Any]:
        """认证请求"""
        # 检查Authorization头
        auth_header = api_request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return {'success': False, 'error': 'Missing or invalid authorization header'}

        token = auth_header[7:]  # Remove 'Bearer ' prefix

        user_info = self.auth_manager.authenticate(token)
        if not user_info:
            return {'success': False, 'error': 'Invalid or expired token'}

        return {
            'success': True,
            'user_id': user_info.get('user_id'),
            'user_info': user_info
        }

    def _get_rate_limit_key(self, rule: RateLimitRule, api_request: ApiRequest) -> str:
        """获取限流键"""
        if rule.limit_type == RateLimitType.IP:
            return api_request.client_ip
        elif rule.limit_type == RateLimitType.USER:
            return api_request.user_id or 'anonymous'
        elif rule.limit_type == RateLimitType.API_KEY:
            return api_request.api_key or 'no_key'
        elif rule.limit_type == RateLimitType.GLOBAL:
            return 'global'
        else:
            return 'default'

    async def _route_to_service(self, api_request: ApiRequest, route_rule: RouteRule) -> ApiResponse:
        """路由到上游服务"""
        service_name = route_rule.service_name

        # 获取服务端点
        load_balancer = self.services.get(service_name)
        if not load_balancer:
            raise Exception(f"Service not found: {service_name}")

        circuit_breaker = self.circuit_breakers.get(service_name)

        # 使用熔断器调用服务

        def call_service():

            return asyncio.run(self._call_upstream_service(api_request, route_rule, load_balancer))

        if circuit_breaker:
            return circuit_breaker.call(call_service)
        else:
            return await self._call_upstream_service(api_request, route_rule, load_balancer)

    async def _call_upstream_service(self, api_request: ApiRequest, route_rule: RouteRule,
                                     load_balancer: LoadBalancer) -> ApiResponse:
        """调用上游服务"""
        endpoint = load_balancer.get_endpoint()
        if not endpoint:
            raise Exception(f"No healthy endpoint available for service: {route_rule.service_name}")

        # 构建上游URL
        upstream_path = api_request.path
        if route_rule.strip_prefix:
            # 移除服务名前缀
            prefix = f"/{route_rule.service_name}"
            if upstream_path.startswith(prefix):
                upstream_path = upstream_path[len(prefix):]

        upstream_url = f"{endpoint.upstream_url}{upstream_path}"

        # 添加查询参数
        if api_request.query_params:
            query_string = '&'.join([f"{k}={v}" for k, v in api_request.query_params.items()])
            upstream_url += f"?{query_string}"

        # 准备请求头
        headers = dict(api_request.headers)
        headers.pop('Host', None)  # 移除Host头

        # 发送请求
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    api_request.method.value,
                    upstream_url,
                    headers=headers,
                    data=api_request.body,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as resp:
                    response_body = await resp.read()
                    response_headers = dict(resp.headers)

                    return ApiResponse(
                        status_code=resp.status,
                        headers=response_headers,
                        body=response_body,
                        processing_time=0.0,  # 将在更高层级计算
                        upstream_url=upstream_url
                    )

            except asyncio.TimeoutError:
                raise Exception(f"Upstream service timeout: {upstream_url}")
            except Exception as e:
                logger.error(f"Upstream service call failed: {e}")
                raise

    def _generate_cache_key(self, api_request: ApiRequest) -> str:
        """生成缓存键"""
        key_data = {
            'method': api_request.method.value,
            'path': api_request.path,
            'query': sorted(api_request.query_params.items()),
            'user_id': api_request.user_id
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[ApiResponse]:
        """获取缓存的响应"""
        with self.cache_lock:
            if cache_key in self.response_cache:
                response, expiry_time = self.response_cache[cache_key]
                if time.time() < expiry_time:
                    # 创建新的响应对象并标记为来自缓存
                    return ApiResponse(
                        status_code=response.status_code,
                        headers=response.headers,
                        body=response.body,
                        processing_time=response.processing_time,
                        upstream_url=response.upstream_url,
                        cached=True
                    )
                else:
                    del self.response_cache[cache_key]

        return None

    def _cache_response(self, cache_key: str, response: ApiResponse, ttl: int):
        """缓存响应"""
        with self.cache_lock:
            expiry_time = time.time() + ttl
            self.response_cache[cache_key] = (response, expiry_time)

            # 清理过期缓存
            current_time = time.time()
            expired_keys = [k for k, (_, exp) in self.response_cache.items() if current_time >= exp]
            for key in expired_keys:
                del self.response_cache[key]

    def _create_web_response(self, api_response: ApiResponse) -> web.Response:
        """创建web响应"""
        return web.Response(
            status=api_response.status_code,
            headers=api_response.headers,
            body=api_response.body
        )

    def _get_client_ip(self, request: web.Request) -> str:
        """获取客户端IP"""
        # 检查代理头
        forwarded_for = request.headers.get('X - Forwarded - For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()

        real_ip = request.headers.get('X - Real - IP')
        if real_ip:
            return real_ip

        # 默认使用远程地址
        return request.remote or 'unknown'

    def _update_avg_response_time(self, processing_time: float):
        """更新平均响应时间"""
        total_requests = self.stats['successful_requests'] + self.stats['failed_requests']
        if total_requests > 0:
            current_avg = self.stats['avg_response_time']
            self.stats['avg_response_time'] = (
                (current_avg * (total_requests - 1)) + processing_time
            ) / total_requests

    async def health_check(self, request: web.Request) -> web.Response:
        """健康检查"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })

    async def get_metrics(self, request: web.Request) -> web.Response:
        """获取指标"""
        return web.json_response({
            'stats': self.stats,
            'services': {
                name: {
                    'endpoints_count': len(lb.endpoints) if hasattr(lb, 'endpoints') else 0,
                    'healthy_endpoints': len([ep for ep in lb.endpoints if ep.status == ServiceStatus.HEALTHY]) if hasattr(lb, 'endpoints') else 0
                } for name, lb in self.services.items()
            },
            'routes_count': len(self.routes),
            'cache_entries': len(self.response_cache),
            'timestamp': datetime.now().isoformat()
        })

    def start(self):
        """启动API网关"""
        logger.info(f"启动API网关: {self.host}:{self.port}")
        web.run_app(self.app, host=self.host, port=self.port)

    def run_in_background(self):
        """在后台运行API网关"""
        import threading
        server_thread = threading.Thread(target=self.start, daemon=True)
        server_thread.start()
        logger.info("API网关已在后台启动")

    def _select_endpoint(self, service_name: str) -> Optional[ServiceEndpoint]:
        """选择服务端点（负载均衡）"""
        if service_name not in self.services:
            return None

        load_balancer = self.services[service_name]
        return load_balancer.select_endpoint()

    def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> bool:
        """检查端点健康状态"""
        # 简化健康检查逻辑，基于endpoint.status
        try:
            return endpoint.status == ServiceStatus.HEALTHY
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def _is_duplicate_request(self, request_id: str, ttl: int = 60) -> bool:
        """检查请求是否重复"""
        import time
        current_time = time.time()

        with self.request_cache_lock:
            if request_id in self.request_cache:
                last_request_time = self.request_cache[request_id]
                if current_time - last_request_time < ttl:
                    return True

            self.request_cache[request_id] = current_time
            return False

    def _collect_metrics(self, request: Optional['ApiRequest'] = None, response: Optional['ApiResponse'] = None) -> Dict[str, Any]:
        """收集性能指标"""
        # 如果提供了请求和响应，更新统计
        if request and response:
            # 更新stats
            self.stats['total_requests'] += 1
            if response.status_code >= 200 and response.status_code < 300:
                self.stats['successful_requests'] += 1
            elif response.status_code >= 400:
                self.stats['failed_requests'] += 1

            # 更新平均响应时间
            total_time = self.stats['avg_response_time'] * \
                (self.stats['total_requests'] - 1) + response.processing_time
            self.stats['avg_response_time'] = total_time / self.stats['total_requests']

            # 同时更新metrics以保持一致性
            self.metrics['total_requests'] = self.stats['total_requests']
            self.metrics['total_responses'] = self.stats['total_requests']  # 每个请求都有一个响应
            self.metrics['avg_response_time'] = self.stats['avg_response_time']

        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'avg_response_time': self.stats['avg_response_time'],
            'rate_limited_requests': self.stats['rate_limited_requests'],
            'auth_failed_requests': self.stats['auth_failed_requests'],
            'active_routes': len(self.routes),
            'active_services': len(self.services),
            'cache_size': len(self.cache),
            'current_requests': self.current_requests
        }

    def _create_error_response(self, error_code: int, message: str) -> ApiResponse:
        """创建错误响应"""
        import json
        error_body = json.dumps({
            'error': {
                'code': error_code,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        }).encode('utf-8')

        return ApiResponse(
            status_code=error_code,
            headers={'Content-Type': 'application/json'},
            body=error_body,
            processing_time=0.0,
            upstream_url="internal"  # 错误响应没有上游服务
        )

    async def process_request(self, request: ApiRequest) -> ApiResponse:
        """处理API请求"""
        try:
            # 验证请求
            if hasattr(request, 'headers'):
                auth_header = request.headers.get('Authorization', '')
                api_key = request.headers.get('X-API-Key', '')

                # JWT认证
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]
                    is_valid, user_info = self._validate_jwt_token(token)
                    if not is_valid:
                        return self._create_error_response(401, "Invalid token")

                # API密钥认证
                elif api_key:
                    is_valid, user_info = self._validate_api_key(api_key)
                    if not is_valid:
                        return self._create_error_response(401, "Invalid API key")

            # 匹配路由
            matched_route, path_params = self._match_route(request.method, request.path)
            if not matched_route:
                return self._create_error_response(404, "Route not found")

            # 选择端点
            endpoint = self._select_endpoint(matched_route.service_name)
            if not endpoint:
                return self._create_error_response(503, "Service unavailable")

            # 转发请求
            upstream_response = await self._forward_request(request, endpoint)

            return upstream_response

        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return self._create_error_response(500, "Internal server error")

    async def _forward_request(self, request: ApiRequest, endpoint: ServiceEndpoint) -> ApiResponse:
        """转发请求到上游服务"""
        try:
            # 这里简化实现，实际应该使用aiohttp转发请求
            import asyncio
            await asyncio.sleep(0.01)  # 模拟网络延迟

            return ApiResponse(
                status_code=200,
                headers={'Content-Type': 'application/json'},
                body=b'{"status": "success"}',
                processing_time=0.01
            )
        except Exception as e:
            logger.error(f"Forward request error: {e}")
            return self._create_error_response(500, "Forward error")

    def _check_rate_limit(self, client_ip: str, rule: Optional['RateLimitRule'] = None) -> Tuple[bool, Optional[int]]:
        """检查速率限制"""
        if rule is None:
            return True, None

        # 简化实现：使用内存中的计数器
        key = f"{rule.limit_type.value}:{client_ip}"
        current_time = time.time()

        if key not in self.rate_limits:
            self.rate_limits[key] = {'count': 0, 'window_start': current_time}
            record = self.rate_limits[key]
        else:
            record = self.rate_limits[key]

        # 检查是否需要重置窗口
        if current_time - record['window_start'] >= rule.window:
            record['count'] = 0
            record['window_start'] = current_time

        # 检查是否超过限制
        if record['count'] >= rule.limit:
            return False, int(rule.window - (current_time - record['window_start']))

        # 允许请求，增加计数
        record['count'] += 1
        return True, None

    def _invalidate_cache(self, key: str):
        """使缓存失效"""
        with self.cache_lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_expiration:
                del self.cache_expiration[key]
