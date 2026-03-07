"""网关类型定义

包含枚举类型和数据类定义
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


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
    API_KEY = "api_key"


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
    headers: dict
    query_params: dict
    body: Optional[bytes] = None
    client_ip: str = ""
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ApiResponse:
    """API响应"""
    status_code: int
    headers: dict
    body: bytes
    processing_time: float
    upstream_url: str = ""
    cached: bool = False


__all__ = [
    'HttpMethod',
    'ServiceStatus',
    'RateLimitType',
    'ServiceEndpoint',
    'RateLimitRule',
    'RouteRule',
    'ApiRequest',
    'ApiResponse'
]

