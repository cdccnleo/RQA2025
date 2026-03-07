"""
Core API Gateway模块别名

向后兼容的导入路径
"""

from .integration.apis.api_gateway import (
    ApiGateway, APIGateway, APIGatewayManager,
    RouteRule, ServiceEndpoint, RateLimitRule,
    HttpMethod, ServiceStatus, RateLimitType,
    CircuitBreaker, ApiRequest, ApiResponse, LoadBalancer, RateLimiter
)

__all__ = [
    'ApiGateway', 'APIGateway', 'APIGatewayManager',
    'RouteRule', 'ServiceEndpoint', 'RateLimitRule',
    'HttpMethod', 'ServiceStatus', 'RateLimitType',
    'CircuitBreaker', 'ApiRequest', 'ApiResponse', 'LoadBalancer', 'RateLimiter'
]

