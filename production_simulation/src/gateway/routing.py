"""
网关路由模块

提供路由规则定义和路由管理功能
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List


@dataclass
class RouteRule:
    """路由规则"""
    path: str
    method: str = "GET"
    handler: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None
    middleware: Optional[List[Callable]] = None
    
    def matches(self, request_path: str, request_method: str) -> bool:
        """检查请求是否匹配该路由"""
        return self.path == request_path and self.method == request_method


class RouteRegistry:
    """路由注册表"""
    
    def __init__(self):
        self.routes: List[RouteRule] = []
    
    def add_route(self, rule: RouteRule) -> bool:
        """添加路由规则"""
        self.routes.append(rule)
        return True
    
    def find_route(self, path: str, method: str) -> Optional[RouteRule]:
        """查找匹配的路由"""
        for route in self.routes:
            if route.matches(path, method):
                return route
        return None
    
    def list_routes(self) -> List[RouteRule]:
        """列出所有路由"""
        return self.routes.copy()


__all__ = ['RouteRule', 'RouteRegistry']

