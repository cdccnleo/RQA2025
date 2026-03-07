"""
网关路由模块
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class RouteRule:
    """路由规则"""
    path: str
    method: str = "GET"
    target: Optional[str] = None
    middleware: List[str] = None
    
    def __post_init__(self):
        if self.middleware is None:
            self.middleware = []


__all__ = ['RouteRule']

