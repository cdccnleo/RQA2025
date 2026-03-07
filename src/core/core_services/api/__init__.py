"""
API服务组件

提供API相关服务：
- APIService: API服务
- APIGateway: API网关
"""

try:
    from .api_service import TradingAPIService as APIService
    # APIGateway可能在其他地方定义，这里先提供基础实现
    class APIGateway:
        pass
except ImportError:
    # 如果导入失败，提供基础实现
    class APIService:
        pass
    class APIGateway:
        pass

__all__ = [
    "APIService",
    "APIGateway"
]
