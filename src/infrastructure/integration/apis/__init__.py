"""
API集成模块
"""

try:
    from .api_gateway import APIGateway, APIGatewayManager
except ImportError:
    try:
        from ..gateway.api_gateway import APIGateway, APIGatewayManager
    except ImportError:
        # 提供基础实现
        class APIGateway:
            pass
        
        APIGatewayManager = APIGateway

__all__ = ['APIGateway', 'APIGatewayManager']

