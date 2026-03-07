"""
API网关模块（别名模块）
提供向后兼容的导入路径
"""

try:
    from src.core.api_gateway import APIGateway, APIGatewayManager, ApiGateway
except ImportError:
    # 提供基础实现
    class APIGateway:
        pass
    
    APIGatewayManager = APIGateway
    ApiGateway = APIGateway

__all__ = ['APIGateway', 'APIGatewayManager', 'ApiGateway']

