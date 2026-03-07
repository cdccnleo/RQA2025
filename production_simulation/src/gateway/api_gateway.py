"""
API网关模块（别名模块）
提供向后兼容的导入路径
"""

try:
    # 优先使用核心层的完整API网关
    from src.core.api_gateway import APIGateway, APIGatewayManager, ApiGateway
except ImportError:
    try:
        # 降级使用本层api/目录的网关实现
        from .api.api_gateway import ApiGateway
        APIGateway = ApiGateway
        APIGatewayManager = ApiGateway
    except ImportError:
        # 提供基础实现
        class APIGateway:
            pass
        
        APIGatewayManager = APIGateway
        ApiGateway = APIGateway

__all__ = ['APIGateway', 'APIGatewayManager', 'ApiGateway']

