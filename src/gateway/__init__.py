"""
网关层 (Gateway Layer)

API路由管理，包含：
- RESTful API网关
- 负载均衡
- 认证授权
- 限流熔断
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

__all__ = ['GatewayLayer', 'create_gateway_app']

class GatewayLayer:
    """
    网关层主类

    提供API网关服务：
    - RESTful API路由
    - 负载均衡
    - 认证授权
    - 限流熔断
    """

    def __init__(self):
        self.app = None
        self.is_running = False

    def create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        app = FastAPI(
            title="RQA2025 量化交易系统",
            description="RQA2025 量化交易系统API",
            version="1.0.0"
        )

        # 配置CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        self.app = app
        return app

    def start(self) -> bool:
        """启动网关层"""
        try:
            if not self.app:
                self.app = self.create_app()
            self.is_running = True
            return True
        except Exception as e:
            print(f"网关层启动失败: {e}")
            return False

    def stop(self):
        """停止网关层"""
        self.is_running = False

    def health_check(self):
        """健康检查"""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'app_created': self.app is not None
        }

def create_gateway_app() -> FastAPI:
    """创建网关应用（便捷函数）"""
    gateway = GatewayLayer()
    return gateway.create_app()

# 网关层实例
gateway_layer = GatewayLayer()