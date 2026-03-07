"""
API模块

提供FastAPI应用实例和路由注册
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import pipeline, monitoring, alerts


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例
    
    Returns:
        FastAPI应用实例
    """
    app = FastAPI(
        title="Quant Trading System API",
        description="量化交易系统API - ML管道监控和管理",
        version="1.0.0"
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(pipeline.router)
    app.include_router(monitoring.router)
    app.include_router(alerts.router)
    
    # 健康检查端点
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "service": "quant-trading-api",
            "version": "1.0.0"
        }
    
    # 根端点
    @app.get("/")
    async def root():
        """API根路径"""
        return {
            "message": "Quant Trading System API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    return app


# 创建应用实例
app = create_app()
