"""
基础设施层Web服务模块

功能：
1. FastAPI应用工厂
2. API路由管理
3. 应用监控集成

使用示例：
    from src.infrastructure.web import create_app
    from src.infrastructure.config import ConfigManager

    # 创建应用实例
    config = ConfigManager().get("web")
    app = create_app(config)

    # 启动应用
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

版本历史：
- v1.0 (2024-03-01): 初始版本
- v1.1 (2024-04-15): 添加应用监控集成
"""
from .app_factory import create_app

__all__ = [
    'create_app'
]
