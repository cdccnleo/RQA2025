"""
基础设施层Web服务模块

功能：
1. FastAPI应用工厂
2. API路由管理
3. 应用监控集成
4. 前端性能优化

使用示例：
    from src.engine.web import create_app
    from src.infrastructure.config import ConfigManager

    # 创建应用实例
    config = ConfigManager().get("web")
    app = create_app(config)

    # 启动应用
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

版本历史：
- v1.0 (2024 - 03 - 01): 初始版本
- v1.1 (2024 - 04 - 15): 添加应用监控集成
- v1.2 (2026 - 02 - 13): 添加前端性能优化组件
"""
from .app_factory import create_app

# 导入前端优化器
try:
    from .frontend_optimizer import (
        FrontendOptimizer,
        FrontendOptimizerConfig,
        ResponseCache,
        DataPaginator,
        DataFilter,
        ResponseCompressor,
        PaginatedResponse,
        CompressionType,
        CacheControl,
        get_global_optimizer,
        clear_global_optimizer,
    )
except ImportError:
    class FrontendOptimizer:
        pass
    class FrontendOptimizerConfig:
        pass
    class ResponseCache:
        pass
    class DataPaginator:
        pass
    class DataFilter:
        pass
    class ResponseCompressor:
        pass
    class PaginatedResponse:
        pass
    class CompressionType:
        pass
    class CacheControl:
        pass
    
    def get_global_optimizer(config=None):
        pass
    
    def clear_global_optimizer():
        pass

__all__ = [
    'create_app',
    # 前端优化器
    'FrontendOptimizer',
    'FrontendOptimizerConfig',
    'ResponseCache',
    'DataPaginator',
    'DataFilter',
    'ResponseCompressor',
    'PaginatedResponse',
    'CompressionType',
    'CacheControl',
    'get_global_optimizer',
    'clear_global_optimizer',
]
