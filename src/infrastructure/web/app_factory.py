import time

from fastapi import FastAPI
from src.infrastructure.health.health_check import HealthCheck
from ..monitoring.application_monitor import ApplicationMonitor
from ..config.config_manager import ConfigManager
import logging

from ..monitoring.resource_api import ResourceAPI
from ..resource import GPUManager

logger = logging.getLogger(__name__)

def create_app(config: dict = None) -> FastAPI:
    """创建FastAPI应用实例"""
    
    # 初始化应用
    app = FastAPI(
        title="RQA2025量化平台",
        description="A股量化交易系统API",
        version="1.0.0"
    )
    
    # 加载配置
    config_manager = ConfigManager()
    if config:
        config_manager.update_config(config)
    
    # 初始化健康检查
    health_check = HealthCheck()
    
    # 添加核心依赖检查
    def check_database():
        # 实际项目中替换为真实数据库检查
        return {"status": "ok", "tables": 42}
    
    def check_cache():
        # 实际项目中替换为真实缓存检查
        return {"status": "ok", "items": 100}
    
    health_check.add_dependency_check("database", check_database)
    health_check.add_dependency_check("cache", check_cache)
    
    # 初始化资源监控API
    resource_api = ResourceAPI(
        resource_manager=ResourceManager(),
        gpu_manager=GPUManager()
    )
    
    # 添加路由
    app.include_router(health_check.router, prefix="/api/v1")
    app.include_router(resource_api.router, prefix="/api/v1")
    
    # 初始化应用监控
    app.state.monitor = ApplicationMonitor(
        app_name="rqa2025_api",
        influx_config=config_manager.get("monitoring.influxdb")
    )
    
    # 添加中间件
    @app.middleware("http")
    async def monitor_requests(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        app.state.monitor.record_function(
            name=f"http_{request.method}_{request.url.path}",
            execution_time=process_time,
            success=response.status_code < 500
        )
        
        return response
    
    logger.info("Application initialized successfully")
    return app
