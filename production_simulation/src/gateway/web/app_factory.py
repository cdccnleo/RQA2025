import time
from src.infrastructure.logging.core.unified_logger import get_unified_logger

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from src.infrastructure.health.services.health_check_service import HealthCheck
from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager as ConfigManager  # 合理跨层级导入：基础设施层配置管理
from src.infrastructure.error.error_handler import ErrorHandler  # 合理跨层级导入：基础设施层错误处理

# 可选导入：基础设施层资源API
try:
    from src.infrastructure.resource.resource_api import ResourceAPI
except ImportError:
    ResourceAPI = None
try:
    from src.infrastructure.resource.gpu_manager import GPUManager
except ImportError:
    GPUManager = None
try:
    from src.infrastructure.resource.resource_manager import ResourceManager
except ImportError:
    ResourceManager = None
from .data_api import router as data_router  # 当前层级内部导入：数据API路由
from .websocket_api import router as websocket_router  # 当前层级内部导入：WebSocket API路由
import logging


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
        for key, value in config.items():
            config_manager.update_config(key, value)

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

    # 初始化错误处理器
    error_handler = ErrorHandler()

    # 初始化应用监控
    app.state.monitor = ApplicationMonitor(
        app_name="rqa2025_api",
        influx_config=config_manager.get("monitoring.influxdb")
    )

    # 添加根路由
    @app.get("/")
    async def root():
        return {"message": "RQA2025 API"}

    # 添加健康检查路由
    @app.get("/health")
    async def health():
        try:
            checks = app.state.monitor.run_health_checks()
            return {"status": "healthy", "checks": checks}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # 添加指标路由
    @app.get("/metrics")
    async def metrics():
        try:
            metrics = app.state.monitor.get_metrics()
            return metrics
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 添加路由
    app.include_router(health_check.router, prefix="/api / v1")
    app.include_router(resource_api.router, prefix="/api / v1")
    app.include_router(data_router, prefix="/api / v1")
    app.include_router(websocket_router)

    # 添加中间件
    @app.middleware("http")
    async def monitor_requests(request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # 记录请求指标
            app.state.monitor.record_metric(
                "request_duration",
                process_time,
                tags={"method": request.method, "path": request.url.path}
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time

            # 记录错误
            error_handler.handle(e, context={
                "method": request.method,
                "path": request.url.path,
                "duration": process_time
            })

            # 记录错误指标
            app.state.monitor.record_metric(
                "request_error",
                1,
                tags={"method": request.method, "path": request.url.path}
            )

            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    logger.info("Application initialized successfully")
    return app
