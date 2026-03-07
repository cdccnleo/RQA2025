import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
RQA2025 统一Web管理界面
整合所有模块的Web管理功能，提供统一的访问入口

功能模块：
1. 系统概览 - 整体系统状态监控
2. 配置管理 - 统一配置管理界面
3. 策略管理 - 策略配置和监控
4. 数据管理 - 数据源和数据集管理
5. 回测管理 - 回测配置和结果查看
6. 监控告警 - 系统监控和告警管理
7. 资源管理 - 计算资源监控和管理
8. 用户管理 - 用户权限和访问控制

设计特点：
- 现代化响应式设计
- 模块化组件架构
- 统一认证和权限控制
- 实时数据更新
- 可扩展的插件系统
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from src.infrastructure.logging.core.unified_logger import get_unified_logger  # 当前层级内部导入：统一日志器
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager  # 合理跨层级导入：基础设施层配置管理
from src.infrastructure.monitoring.application_monitor import ApplicationMonitor  # 合理跨层级导入：基础设施层应用监控
from src.infrastructure.resource.resource_manager import ResourceManager  # 合理跨层级导入：基础设施层资源管理
# from src.infrastructure.health.services.health_check_service import HealthCheck  # 合理跨层级导入：基础设施层健康检查

logger = logging.getLogger(__name__)


class DashboardConfig(BaseModel):

    """仪表板配置模型"""
    title: str = "RQA2025 统一管理平台"
    version: str = "1.0.0"
    theme: str = "modern"
    refresh_interval: int = 30  # 秒
    max_connections: int = 100
    enable_websocket: bool = True
    enable_real_time: bool = True


class ModuleInfo(BaseModel):

    """模块信息模型"""
    name: str
    display_name: str
    description: str
    icon: str
    route: str
    enabled: bool = True
    permissions: List[str] = []


class UnifiedDashboard:

    """统一Web管理界面"""

    def __init__(self, config: Optional[DashboardConfig] = None):

        self.config = config or DashboardConfig()
        self.app = FastAPI(
            title=self.config.title,
            description="RQA2025量化交易系统统一管理平台",
            version=self.config.version,
            docs_url="/api / docs",
            redoc_url="/api / redoc"
        )

        # 初始化核心组件
        self.config_manager = UnifiedConfigManager()
        self.resource_manager = ResourceManager()
        self.health_check = HealthCheck()
        self.app_monitor = ApplicationMonitor("rqa2025_dashboard")

        # WebSocket连接管理
        self.active_connections: List[WebSocket] = []

        # 模块注册表
        self.modules: Dict[str, ModuleInfo] = {}

        # 初始化模板和静态文件
        self._setup_static_files()
        self._setup_templates()

        # 注册模块
        self._register_modules()

        # 设置路由
        self._setup_routes()

        # 设置中间件
        self._setup_middleware()

        # 集成模块路由
        self._integrate_module_routes()

        # 初始化模块（异步）
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(self._initialize_modules())
        else:
            # 如果没有运行的事件循环，延迟初始化
            pass

    def _setup_static_files(self):
        """设置静态文件服务"""
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _setup_templates(self):
        """设置模板引擎"""
        templates_dir = Path(__file__).parent / "templates"
        templates_dir.mkdir(exist_ok=True)
        self.templates = Jinja2Templates(directory=str(templates_dir))

    def _register_modules(self):
        """注册管理模块"""
        # 导入模块化组件
        from .modules import ModuleRegistry, ModuleFactory
        from .modules.config_module import ConfigModule
        from .modules.fpga_module import FPGAModule, FPGAModuleConfig
        from .modules.resource_module import ResourceModule, ResourceModuleConfig
        from .modules.features_module import FeaturesModule, FeaturesModuleConfig

        # 初始化模块注册表和工厂
        self.module_registry = ModuleRegistry()
        self.module_factory = ModuleFactory()

        # 注册配置管理模块
        config_config = ConfigModule.get_default_config()
        self.module_registry.register_module(ConfigModule, config_config)

        # 注册FPGA监控模块
        fpga_config = FPGAModuleConfig(
            name="fpga_monitoring",
            display_name="FPGA监控",
            description="FPGA性能监控和告警管理",
            icon="fas fa - microchip",
            route="/fpga",
            refresh_interval=5,
            alert_thresholds={
                "latency": 1.0,
                "utilization": 90.0,
                "temperature": 85.0,
                "power": 200.0
            },
            enable_alerts=True,
            enable_history=True,
            history_hours=24
        )
        self.module_registry.register_module(FPGAModule, fpga_config)

        # 注册资源监控模块
        resource_config = ResourceModuleConfig(
            name="resource_monitoring",
            display_name="资源监控",
            description="系统资源使用情况监控",
            icon="fas fa - server",
            route="/resources",
            refresh_interval=5,
            alert_thresholds={
                "cpu_usage": 90.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "gpu_memory_usage": 90.0,
                "gpu_utilization": 95.0,
                "gpu_temperature": 85.0
            },
            enable_alerts=True,
            enable_history=True,
            history_hours=24,
            monitor_gpu=True,
            monitor_network=True
        )
        self.module_registry.register_module(ResourceModule, resource_config)

        # 注册特征层监控模块
        features_config = FeaturesModuleConfig(
            name="features_monitoring",
            display_name="特征监控",
            description="特征工程性能和数据质量监控",
            icon="fas fa - chart - line",
            route="/features",
            refresh_interval=5,
            enable_real_time_monitoring=True,
            enable_data_quality_monitoring=True,
            enable_performance_monitoring=True,
            enable_config_monitoring=True,
            alert_thresholds={
                "feature_engineering_time": 1000.0,
                "data_quality_score": 80.0,
                "memory_usage": 85.0,
                "cpu_usage": 90.0,
                "error_rate": 5.0
            },
            history_hours=24,
            auto_refresh=True,
            output_dir="./features_dashboard"
        )
        self.module_registry.register_module(FeaturesModule, features_config)

        # 注册其他模块（这里可以动态发现和注册）
        modules = [
            ModuleInfo(
                name="system_overview",
                display_name="系统概览",
                description="系统整体状态监控",
                icon="fas fa - tachometer - alt",
                route="/system",
                enabled=True,
                permissions=["read"]
            ),
            ModuleInfo(
                name="strategy_management",
                display_name="策略管理",
                description="策略配置和监控",
                icon="fas fa - chart - line",
                route="/strategy",
                enabled=True,
                permissions=["read", "write"]
            ),
            ModuleInfo(
                name="data_management",
                display_name="数据管理",
                description="数据源和数据集管理",
                icon="fas fa - database",
                route="/data",
                enabled=True,
                permissions=["read", "write"]
            ),
            ModuleInfo(
                name="backtest_management",
                display_name="回测管理",
                description="回测配置和结果查看",
                icon="fas fa - chart - bar",
                route="/backtest",
                enabled=True,
                permissions=["read", "write"]
            ),
            ModuleInfo(
                name="features_monitoring",
                display_name="特征层监控",
                description="特征工程性能监控和数据质量监控",
                icon="fas fa - chart - line",
                route="/features",
                enabled=True,
                permissions=["read", "write"]
            ),
            ModuleInfo(
                name="monitoring_alerts",
                display_name="监控告警",
                description="系统监控和告警管理",
                icon="fas fa - bell",
                route="/monitoring",
                enabled=True,
                permissions=["read", "write"]
            ),
            ModuleInfo(
                name="user_management",
                display_name="用户管理",
                description="用户权限和访问控制",
                icon="fas fa - users",
                route="/users",
                enabled=True,
                permissions=["admin"]
            )
        ]

        for module in modules:
            self.modules[module.name] = module

        # 初始化所有模块
        # 启动模块初始化（在异步环境中）
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._initialize_modules())
            else:
                # 在非异步环境中，延迟初始化
                pass
        except RuntimeError:
            # 没有运行的事件循环，延迟初始化
            pass

    async def _initialize_modules(self):
        """初始化所有模块"""
        try:
            # 初始化模块注册表中的模块
            init_results = await self.module_registry.initialize_all_modules()
            logger.info(f"模块初始化结果: {init_results}")

            # 启动所有模块
            start_results = await self.module_registry.start_all_modules()
            logger.info(f"模块启动结果: {start_results}")

            # 注册模块路由
            for name, module in self.module_registry.get_enabled_modules().items():
                self.app.include_router(module.get_router(), prefix=f"/api/{name}")
                logger.info(f"注册模块路由: {name}")

        except Exception as e:
            logger.error(f"模块初始化失败: {str(e)}")

    def _setup_routes(self):
        """设置路由"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """主仪表板页面"""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": {},
                    "config": self.config,
                    "modules": list(self.modules.values()),
                    "system_info": await self._get_system_info()
                }
            )

        @self.app.get("/api / modules")
        async def get_modules():
            """获取模块列表"""
            return {
                "modules": list(self.modules.values()),
                "total": len(self.modules)
            }

        @self.app.get("/api / system / overview")
        async def get_system_overview():
            """获取系统概览数据"""
            return await self._get_system_overview()

        @self.app.get("/api / config")
        async def get_config():
            """获取配置信息"""
            return await self._get_config_info()

        @self.app.get("/api / strategy")
        async def get_strategies():
            """获取策略列表"""
            return await self._get_strategies()

        @self.app.get("/api / data")
        async def get_data_sources():
            """获取数据源信息"""
            return await self._get_data_sources()

        @self.app.get("/api / backtest")
        async def get_backtests():
            """获取回测信息"""
            return await self._get_backtests()

        @self.app.get("/api / monitoring")
        async def get_monitoring():
            """获取监控信息"""
            return await self._get_monitoring_info()

        @self.app.get("/api / resources")
        async def get_resources():
            """获取资源信息"""
            return await self._get_resource_info()

        @self.app.get("/api / health")
        async def health_check():
            """健康检查"""
            return await self._get_health_status()

        # 集成模块路由
        self._integrate_module_routes()

        # WebSocket路由
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket连接端点"""
            await self._handle_websocket(websocket)

    def _integrate_module_routes(self):
        """集成模块路由到主应用"""
        try:
            # 获取所有注册的模块
            registered_modules = self.module_registry.get_all_modules()
            logger.info(f"注册的模块: {list(registered_modules.keys())}")

            for module_name, module_instance in registered_modules.items():
                if hasattr(module_instance, 'router') and module_instance.router:
                    # 将模块的路由包含到主应用中
                    self.app.include_router(
                        module_instance.router,
                        prefix=f"/api / modules/{module_name}",
                        tags=[module_instance.config.display_name]
                    )
                    logger.info(f"集成模块路由: {module_name} -> /api / modules/{module_name}")
                else:
                    logger.warning(f"模块 {module_name} 没有路由器")

        except Exception as e:
            logger.error(f"集成模块路由失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _setup_middleware(self):
        """设置中间件"""

        @self.app.middleware("http")
        async def dashboard_middleware(request, call_next):
            """仪表板中间件"""
            start_time = datetime.now()

            # 记录请求
            logger.info(f"Dashboard request: {request.method} {request.url.path}")

            try:
                response = await call_next(request)
                process_time = (datetime.now() - start_time).total_seconds()

                # 记录性能指标
                self.app_monitor.record_metric(
                    "dashboard_request_duration",
                    process_time,
                    tags={"method": request.method, "path": request.url.path}
                )

                return response

            except Exception as e:
                logger.error(f"Dashboard error: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )

    async def _handle_websocket(self, websocket: WebSocket):
        """处理WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)

        try:
            while True:
                # 接收消息
                data = await websocket.receive_text()
                message = json.loads(data)

                # 处理消息
                response = await self._process_websocket_message(message)

                # 发送响应
                await websocket.send_text(json.dumps(response))

        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def _process_websocket_message(self, message: Dict) -> Dict:
        """处理WebSocket消息"""
        msg_type = message.get("type")

        if msg_type == "subscribe":
            return {"type": "subscribed", "data": "Success"}
        elif msg_type == "get_system_info":
            return {
                "type": "system_info",
                "data": await self._get_system_info()
            }
        elif msg_type == "get_metrics":
            return {
                "type": "metrics",
                "data": await self._get_real_time_metrics()
            }
        else:
            return {"type": "error", "data": "Unknown message type"}

    async def _get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            "platform": "RQA2025",
            "version": self.config.version,
            "uptime": await self._get_uptime(),
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }

    async def _get_system_overview(self) -> Dict:
        """获取系统概览数据"""
        return {
            "system": await self._get_system_info(),
            "resources": await self._get_resource_summary(),
            "alerts": await self._get_active_alerts(),
            "performance": await self._get_performance_metrics()
        }

    async def _get_config_info(self) -> Dict:
        """获取配置信息"""
        try:
            config_data = self.config_manager.get_all_configs()
            return {
                "configs": config_data,
                "total_configs": len(config_data),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取配置信息失败: {str(e)}")
            return {"error": str(e)}

    async def _get_strategies(self) -> Dict:
        """获取策略信息"""
        # 这里应该从策略管理器获取数据
        return {
            "strategies": [],
            "total_strategies": 0,
            "active_strategies": 0
        }

    async def _get_data_sources(self) -> Dict:
        """获取数据源信息"""
        # 这里应该从数据管理器获取数据
        return {
            "data_sources": [],
            "total_sources": 0,
            "active_sources": 0
        }

    async def _get_backtests(self) -> Dict:
        """获取回测信息"""
        # 这里应该从回测管理器获取数据
        return {
            "backtests": [],
            "total_backtests": 0,
            "running_backtests": 0
        }

    async def _get_monitoring_info(self) -> Dict:
        """获取监控信息"""
        try:
            metrics = self.app_monitor.get_metrics()
            return {
                "metrics": metrics,
                "alerts": await self._get_active_alerts(),
                "health_status": await self._get_health_status()
            }
        except Exception as e:
            logger.error(f"获取监控信息失败: {str(e)}")
            return {"error": str(e)}

    async def _get_resource_info(self) -> Dict:
        """获取资源信息"""
        try:
            return {
                "system_resources": await self._get_system_resources(),
                "gpu_resources": await self._get_gpu_resources(),
                "quota_usage": await self._get_quota_usage()
            }
        except Exception as e:
            logger.error(f"获取资源信息失败: {str(e)}")
            return {"error": str(e)}

    async def _get_health_status(self) -> Dict:
        """获取健康状态"""
        try:
            checks = self.health_check.run_all_checks()
            return {
                "status": "healthy" if all(c["status"] == "ok" for c in checks) else "unhealthy",
                "checks": checks,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _get_uptime(self) -> str:
        """获取系统运行时间"""
        # 这里应该从系统启动时间计算
        return "24h 30m 15s"

    async def _get_resource_summary(self) -> Dict:
        """获取资源摘要"""
        return {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.4,
            "gpu_usage": 12.5
        }

    async def _get_active_alerts(self) -> List[Dict]:
        """获取活跃告警"""
        return []

    async def _get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            "response_time": 125.6,
            "throughput": 1500,
            "error_rate": 0.02,
            "availability": 99.95
        }

    async def _get_system_resources(self) -> Dict:
        """获取系统资源"""
        return {
            "cpu": {"usage": 45.2, "cores": 16},
            "memory": {"usage": 67.8, "total": "32GB"},
            "disk": {"usage": 23.4, "total": "1TB"},
            "network": {"in": 1024, "out": 2048}
        }

    async def _get_gpu_resources(self) -> Dict:
        """获取GPU资源"""
        return {
            "gpu_count": 2,
            "gpu_usage": 12.5,
            "memory_usage": 34.2,
            "temperature": 65.0
        }

    async def _get_quota_usage(self) -> Dict:
        """获取配额使用情况"""
        return {
            "cpu_quota": {"used": 30, "total": 100},
            "memory_quota": {"used": 8, "total": 32},
            "gpu_quota": {"used": 1, "total": 2}
        }

    async def _get_real_time_metrics(self) -> Dict:
        """获取实时指标"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": await self._get_performance_metrics(),
            "resources": await self._get_resource_summary()
        }

    async def broadcast_update(self, message: Dict):
        """广播更新消息"""
        if not self.active_connections:
            return

        message_text = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"WebSocket发送失败: {str(e)}")
                disconnected.append(connection)

        # 清理断开的连接
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

    def run(self, host: str = "0.0.0.0", port: int = 8080, **kwargs):
        """启动Web服务"""
        logger.info(f"启动统一Web管理界面: http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


def create_dashboard(config: Optional[DashboardConfig] = None) -> UnifiedDashboard:
    """创建统一Web管理界面实例"""
    return UnifiedDashboard(config)


if __name__ == "__main__":
    # 创建并启动仪表板
    dashboard = create_dashboard()
    dashboard.run(host="0.0.0.0", port=8080, reload=True)
