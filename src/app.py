#!/usr/bin/env python3
"""
RQA2025量化交易系统应用入口

整合所有核心服务，提供完整的应用功能。
支持REST API、WebSocket实时数据、后台任务处理等。
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.core_services.api.api_service import create_trading_api_app
from src.core.core_services.core.database_service import get_database_service
from src.core.core_services.core.business_service import get_business_service
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
from src.monitoring.monitoring_system import MonitoringSystem

logger = get_logger(__name__)


class RQAApplication:
    """RQA应用主类"""

    def __init__(self) -> Any:
        """__init__ 函数的文档字符串"""

        self.config_manager = UnifiedConfigManager()
        self.database_service = None
        self.business_service = None
        self.monitoring_system = None
        self.app_monitor = None

        # FastAPI应用
        self.app = None

        # 运行状态
        self.running = False
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """初始化应用"""
        try:
            logger.info("开始初始化RQA2025应用...")

            # 初始化核心服务
            self.database_service = await get_database_service()
            logger.info("数据库服务初始化完成")

            self.business_service = await get_business_service()
            logger.info("业务服务初始化完成")

            # 初始化监控系统
            try:
                self.monitoring_system = MonitoringSystem()
                logger.info("监控系统初始化完成")
            except Exception as e:
                logger.warning(f"监控系统初始化失败: {e}")
                self.monitoring_system = None

            # 初始化应用监控
            try:
                self.app_monitor = ApplicationMonitor()
                logger.info("应用监控初始化完成")
            except Exception as e:
                logger.warning(f"应用监控初始化失败: {e}")
                self.app_monitor = None

            # 创建FastAPI应用
            self.app = await self.create_fastapi_app()

            logger.info("RQA2025应用初始化完成")

        except Exception as e:
            logger.error(f"应用初始化失败: {e}")
            raise

    async def create_fastapi_app(self) -> FastAPI:
        """创建FastAPI应用"""

        # 生命周期管理
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # 启动
            logger.info("RQA2025应用启动...")
            self.running = True

            # 启动后台任务
            asyncio.create_task(self.background_tasks())

            yield

            # 关闭
            logger.info("RQA2025应用关闭...")
            self.running = False
            self.shutdown_event.set()

            # 清理资源
            if self.database_service:
                await self.database_service.close()

        # 创建主应用
        app = FastAPI(
            title="RQA2025量化交易平台",
            description="A股量化交易系统 - 完整的交易解决方案",
            version="1.0.0",
            lifespan=lifespan
        )

        # 添加中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境应该配置具体域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 挂载交易API
        from src.core.core_services.api.api_service import trading_api_app
        app.mount("/api/v1/trading", trading_api_app)

        # 添加主路由
        @app.get("/", response_class=HTMLResponse)
        async def root():
            """根路径"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>RQA2025量化交易平台</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #2c3e50; text-align: center; }
                    .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                    .healthy { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                    .degraded { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
                    .api-links { margin: 20px 0; }
                    .api-links a { display: inline-block; margin: 5px 10px; padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
                    .api-links a:hover { background: #0056b3; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 RQA2025量化交易平台</h1>
                    <p style="text-align: center; color: #666;">A股量化交易系统 - 专业的交易解决方案</p>

                    <div class="status healthy">
                        <strong>✅ 系统状态:</strong> 正常运行
                    </div>

                    <h3>📋 API接口</h3>
                    <div class="api-links">
                        <a href="/docs">📖 API文档</a>
                        <a href="/api/v1/trading/docs">🏛️ 交易API</a>
                        <a href="/health">💚 健康检查</a>
                        <a href="/metrics">📊 系统指标</a>
                    </div>

                    <h3>🔧 核心功能</h3>
                    <ul>
                        <li>✅ 用户管理与认证</li>
                        <li>✅ 实时交易订单处理</li>
                        <li>✅ 投资组合管理</li>
                        <li>✅ 量化策略执行</li>
                        <li>✅ 风险控制系统</li>
                        <li>✅ 实时市场数据</li>
                        <li>✅ 系统监控告警</li>
                    </ul>

                    <h3>📈 系统架构</h3>
                    <ul>
                        <li><strong>前端:</strong> REST API + WebSocket</li>
                        <li><strong>后端:</strong> FastAPI + AsyncPG + Redis</li>
                        <li><strong>数据:</strong> PostgreSQL + Redis缓存</li>
                        <li><strong>部署:</strong> Docker + Kubernetes</li>
                    </ul>
                </div>
            </body>
            </html>
            """

        @app.get("/health")
        async def health_check():
            """健康检查"""
            try:
                health_data = {}

                # 数据库健康检查
                if self.database_service:
                    health_data["database"] = await self.database_service.health_check()

                # 业务服务健康检查
                if self.business_service:
                    health_data["business"] = await self.business_service.health_check()

                # 监控系统健康检查
                if self.monitoring_system:
                    try:
                        health_data["monitoring"] = await self.monitoring_system.health_check()
                    except:
                        health_data["monitoring"] = {"status": "error"}

                # 总体状态
                services_status = [h.get("status", "unknown")
                                   for h in health_data.values()]
                overall_status = "healthy" if all(
                    s in ["healthy", "simulated"] for s in services_status) else "degraded"

                return {
                    "status": overall_status,
                    "timestamp": asyncio.get_event_loop().time(),
                    "version": "1.0.0",
                    "services": health_data
                }

            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                }

        @app.get("/metrics")
        async def get_metrics():
            """获取系统指标"""
            try:
                metrics = {
                    "application": {
                        "uptime": asyncio.get_event_loop().time(),
                        "version": "1.0.0"
                    }
                }

                # 数据库指标
                if self.database_service:
                    db_health = await self.database_service.health_check()
                    metrics["database"] = {
                        "status": db_health.get("status"),
                        "response_time": db_health.get("response_time"),
                        "connections": db_health.get("connection_pool_size", 0)
                    }

                # 业务指标
                if self.business_service:
                    business_health = await self.business_service.health_check()
                    metrics["business"] = {
                        "active_processes": business_health.get("active_processes", 0),
                        "active_strategies": business_health.get("active_strategies", 0),
                        "components_status": business_health.get("components", {})
                    }

                return metrics

            except Exception as e:
                logger.error(f"获取指标失败: {e}")
                return {"error": str(e)}

        # 业务流程管理API
        @app.post("/api/v1/business/strategies")
        async def create_strategy(request: Request):
            """创建交易策略"""
            try:
                data = await request.json()
                user_id = data.get("user_id", 1)  # 临时默认值
                strategy_data = data.get("strategy", {})

                if not self.business_service:
                    return JSONResponse(status_code=503, content={"error": "业务服务不可用"})

                result = await self.business_service.create_strategy(user_id, strategy_data)
                return result

            except Exception as e:
                logger.error(f"创建策略失败: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.post("/api/v1/business/strategies/{strategy_id}/execute")
        async def execute_strategy(strategy_id: str, request: Request):
            """执行策略"""
            try:
                data = await request.json()
                market_data = data.get("market_data", {})

                if not self.business_service:
                    return JSONResponse(status_code=503, content={"error": "业务服务不可用"})

                result = await self.business_service.execute_strategy(strategy_id, market_data)
                return result

            except Exception as e:
                logger.error(f"执行策略失败: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.post("/api/v1/business/portfolio/rebalance")
        async def rebalance_portfolio(request: Request):
            """组合再平衡"""
            try:
                data = await request.json()
                user_id = data.get("user_id", 1)
                target_allocation = data.get("target_allocation", {})

                if not self.business_service:
                    return JSONResponse(status_code=503, content={"error": "业务服务不可用"})

                result = await self.business_service.rebalance_portfolio(user_id, target_allocation)
                return result

            except Exception as e:
                logger.error(f"组合再平衡失败: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.get("/api/v1/business/processes/{process_id}")
        async def get_process_status(process_id: str):
            """获取流程状态"""
            try:
                if not self.business_service:
                    return JSONResponse(status_code=503, content={"error": "业务服务不可用"})

                result = await self.business_service.get_process_status(process_id)
                if result:
                    return result
                else:
                    return JSONResponse(status_code=404, content={"error": "流程不存在"})

            except Exception as e:
                logger.error(f"获取流程状态失败: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.post("/api/v1/business/analysis/market")
        async def analyze_market(request: Request):
            """市场数据分析"""
            try:
                data = await request.json()
                symbols = data.get("symbols", [])
                analysis_type = data.get("analysis_type", "technical")

                if not self.business_service:
                    return JSONResponse(status_code=503, content={"error": "业务服务不可用"})

                result = await self.business_service.analyze_market_data(symbols, analysis_type)
                return result

            except Exception as e:
                logger.error(f"市场分析失败: {e}")
                return JSONResponse(status_code=500, content={"error": str(e)})

        return app

    async def background_tasks(self):
        """后台任务"""
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    # 每30秒执行一次健康检查和指标收集
                    await asyncio.sleep(30)

                    # 收集应用指标
                    if self.app_monitor:
                        try:
                            await self.app_monitor.collect_metrics()
                        except Exception as e:
                            logger.warning(f"应用指标收集失败: {e}")

                    # 检查业务流程状态
                    if self.business_service:
                        try:
                            # 清理完成的流程（保留最近1小时的）
                            current_time = asyncio.get_event_loop().time()
                            cutoff_time = current_time - 3600  # 1小时前

                            processes_to_remove = []
                            for process_id, process in self.business_service.active_processes.items():
                                if (process.status in [process.status.__class__.COMPLETED,
                                                       process.status.__class__.FAILED,
                                                       process.status.__class__.CANCELLED] and
                                    process.completed_at and
                                        (asyncio.get_event_loop().time() - process.completed_at.timestamp()) > 3600):
                                    processes_to_remove.append(process_id)

                            for process_id in processes_to_remove:
                                del self.business_service.active_processes[process_id]

                            if processes_to_remove:
                                logger.info(f"清理了 {len(processes_to_remove)} 个过期的业务流程")

                        except Exception as e:
                            logger.warning(f"业务流程清理失败: {e}")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"后台任务执行失败: {e}")
                    await asyncio.sleep(10)  # 出错后等待10秒再试

        except Exception as e:
            logger.error(f"后台任务异常退出: {e}")

    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行应用"""
        try:
            # 设置信号处理
            def signal_handler(signum, frame) -> Any:
                """signal_handler 函数的文档字符串"""

                logger.info(f"收到信号 {signum}，开始关闭应用...")
                self.shutdown_event.set()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # 启动应用
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
                server_header=False,
                date_header=False
            )

            server = uvicorn.Server(config)

            logger.info(f"RQA2025应用启动 - http://{host}:{port}")
            logger.info(f"API文档: http://{host}:{port}/docs")
            logger.info(f"交易API: http://{host}:{port}/api/v1/trading/docs")

            await server.serve()

        except Exception as e:
            logger.error(f"应用运行失败: {e}")
            raise
        finally:
            logger.info("RQA2025应用已停止")


async def main():
    """主函数"""
    # 创建应用实例
    app = RQAApplication()

    try:
        # 初始化应用
        await app.initialize()

        # 运行应用
        await app.run()

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        sys.exit(1)


# 创建全局应用实例供uvicorn使用
app = None


def get_app():
    """获取或创建全局FastAPI应用实例"""
    global app
    if app is None:
        # 首先尝试创建完整的应用
        try:
            # 使用同步方式创建，避免asyncio事件循环冲突
            import nest_asyncio
            nest_asyncio.apply()

            async def _create_app():
                application = RQAApplication()
                await application.initialize()
                return application.app

            # 创建新的事件循环来运行异步代码
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                app = loop.run_until_complete(_create_app())
                logger.info("完整RQA应用初始化成功")
            finally:
                loop.close()

        except Exception as e:
            logger.warning(f"完整应用初始化失败，尝试降级模式: {e}")
            # 如果完整应用初始化失败，创建降级版本
            from fastapi import FastAPI

            app = FastAPI(title="RQA2025 - 降级模式")
            error_message = str(e)  # 捕获错误信息供嵌套函数使用

            @app.get("/health")
            async def health():
                return {"status": "degraded", "message": "系统运行在降级模式", "error": error_message}

            @app.get("/")
            async def root():
                return {"message": "RQA2025系统 (降级模式)", "note": "数据库服务不可用"}

            logger.info("降级模式FastAPI应用创建成功")

    return app


# 为uvicorn创建app实例
app = get_app()

if __name__ == "__main__":
    # 运行应用
    asyncio.run(main())
