#!/usr/bin/env python3
"""
RQA2025 量化交易系统主入口
Main entry point for RQA2025 Quantitative Trading System
"""

# 导入必要的模块
import asyncio
import sys
import os
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 模块级别变量
app = None
config_manager = None
logger = None
health_checker = None
cache_manager = None
api_service = None

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from contextlib import asynccontextmanager
    import uvicorn

    # 导入核心服务
    from src.core.core_services.api import APIService
    from src.infrastructure.config import UnifiedConfigManager
    from src.infrastructure.logging import UnifiedLogger
    from src.infrastructure.health import EnhancedHealthChecker
    from src.infrastructure.cache import UnifiedCacheManager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理"""
        global config_manager, logger, health_checker, cache_manager, api_service

        try:
            # 初始化配置管理器
            config_manager = UnifiedConfigManager()
            await config_manager.initialize()

            # 初始化日志系统
            logger = UnifiedLogger()
            await logger.initialize()

            logger.info("正在启动RQA2025量化交易系统...")

            # 初始化健康检查器
            health_checker = EnhancedHealthChecker()
            await health_checker.initialize()

            # 初始化缓存管理器
            cache_manager = UnifiedCacheManager()
            await cache_manager.initialize()

            # 初始化API服务
            api_service = APIService()
            await api_service.initialize()

            # 启动模型训练任务执行器
            try:
                from src.gateway.web.training_job_executor import start_training_job_executor
                await start_training_job_executor()
                logger.info("模型训练任务执行器已启动")
            except Exception as e:
                logger.warning(f"启动模型训练任务执行器失败: {e}")

            # 🚀 启动特征任务执行器（用于执行特征提取任务）
            try:
                from src.gateway.web.feature_task_executor import start_feature_task_executor
                logger.info("🔧 启动特征任务执行器...")
                await start_feature_task_executor()
                logger.info("✅ 特征任务执行器已启动")
            except Exception as e:
                logger.error(f"❌ 启动特征任务执行器失败: {e}")

            # 🎯 启动特征工程系统（符合数据管理层架构设计）
            try:
                from src.gateway.web.feature_engineering_service import initialize_feature_engineering_system
                logger.info("🔧 启动特征工程系统...")
                success = initialize_feature_engineering_system()
                if success:
                    logger.info("✅ 特征工程系统初始化成功")
                else:
                    logger.warning("⚠️ 特征工程系统初始化失败")
            except Exception as e:
                logger.error(f"❌ 启动特征工程系统失败: {e}")

            # 📊 启动数据采集工作器（用于执行数据采集任务）
            try:
                from src.distributed.coordinator.data_collector_worker import get_data_collector_worker
                print("🔧 启动数据采集工作器...")  # 使用print确保日志输出
                logger.info("🔧 启动数据采集工作器...")
                data_collector_worker = get_data_collector_worker()
                data_collector_worker.start()
                print("✅ 数据采集工作器已启动")  # 使用print确保日志输出
                logger.info("✅ 数据采集工作器已启动")
            except Exception as e:
                print(f"❌ 启动数据采集工作器失败: {e}")  # 使用print确保日志输出
                logger.error(f"❌ 启动数据采集工作器失败: {e}")

            # 🎯 启动统一调度器并注册特征提取任务处理器
            try:
                from src.core.orchestration.scheduler import get_unified_scheduler
                
                scheduler = get_unified_scheduler()
                
                # 启动调度器（这会启动工作节点）
                print("🔧 启动统一调度器...")
                logger.info("🔧 启动统一调度器...")
                success = await scheduler.start()
                if success:
                    print("✅ 统一调度器已启动")
                    logger.info("✅ 统一调度器已启动")
                else:
                    print("⚠️ 统一调度器启动失败")
                    logger.warning("⚠️ 统一调度器启动失败")
                
                worker_manager = scheduler._worker_manager
                
                def feature_extraction_handler(payload: dict):
                    """特征提取任务处理器"""
                    try:
                        stock_code = payload.get("stock_code")
                        stock_name = payload.get("stock_name", "")
                        
                        print(f"🚀 开始执行特征提取任务: {stock_code} ({stock_name})")
                        logger.info(f"🚀 开始执行特征提取任务: {stock_code} ({stock_name})")
                        
                        # 调用特征提取引擎
                        from src.features.core.engine import FeatureEngine
                        engine = FeatureEngine()
                        
                        result = engine.extract_features(payload)
                        
                        print(f"✅ 特征提取任务完成: {stock_code}, 提取了 {result.get('feature_count', 0)} 个特征")
                        logger.info(f"✅ 特征提取任务完成: {stock_code}, 提取了 {result.get('feature_count', 0)} 个特征")
                        
                        return {
                            "status": "success",
                            "stock_code": stock_code,
                            "feature_count": result.get("feature_count", 0),
                            "timestamp": time.time()
                        }
                    except Exception as e:
                        print(f"❌ 特征提取任务失败: {payload.get('stock_code')}: {e}")
                        logger.error(f"❌ 特征提取任务失败: {payload.get('stock_code')}: {e}")
                        return {
                            "status": "failed",
                            "stock_code": payload.get("stock_code"),
                            "error": str(e),
                            "timestamp": time.time()
                        }
                
                # 注册处理器
                worker_manager.register_task_handler("feature_extraction", feature_extraction_handler)
                print("✅ 特征提取任务处理器已注册 (feature_extraction)")
                logger.info("✅ 特征提取任务处理器已注册 (feature_extraction)")
                
            except Exception as e:
                print(f"❌ 注册特征提取处理器失败: {e}")
                logger.error(f"❌ 注册特征提取处理器失败: {e}")

            # 🎯 初始化特征工程事件监听器（用于监听数据采集完成事件并自动创建特征提取任务）
            try:
                from src.features.core.event_listeners import initialize_event_listeners
                from src.core.event_bus import get_event_bus
                from src.core.orchestration.scheduler import get_unified_scheduler
                
                event_bus = get_event_bus()
                scheduler = get_unified_scheduler()
                
                print("🔧 初始化特征工程事件监听器...")
                logger.info("🔧 初始化特征工程事件监听器...")
                
                initialize_event_listeners(event_bus, scheduler)
                
                print("✅ 特征工程事件监听器已初始化")
                logger.info("✅ 特征工程事件监听器已初始化")
                
            except Exception as e:
                print(f"⚠️ 初始化特征工程事件监听器失败: {e}")
                logger.warning(f"⚠️ 初始化特征工程事件监听器失败: {e}")

            logger.info("RQA2025量化交易系统启动完成")
            print("✅ RQA2025量化交易系统启动完成")

            yield

        except Exception as e:
            print(f"系统启动失败: {e}")
            sys.exit(1)
        finally:
            # 清理资源
            if logger:
                logger.info("正在关闭RQA2025量化交易系统...")

                try:
                    # 停止特征任务执行器
                    try:
                        from src.gateway.web.feature_task_executor import stop_feature_task_executor
                        await stop_feature_task_executor()
                        logger.info("特征任务执行器已停止")
                    except Exception as e:
                        logger.warning(f"停止特征任务执行器失败: {e}")

                    # 停止模型训练任务执行器
                    try:
                        from src.gateway.web.training_job_executor import stop_training_job_executor
                        await stop_training_job_executor()
                        logger.info("模型训练任务执行器已停止")
                    except Exception as e:
                        logger.warning(f"停止模型训练任务执行器失败: {e}")

                    if api_service:
                        await api_service.shutdown()
                    if cache_manager:
                        await cache_manager.shutdown()
                    if health_checker:
                        await health_checker.shutdown()
                    if logger:
                        await logger.shutdown()

                    logger.info("RQA2025量化交易系统已关闭")

                except Exception as e:
                    print(f"系统关闭时发生错误: {e}")

    # 创建FastAPI应用
    app = FastAPI(
        title="RQA2025 量化交易系统",
        description="基于业务流程驱动的智能化量化交易平台",
        version="4.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 添加WebSocket CORS支持
    @app.middleware("http")
    async def websocket_cors_middleware(request, call_next):
        """为WebSocket连接添加CORS支持"""
        response = await call_next(request)
        
        # 添加CORS头到所有响应
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response

    @app.get("/health")
    async def health_check():
        """健康检查接口"""
        try:
            health_status = await health_checker.check_health()
            return JSONResponse(
                content={
                    "status": "healthy" if health_status else "unhealthy",
                    "timestamp": asyncio.get_event_loop().time(),
                    "service": "RQA2025",
                    "version": "4.0.0"
                },
                status_code=200 if health_status else 503
            )
        except Exception as e:
            return JSONResponse(
                content={
                    "status": "error",
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                },
                status_code=500
            )

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "message": "RQA2025 量化交易系统 API",
            "version": "4.0.0",
            "status": "running",
            "docs": "/docs"
        }

    # 包含API路由
    if api_service:
        app.include_router(api_service.router, prefix="/api/v1")

    def main():
        """主函数"""
        # 获取环境变量
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        # 启动服务器
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"系统错误: {e}")
    sys.exit(1)
