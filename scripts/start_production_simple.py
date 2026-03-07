#!/usr/bin/env python3
"""
RQA2025 生产环境启动脚本 - 简化版本
启动生产环境的量化交易系统服务
"""

import os
import sys
import time
import logging
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/production.log')
    ]
)

logger = logging.getLogger(__name__)

def check_environment():
    """检查运行环境"""
    logger.info("检查生产环境配置...")

    required_env_vars = [
        'RQA_ENV',
        'DATABASE_URL',
        'REDIS_URL'
    ]

    for var in required_env_vars:
        if not os.getenv(var):
            logger.warning(f"环境变量 {var} 未设置")

    # 检查数据库连接
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL', '')
        if db_url:
            logger.info("数据库配置检查完成")
        else:
            logger.warning("DATABASE_URL 未配置")
    except ImportError:
        logger.warning("psycopg2 未安装，跳过数据库检查")

    # 检查Redis连接
    try:
        import redis
        redis_url = os.getenv('REDIS_URL', '')
        if redis_url:
            logger.info("Redis配置检查完成")
        else:
            logger.warning("REDIS_URL 未配置")
    except ImportError:
        logger.warning("redis 未安装，跳过Redis检查")

    logger.info("环境检查完成")

def initialize_services():
    """初始化各项服务"""
    logger.info("初始化RQA2025服务...")

    # 初始化数据源配置
    try:
        logger.info("初始化数据源配置...")
        # 动态导入以避免循环依赖
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        from gateway.web.api import initialize_data_sources_if_needed
        initialize_data_sources_if_needed()
        logger.info("数据源配置初始化完成")
    except Exception as e:
        logger.error(f"数据源配置初始化失败: {e}")
        # 不阻止服务启动，但记录错误

    logger.info("服务初始化完成")

def start_data_collection_scheduler():
    """
    启动数据采集调度器（基于核心服务层业务流程编排）
    
    注意：这是备用实现（线程模式），用于独立启动调度器。
    生产环境推荐使用事件驱动方式启动（通过 app_startup_listener）。
    如果核心服务层调度器已运行，此函数将创建独立的线程调度器。
    """
    import threading
    import time
    import asyncio
    import json
    from src.gateway.web.data_source_config_manager import DataSourceConfigManager

    def data_collection_worker():
        """数据采集工作线程"""
        logger.info("数据采集调度器启动（基于核心服务层编排）")

        # 初始化数据源配置管理器
        try:
            data_source_manager = DataSourceConfigManager()
            logger.info("数据源配置管理器初始化成功")
        except Exception as e:
            logger.error(f"数据源配置管理器初始化失败: {e}")
            return

        # 初始化业务流程编排器（符合核心服务层架构设计）
        orchestrator = None
        event_bus = None
        try:
            # 初始化事件总线（符合核心服务层架构设计：事件驱动通信）
            from src.core.event_bus.core import EventBus
            event_bus = EventBus()
            event_bus.initialize()
            logger.info("事件总线初始化成功")
            
            # 初始化业务流程编排器（符合核心服务层架构设计：业务流程编排）
            from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow
            orchestrator = DataCollectionWorkflow()
            logger.info("数据采集业务流程编排器初始化成功（符合核心服务层架构设计）")
        except Exception as e:
            logger.error(f"初始化业务流程编排器失败: {e}")
            logger.info("将继续使用简化模式运行数据采集")

        # 简单的内存缓存用于跟踪最后采集时间
        last_collection_times = {}
        active_workflows = {}  # 跟踪活跃的工作流

        while True:
            try:
                # 加载数据源配置
                try:
                    sources = data_source_manager.get_data_sources()
                    if not sources:
                        logger.warning("没有找到数据源配置，等待配置...")
                        time.sleep(60)
                        continue
                except Exception as e:
                    logger.error(f"加载数据源配置失败: {e}")
                    time.sleep(60)
                    continue

                current_time = time.time()

                for source in sources:
                    if not source.get('enabled', False):
                        continue

                    source_id = source['id']
                    rate_limit = source.get('rate_limit', '60次/分钟')

                    # 解析频率限制
                    interval_seconds = parse_rate_limit(rate_limit)

                    # 检查是否到了采集时间
                    if source_id in last_collection_times:
                        time_since_last = current_time - last_collection_times[source_id]
                        if time_since_last < interval_seconds:
                            continue

                    # 检查是否有活跃的工作流
                    if source_id in active_workflows:
                        # 检查工作流状态
                        if orchestrator:
                            workflow_status = orchestrator.get_workflow_status(source_id)
                            if workflow_status in ['COMPLETED', 'FAILED', None]:
                                # 工作流已完成或失败，清理
                                del active_workflows[source_id]
                                logger.info(f"数据源 {source_id} 工作流已完成，状态: {workflow_status}")
                            else:
                                # 工作流仍在运行，跳过
                                continue
                        else:
                            # 简化模式下不跟踪工作流状态
                            pass

                    # 执行数据采集（符合核心服务层架构设计：使用业务流程编排器）
                    try:
                        logger.info(f"开始采集数据源: {source_id} (rate_limit: {rate_limit}, interval: {interval_seconds}秒)")

                        if orchestrator:
                            # 使用业务流程编排器（符合核心服务层架构设计）
                            # 通过事件总线发布数据采集开始事件
                            if event_bus:
                                try:
                                    from src.core.event_bus.types import EventType
                                    event_bus.publish(
                                        EventType.DATA_COLLECTION_STARTED,
                                        {
                                            "source_id": source_id,
                                            "source_config": source,
                                            "rate_limit": rate_limit,
                                            "interval_seconds": interval_seconds,
                                            "timestamp": current_time
                                        },
                                        source="data_collection_scheduler"
                                    )
                                except Exception as e:
                                    logger.debug(f"发布数据采集开始事件失败: {e}")
                            
                            # 使用业务流程编排器启动采集流程
                            success = asyncio.run(
                                orchestrator.start_collection_process(source_id, source)
                            )
                            if success:
                                last_collection_times[source_id] = current_time
                                active_workflows[source_id] = current_time
                                logger.info(f"数据源 {source_id} 编排器启动成功")
                                
                                # 发布数据采集完成事件
                                if event_bus:
                                    try:
                                        from src.core.event_bus.types import EventType
                                        event_bus.publish(
                                            EventType.DATA_COLLECTED,
                                            {
                                                "source_id": source_id,
                                                "timestamp": current_time
                                            },
                                            source="data_collection_scheduler"
                                        )
                                    except Exception as e:
                                        logger.debug(f"发布数据采集完成事件失败: {e}")
                            else:
                                logger.warning(f"数据源 {source_id} 编排器启动失败")
                        else:
                            # 降级模式：直接调用网关层API（不符合架构设计，但作为后备方案）
                            try:
                                from src.gateway.web.data_collectors import collect_data_via_data_layer

                                result = asyncio.run(
                                    collect_data_via_data_layer(source, {})
                                )
                                logger.info(f"数据源 {source_id} 采集完成: {result}")

                            except Exception as e:
                                logger.error(f"数据源 {source_id} 采集失败: {e}")

                        # 更新最后采集时间
                        last_collection_times[source_id] = current_time

                    except Exception as e:
                        logger.error(f"数据源 {source_id} 采集异常: {e}")

                # 清理超时的活跃工作流（防止内存泄漏）
                timeout_workflows = []
                for sid, start_time in active_workflows.items():
                    if current_time - start_time > 3600:  # 1小时超时
                        timeout_workflows.append(sid)

                for sid in timeout_workflows:
                    del active_workflows[sid]
                    logger.warning(f"清理超时工作流: {sid}")

                # 等待下一轮检查
                time.sleep(30)  # 每30秒检查一次

            except Exception as e:
                logger.error(f"数据采集调度器异常: {e}")
                time.sleep(60)  # 出错后等待较长时间再试

    # 启动数据采集工作线程
    collection_thread = threading.Thread(target=data_collection_worker, daemon=True)
    collection_thread.start()
    logger.info("数据采集调度器线程已启动")

def start_web_server():
    """启动Web服务器"""
    print("DEBUG: 进入start_web_server函数")
    logger.info("启动RQA2025 Web服务...")

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import time
    import json

    try:
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', '8000'))

        print(f"DEBUG: 主机={host}, 端口={port}")

        # 使用简化版本，但添加必要的API端点
        print("DEBUG: 使用增强的简化FastAPI应用")
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

        @app.get("/")
        async def root():
            return {
                "message": "RQA2025 量化交易系统",
                "status": "running",
                "version": "1.0.0",
                "services": ["strategy", "trading", "risk", "data"],
                "timestamp": time.time()
            }

        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "service": "rqa2025-app",
                "environment": os.getenv("RQA_ENV", "unknown"),
                "container": True,
                "timestamp": time.time()
            }

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok", "message": "Test endpoint working"}

        # 集成网关层API
        try:
            from src.gateway.web.api import setup_api_routes
            setup_api_routes(app)
            logger.info("网关层API路由集成成功")
        except Exception as e:
            logger.error(f"网关层API集成失败: {e}")
            logger.info("继续使用简化API路由")

        # 启动服务器
        logger.info(f"启动Web服务器在 {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")

    except Exception as e:
        logger.error(f"启动Web服务器失败: {e}")
        print(f"DEBUG: Web服务器启动异常: {e}")
        raise

def start_background_services():
    """启动后台服务"""
    logger.info("启动后台服务...")

    try:
        # 启动数据采集调度器
        logger.info("准备启动数据采集调度器...")
        start_data_collection_scheduler()
        logger.info("数据采集调度器启动成功")
    except Exception as e:
        logger.error(f"启动数据采集调度器失败: {e}")
        raise

    logger.info("后台服务启动完成")

def parse_rate_limit(rate_limit_str):
    """
    解析频率限制字符串，返回采集间隔秒数（使用统一函数）
    
    注意：此函数现在使用data_collectors.py中的统一实现，确保所有调度器使用相同的解析逻辑
    """
    try:
        # 优先使用统一的parse_rate_limit函数（符合架构设计：统一实现）
        from src.gateway.web.data_collectors import parse_rate_limit as unified_parse_rate_limit
        return unified_parse_rate_limit(rate_limit_str)
    except ImportError:
        # 降级方案：如果无法导入，使用本地实现
        logger.warning("无法导入统一的parse_rate_limit函数，使用本地实现")
        if not rate_limit_str or rate_limit_str == "无限制":
            return 60.0  # 默认60秒

        if "按协议" in rate_limit_str:
            return 6.0  # 按协议的保守设置：每分钟10次（6秒间隔）

        import re
        # 解析格式如 "5次/分钟", "10次/小时", "1次/天"
        match = re.search(r'(\d+)\s*次\s*/\s*(\w+)', rate_limit_str)
        if match:
            count = int(match.group(1))
            unit = match.group(2)

            if unit in ['分钟', 'minute', 'min']:
                return 60.0 / count if count > 0 else 60.0  # 返回间隔秒数
            elif unit in ['小时', 'hour', 'h']:
                return 3600.0 / count if count > 0 else 3600.0  # 返回间隔秒数
            elif unit in ['天', 'day', 'd']:
                return 86400.0 / count if count > 0 else 86400.0  # 返回间隔秒数
            else:
                return 60.0 / count if count > 0 else 60.0  # 默认按分钟处理
        else:
            # 如果无法解析，返回默认值
            return 60.0

def main():
    """主启动函数"""
    print("DEBUG: main函数开始")
    logger.info("=" * 60)
    logger.info("🚀 RQA2025 量化交易系统生产环境启动")
    logger.info("=" * 60)

    try:
        print("DEBUG: 检查环境")
        # 1. 检查环境
        check_environment()

        print("DEBUG: 初始化服务")
        # 2. 初始化服务
        initialize_services()

        print("DEBUG: 启动Web服务器")
        # 3. 启动Web服务器
        start_web_server()

        print("DEBUG: 启动后台服务")
        # 4. 启动后台服务
        start_background_services()

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"启动过程中发生错误: {e}")
        sys.exit(1)
    finally:
        logger.info("RQA2025服务已停止")

if __name__ == "__main__":
    main()
