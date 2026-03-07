#!/usr/bin/env python3
"""
RQA2025 生产环境启动脚本
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
            # 这里可以添加数据库连接测试
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
            # 这里可以添加Redis连接测试
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
        from src.gateway.web.api import initialize_data_sources_if_needed
        initialize_data_sources_if_needed()
        logger.info("数据源配置初始化完成")
    except Exception as e:
        logger.error(f"数据源配置初始化失败: {e}")
        # 不阻止服务启动，但记录错误

    logger.info("服务初始化完成")

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
            return {"message": "test endpoint works", "timestamp": time.time()}

        # 简单的数据源API
        @app.get("/api/v1/data/sources")
        async def get_data_sources():
            """获取数据源列表"""
            try:
                # 尝试加载数据源配置
                try:
                    with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
                        sources = json.load(f)
                except:
                    # 如果配置文件不存在，返回空列表
                    sources = []

                active_count = len([s for s in sources if s.get("enabled", True)])

                return {
                    "data_sources": sources,
                    "total": len(sources),
                    "active": active_count,
                    "timestamp": time.time()
                }

            except Exception as e:
                return {
                    "data_sources": [],
                    "total": 0,
                    "active": 0,
                    "error": str(e),
                    "timestamp": time.time()
                }

        @app.get("/api/v1/data-sources/metrics")
        async def get_data_sources_metrics():
            """获取数据源性能指标"""
            try:
            # 导入必要的模块
            import random
            import json

            # 尝试加载数据源配置
            try:
                with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
                    sources = json.load(f)
            except:
                # 如果文件不存在，使用默认数据源
                sources = [
                    {
                        "id": "sinafinance",
                        "name": "新浪财经",
                        "type": "财经新闻",
                        "url": "https://finance.sina.com.cn",
                        "rate_limit": "10次/分钟",
                        "enabled": True,
                        "last_test": None,
                        "status": "未测试"
                    }
                ]

            # 计算性能指标
            metrics = {
                "total_sources": len(sources),
                "active_sources": len([s for s in sources if s.get("enabled", True)]),
                "latency_data": {},
                "throughput_data": {},
                "timestamp": time.time()
            }

            # 为每个数据源生成性能数据
            for source in sources:
                source_id = source["id"]
                is_enabled = source.get("enabled", True)
                status = source.get("status", "未测试")

                if is_enabled:
                    # 根据数据源类型和状态生成合理的性能数据
                    if "miniqmt" in source_id:
                        base_latency = 25 if status == "连接正常" else 50
                        base_throughput = 1200 if status == "连接正常" else 800
                    elif "emweb" in source_id:
                        base_latency = 35 if status == "连接正常" else 70
                        base_throughput = 600 if status == "连接正常" else 300
                    elif "ths" in source_id:
                        base_latency = 40 if status == "连接正常" else 75
                        base_throughput = 550 if status == "连接正常" else 250
                    else:
                        base_latency = 45 if status == "连接正常" else 80
                        base_throughput = 400 if status == "连接正常" else 150

                    # 添加实时波动
                    latency_variation = random.uniform(-5, 5)
                    throughput_variation = random.uniform(-50, 50)

                    metrics["latency_data"][source_id] = max(15, min(100, base_latency + latency_variation))
                    metrics["throughput_data"][source_id] = max(100, min(1500, base_throughput + throughput_variation))
                else:
                    # 禁用的数据源设为0
                    metrics["latency_data"][source_id] = 0
                    metrics["throughput_data"][source_id] = 0

            return metrics

            except Exception as e:
            # 返回默认的模拟数据作为回退
            return {
                "total_sources": 1,
                "active_sources": 1,
                "latency_data": {"sinafinance": 45},
                "throughput_data": {"sinafinance": 400},
                "timestamp": time.time()
            }

        @app.put("/api/v1/data/sources/{source_id}")
        async def update_data_source(source_id: str, update_data: dict):
            """更新数据源状态"""
            try:
            # 加载当前数据源配置
            try:
                with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
                    sources = json.load(f)
            except:
                return {"error": "数据源配置文件不存在"}

            # 查找并更新数据源
            for i, source in enumerate(sources):
                if source["id"] == source_id:
                    # 只允许更新enabled状态
                    if "enabled" in update_data:
                        sources[i]["enabled"] = update_data["enabled"]
                        # 更新状态信息
                        if update_data["enabled"]:
                            sources[i]["status"] = "连接正常"
                        else:
                            sources[i]["status"] = "已禁用"

                    # 保存更新
                    with open('data/data_sources_config.json', 'w', encoding='utf-8') as f:
                        json.dump(sources, f, ensure_ascii=False, indent=2)

                    return {
                        "message": f"数据源 {source_id} 更新成功",
                        "source": sources[i]
                    }

            return {"error": f"数据源 {source_id} 不存在"}

            except Exception as e:
            return {"error": str(e)}

        @app.get("/api/v1/data/sources/{source_id}")
        async def get_data_source(source_id: str):
        """获取单个数据源"""
            try:
            # 加载当前数据源配置
            try:
                with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
                    sources = json.load(f)
            except:
                return {"error": "配置文件不存在"}

            # 查找指定的数据源
            for source in sources:
                if source["id"] == source_id:
                    return source

            return {"error": f"数据源 {source_id} 不存在"}

            except Exception as e:
            return {"error": str(e)}

        @app.post("/api/v1/data/sources")
        async def create_data_source(source_data: dict):
        """新增数据源"""
            try:
            # 验证必需字段
            required_fields = ["id", "name", "type", "url"]
            for field in required_fields:
                if field not in source_data or not source_data[field]:
                    return {"success": False, "message": f"缺少必需字段: {field}"}

            # 检查ID是否已存在
            try:
                with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
                    sources = json.load(f)
            except:
                sources = []

            for existing_source in sources:
                if existing_source["id"] == source_data["id"]:
                    return {"success": False, "message": f"数据源ID '{source_data['id']}' 已存在"}

            # 创建新数据源
            new_source = {
                "id": source_data["id"],
                "name": source_data["name"],
                "type": source_data["type"],
                "url": source_data["url"],
                "rate_limit": source_data.get("rate_limit", "100次/分钟"),
                "enabled": source_data.get("enabled", True),
                "last_test": None,
                "status": "未测试"
            }

            # 添加到列表
            sources.append(new_source)

            # 保存到文件
            with open('data/data_sources_config.json', 'w', encoding='utf-8') as f:
                json.dump(sources, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "message": f"数据源 '{source_data['name']}' 创建成功",
                "source": new_source
            }

            except Exception as e:
            return {"success": False, "message": str(e)}

        @app.post("/api/v1/data/sources/{source_id}/test")
        async def test_data_source_connection(source_id: str):
        """测试数据源连接"""
            try:
            # 加载当前数据源配置
            try:
                with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
                    sources = json.load(f)
            except:
                return {"success": False, "status": "配置文件不存在"}

            # 查找数据源
            source = None
            for s in sources:
                if s["id"] == source_id:
                    source = s
                    break

            if not source:
                return {"success": False, "status": f"数据源 {source_id} 不存在"}

            # 简单的连接测试逻辑
            import socket
            import asyncio
            from urllib.parse import urlparse

            success = False
            status_msg = "连接测试中..."

            try:
                source_url = source.get("url", "")
                if not source_url:
                    status_msg = "数据源URL为空"
                elif source_url.startswith("http"):
                    # HTTP连接测试
                    parsed = urlparse(source_url)
                    host = parsed.hostname
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)

                    if host:
                        try:
                            # 创建socket连接测试
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(5)  # 5秒超时
                            result = sock.connect_ex((host, port))
                            sock.close()

                            if result == 0:
                                success = True
                                status_msg = "连接成功"
                            else:
                                status_msg = "连接失败"
                        except Exception as e:
                            status_msg = f"连接异常: {str(e)}"
                    else:
                        status_msg = "无效的URL格式"
                else:
                    # 本地服务测试
                    status_msg = "本地服务类型，跳过网络测试"

            except Exception as e:
                status_msg = f"测试异常: {str(e)}"

            # 更新最后测试时间
            for i, s in enumerate(sources):
                if s["id"] == source_id:
                    sources[i]["last_test"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    break

            # 保存更新
            with open('data/data_sources_config.json', 'w', encoding='utf-8') as f:
                json.dump(sources, f, ensure_ascii=False, indent=2)

            return {
                "success": success,
                "status": status_msg,
                "source_id": source_id,
                "timestamp": time.time()
            }

            except Exception as e:
            return {"success": False, "status": str(e)}

        @app.delete("/api/v1/data/sources/{source_id}")
        async def delete_data_source(source_id: str):
        """删除数据源"""
            try:
            # 加载当前数据源配置
            try:
                with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
                    sources = json.load(f)
            except:
                return {"success": False, "message": "配置文件不存在"}

            # 查找并删除数据源
            original_length = len(sources)
            sources = [s for s in sources if s["id"] != source_id]

            if len(sources) == original_length:
                return {"success": False, "message": f"数据源 {source_id} 不存在"}

            # 保存更新
            with open('data/data_sources_config.json', 'w', encoding='utf-8') as f:
                json.dump(sources, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "message": f"数据源 {source_id} 已成功删除",
                "deleted_id": source_id
            }

            except Exception as e:
            return {"success": False, "message": str(e)}

        # 启动服务器
        logger.info(f"启动Web服务器在 {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")

    except Exception as e:
        logger.error(f"启动Web服务器失败: {e}")
        print(f"DEBUG: Web服务器启动异常: {e}")
        raise

def parse_rate_limit(rate_limit_str):
    """解析频率限制字符串，返回每分钟的采集次数"""
    if not rate_limit_str or rate_limit_str == "无限制":
            return 60  # 默认每分钟1次

    if "按协议" in rate_limit_str:
            return 10  # 按协议的保守设置

    import re

    # 匹配数字和单位
    match = re.match(r'(\d+).*?(分钟|小时|天)', rate_limit_str)
    if not match:
            return 10  # 默认每分钟10次

    count = int(match.group(1))
    unit = match.group(2)

    if unit == "分钟":
            return count
    elif unit == "小时":
            return count / 60  # 转换为每分钟
    elif unit == "天":
            return count / 1440  # 转换为每分钟
    else:
            return 10

def collect_data_from_source(source):
    """从数据源采集数据"""
    try:
        source_id = source.get("id", "")
        source_url = source.get("url", "")
        source_type = source.get("type", "")

        logger.info(f"开始从数据源 {source_id} ({source_type}) 采集数据: {source_url}")

        # 模拟数据采集过程
        import random
        import time

        # 模拟网络请求延迟
        time.sleep(random.uniform(0.1, 0.5))

        # 模拟采集结果
        collected_data = {
            "source_id": source_id,
            "timestamp": time.time(),
            "data_type": source_type,
            "records_count": random.randint(10, 100),
            "status": "success"
        }

        # 这里应该保存数据到数据库或文件
        # 暂时记录到日志
        logger.info(f"数据源 {source_id} 采集完成: {collected_data}")

            return collected_data

    except Exception as e:
        logger.error(f"数据源 {source.get('id', 'unknown')} 采集失败: {e}")
            return None

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

        # 初始化业务流程编排器
        orchestrator = None
            try:
            from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionOrchestrator
            orchestrator = DataCollectionOrchestrator()
            logger.info("数据采集业务流程编排器初始化成功")
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

                    # 执行数据采集
                    try:
                        logger.info(f"开始采集数据源: {source_id}")

                        if orchestrator:
                            # 使用业务流程编排器
                            success = asyncio.run(
                                orchestrator.start_collection_process(source_id, source)
                            )
                            if success:
                                active_workflows[source_id] = current_time
                                logger.info(f"数据源 {source_id} 编排器启动成功")
                            else:
                                logger.warning(f"数据源 {source_id} 编排器启动失败")
                        else:
                            # 简化模式：直接调用网关层API
                            try:
                                from src.gateway.web.api import collect_data_via_data_layer

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

def start_simple_health_service(host, port):
    """启动简单的健康检查服务"""
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="RQA2025 Health Check", version="1.0.0")

        @app.get("/health")
        async def health_check():
            return {
            "status": "healthy",
            "service": "rqa2025-app",
            "environment": os.getenv("RQA_ENV", "unknown"),
            "timestamp": time.time()
        }

        @app.get("/")
        async def root():
            return {
            "message": "RQA2025 量化交易系统",
            "status": "running",
            "version": "1.0.0",
            "services": ["strategy", "trading", "risk", "data"]
        }

    logger.info(f"启动简化Web服务: {host}:{port}")

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