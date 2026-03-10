"""
RQA2025 Web API
提供REST API接口
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import os
import json
import logging
import random
import asyncio
import aiohttp
import socket
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# 导入pandas用于数据处理
try:
    import pandas as pd
except ImportError:
    pd = None

# 导入拆分后的模块
# RQA2025 API模块

# 导入拆分后的模块
try:
    print("DEBUG: 开始导入config_manager")
    # 注意：load_data_sources 现在由 api.py 自己提供，使用 data_source_config_manager 的缓存机制
    from .config_manager import save_data_sources, DataSourceConfig, get_data_source_config_manager_instance
    print("DEBUG: config_manager导入成功")

    print("DEBUG: 开始导入websocket_manager")
    from .websocket_manager import ConnectionManager
    websocket_manager = ConnectionManager()
    print("DEBUG: websocket_manager导入成功")

    print("DEBUG: 开始导入basic_routes")
    from .basic_routes import router as basic_router
    print("DEBUG: basic_routes导入成功")

    # 修复循环导入问题 - 在应用启动后动态导入
    print("DEBUG: 开始导入strategy_routes")
    from .strategy_routes import router as strategy_router
    print("DEBUG: strategy_routes导入成功")
    
    # 导入策略执行和优化路由
    try:
        from .strategy_execution_routes import router as strategy_execution_router
        print("DEBUG: strategy_execution_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_execution_routes导入失败: {e}")
        strategy_execution_router = None
    
    try:
        from .strategy_lifecycle_routes import router as strategy_lifecycle_router
        print("DEBUG: strategy_lifecycle_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_lifecycle_routes导入失败: {e}")
        strategy_lifecycle_router = None
    
    # 导入策略版本管理路由
    try:
        from .strategy_version_routes import router as strategy_version_router
        print("DEBUG: strategy_version_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_version_routes导入失败: {e}")
        strategy_version_router = None
    
    # 导入策略推荐系统路由
    try:
        from .strategy_recommendation_routes import router as strategy_recommendation_router
        print("DEBUG: strategy_recommendation_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_recommendation_routes导入失败: {e}")
        strategy_recommendation_router = None
    
    # 导入策略性能监控路由
    try:
        from .strategy_performance_routes import router as strategy_performance_router
        print("DEBUG: strategy_performance_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_performance_routes导入失败: {e}")
        strategy_performance_router = None
    
    try:
        from .strategy_optimization_routes import router as strategy_optimization_router
        print("DEBUG: strategy_optimization_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_optimization_routes导入失败: {e}")
        strategy_optimization_router = None
    
    # 导入策略工作流路由
    try:
        from .strategy_workflow_routes import router as strategy_workflow_router
        print("DEBUG: strategy_workflow_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_workflow_routes导入失败: {e}")
        strategy_workflow_router = None
    
    # 导入WebSocket路由
    try:
        from .websocket_routes import router as websocket_router
        print("DEBUG: websocket_routes导入成功")
    except Exception as e:
        print(f"DEBUG: websocket_routes导入失败: {e}")
        websocket_router = None

    print("DEBUG: 开始导入data_collectors")
    from .data_collectors import collect_data_via_data_layer
    print("DEBUG: data_collectors导入成功")

    print("DEBUG: 开始导入api_utils")
    from .api_utils import persist_collected_data, broadcast_data_source_change, generate_data_sample
    print("DEBUG: api_utils导入成功")

    print("DEBUG: 主要模块导入成功")

    # 延迟导入数据源路由，避免循环导入
    datasource_router = None
    
    # 导入数据管理层路由
    try:
        from .data_management_routes import router as data_management_router
        print("DEBUG: data_management_routes导入成功")
    except Exception as e:
        print(f"DEBUG: data_management_routes导入失败: {e}")
        data_management_router = None
    
    # 导入特征工程路由
    try:
        from .feature_engineering_routes import router as feature_engineering_router
        print("DEBUG: feature_engineering_routes导入成功")
    except Exception as e:
        print(f"DEBUG: feature_engineering_routes导入失败: {e}")
        feature_engineering_router = None
    
    # 导入模型训练路由
    try:
        from .model_training_routes import router as model_training_router
        print("DEBUG: model_training_routes导入成功")
    except Exception as e:
        print(f"DEBUG: model_training_routes导入失败: {e}")
        model_training_router = None

    # 导入模型管理路由
    try:
        from .model_management_routes import router as model_management_router
        print("DEBUG: model_management_routes导入成功")
    except Exception as e:
        print(f"DEBUG: model_management_routes导入失败: {e}")
        model_management_router = None
    
    # 导入策略性能评估路由
    try:
        from .strategy_performance_routes import router as strategy_performance_router
        print("DEBUG: strategy_performance_routes导入成功")
    except Exception as e:
        print(f"DEBUG: strategy_performance_routes导入失败: {e}")
        strategy_performance_router = None
    
    # 导入交易信号路由
    try:
        from .trading_signal_routes import router as trading_signal_router
        print("DEBUG: trading_signal_routes导入成功")
    except Exception as e:
        print(f"DEBUG: trading_signal_routes导入失败: {e}")
        trading_signal_router = None
    
    # 导入订单路由路由
    try:
        from .order_routing_routes import router as order_routing_router
        print("DEBUG: order_routing_routes导入成功")
    except Exception as e:
        print(f"DEBUG: order_routing_routes导入失败: {e}")
        order_routing_router = None
    
    # 导入风险报告路由
    try:
        from .risk_reporting_routes import router as risk_reporting_router
        print("DEBUG: risk_reporting_routes导入成功")
    except Exception as e:
        print(f"DEBUG: risk_reporting_routes导入失败: {e}")
        risk_reporting_router = None
    
    # 导入回测路由
    try:
        from .backtest_routes import router as backtest_router
        print("DEBUG: backtest_routes导入成功")
    except Exception as e:
        print(f"DEBUG: backtest_routes导入失败: {e}")
        backtest_router = None
    
    # 导入架构状态监控路由
    try:
        from .architecture_routes import router as architecture_router
        print("DEBUG: architecture_routes导入成功")
    except Exception as e:
        print(f"DEBUG: architecture_routes导入失败: {e}")
        architecture_router = None
    
    # 导入事件监控路由
    try:
        from .events_routes import router as events_router
        print("DEBUG: events_routes导入成功")
    except Exception as e:
        print(f"DEBUG: events_routes导入失败: {e}")
        events_router = None
    
    # 导入策略部署路由
    try:
        from .deployment_routes import router as deployment_router
        print("DEBUG: deployment_routes导入成功")
    except Exception as e:
        print(f"DEBUG: deployment_routes导入失败: {e}")
        deployment_router = None

except Exception as e:
    print(f"DEBUG: 模块导入失败: {e}")
    import traceback
    print("DEBUG: 完整堆栈:")
    traceback.print_exc()
    raise

logger = logging.getLogger(__name__)

# WebSocket管理器已在websocket_manager模块中定义

# 临时简化：避免复杂的导入问题
try:
    from src.data import DataManager
    data_manager = DataManager()
except Exception as e:
    print(f"DataManager import failed: {e}")
    data_manager = None

# DataSourceConfig已在config_manager模块中定义

print("🚀🚀🚀 即将导入数据源路由器 🚀🚀🚀")

# 在应用创建之前尝试注册数据源路由器
# 先导入数据源路由器，完全独立于config_manager
print("🎯🎯🎯 开始导入数据源路由器... 🎯🎯🎯")
datasource_router = None
try:
    print("🎯 执行独立导入 datasource_routes")
    # 直接导入，不依赖其他模块
    import importlib.util
    import sys
    import os

    # 获取当前文件目录
    current_dir = os.path.dirname(__file__)
    datasource_routes_path = os.path.join(current_dir, 'datasource_routes.py')

    # 动态导入datasource_routes模块
    spec = importlib.util.spec_from_file_location("datasource_routes", datasource_routes_path)
    if spec and spec.loader:
        datasource_routes_module = importlib.util.module_from_spec(spec)
        sys.modules["datasource_routes"] = datasource_routes_module
        spec.loader.exec_module(datasource_routes_module)

        datasource_router = datasource_routes_module.router
        print("✅ 数据源路由器动态导入成功")
        print(f"🎯 数据源路由器包含 {len(datasource_router.routes)} 个路由")
    else:
        raise ImportError("无法创建datasource_routes模块规格")

except Exception as e:
    print(f"❌ 数据源路由器导入失败: {e}")
    import traceback
    traceback.print_exc()
    # 只有在确实没有导入成功的情况下才重置为None
    if 'datasource_router' not in locals() or datasource_router is None:
        datasource_router = None

# 环境感知的配置文件路径
def _get_config_file_path():
    """根据环境获取配置文件路径"""
    env = os.getenv("RQA_ENV", "development").lower()

    if env == "production":
        # 生产环境也使用主配置文件，确保配置一致性
        config_file = "data/data_sources_config.json"
    elif env == "testing":
        # 测试环境使用测试目录
        config_file = "data/testing/data_sources_config.json"
    else:
        # 开发环境使用默认目录
        config_file = "data/data_sources_config.json"

    return config_file

DATA_SOURCES_CONFIG_FILE = _get_config_file_path()

def _load_data_sources_from_file() -> List[Dict]:
    """降级方案：直接从JSON文件加载数据源配置"""
    try:
        config_file = _get_config_file_path()
        logger.debug(f"降级方案：直接从文件加载配置: {config_file}")

        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 处理不同格式的配置文件
            if isinstance(config_data, list):
                data_sources = config_data
            elif isinstance(config_data, dict):
                data_sources = config_data.get('data_sources', [])
            else:
                data_sources = []

            logger.debug(f"降级方案：从文件加载了 {len(data_sources)} 个数据源")
            return data_sources
        else:
            logger.warning(f"配置文件不存在: {config_file}")
            return []
    except Exception as e:
        logger.error(f"降级方案加载数据源配置失败: {e}")
        return []


def load_data_sources() -> List[Dict]:
    """加载数据源配置 - 使用 data_source_config_manager 以利用缓存机制"""
    try:
        # 使用 data_source_config_manager 获取数据源，利用其缓存机制避免重复加载
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        config_manager = get_data_source_config_manager()
        data_sources = config_manager.get_data_sources()
        
        logger.debug(f"从 data_source_config_manager 获取了 {len(data_sources)} 个数据源")
        return data_sources
    except Exception as e:
        logger.error(f"从 data_source_config_manager 获取数据源失败: {e}")
        # 降级方案：直接从文件加载
        return _load_data_sources_from_file()


def _get_default_data_sources() -> List[Dict]:
    """获取默认数据源配置（仅用于开发/测试环境）"""
    return [
        {
            "id": "alpha-vantage",
            "name": "Alpha Vantage",
            "type": "股票数据",
            "url": "https://www.alphavantage.co",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "binance",
            "name": "Binance API",
            "type": "加密货币",
            "url": "https://api.binance.com",
            "rate_limit": "10次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "yahoo",
            "name": "Yahoo Finance",
            "type": "市场指数",
            "url": "https://finance.yahoo.com",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "newsapi",
            "name": "NewsAPI",
            "type": "新闻数据",
            "url": "https://newsapi.org",
            "rate_limit": "100次/天",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "miniqmt",
            "name": "MiniQMT",
            "type": "本地交易",
            "url": "http://localhost:8888",
            "rate_limit": "无限制",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "fred",
            "name": "FRED API",
            "type": "宏观经济",
            "url": "https://fred.stlouisfed.org",
            "rate_limit": "无限制",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "coingecko",
            "name": "CoinGecko",
            "type": "加密货币",
            "url": "https://api.coingecko.com",
            "rate_limit": "10-50次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "emweb",
            "name": "东方财富",
            "type": "行情数据",
            "url": "https://emweb.securities.com.cn",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "ths",
            "name": "同花顺",
            "type": "行情数据",
            "url": "https://data.10jqka.com.cn",
            "rate_limit": "5次/分钟",
            "enabled": True,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "xueqiu",
            "name": "雪球",
            "type": "社区数据",
            "url": "https://xueqiu.com",
            "rate_limit": "60次/小时",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "wind",
            "name": "Wind",
            "type": "专业数据",
            "url": "https://www.wind.com.cn",
            "rate_limit": "按协议",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "bloomberg",
            "name": "Bloomberg",
            "type": "专业数据",
            "url": "https://www.bloomberg.com",
            "rate_limit": "按协议",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "qqfinance",
            "name": "腾讯财经",
            "type": "财经新闻",
            "url": "https://finance.qq.com",
            "rate_limit": "10次/分钟",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        },
        {
            "id": "sinafinance",
            "name": "新浪财经",
            "type": "财经新闻",
            "url": "https://finance.sina.com.cn",
            "rate_limit": "10次/分钟",
            "enabled": False,
            "last_test": None,
            "status": "未测试"
        }
    ]

def save_data_sources(sources: List[Dict]):
    """保存数据源配置到文件，带生产环境保护"""
    env = os.getenv("RQA_ENV", "development").lower()

    # 生产环境保护：检查是否正在用默认数据覆盖生产数据
    if env == "production":
        try:
            # 检查现有文件是否存在且不为空
            if os.path.exists(DATA_SOURCES_CONFIG_FILE):
                with open(DATA_SOURCES_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                # 如果现有数据不为空，但新数据看起来像是默认数据，则拒绝保存
                if (len(existing_data) > 0 and len(sources) > 0 and
                    _is_likely_default_data(sources) and not _is_likely_default_data(existing_data)):
                    print("生产环境保护：拒绝用默认数据覆盖现有生产配置")
                    print("如果需要重置配置，请手动删除配置文件后重启")
                    return
        except Exception as e:
            print(f"生产环境数据保护检查失败: {e}")

    try:
        os.makedirs(os.path.dirname(DATA_SOURCES_CONFIG_FILE), exist_ok=True)

        # 创建备份
        if os.path.exists(DATA_SOURCES_CONFIG_FILE):
            backup_file = f"{DATA_SOURCES_CONFIG_FILE}.backup"
            import shutil
            shutil.copy2(DATA_SOURCES_CONFIG_FILE, backup_file)
            print(f"创建配置文件备份: {backup_file}")

        # 保存前再次检查数据
        print(f"保存数据源配置 ({len(sources)} 个数据源):")
        for i, source in enumerate(sources):
            print(f"  {i}: {source.get('name')} - id={repr(source.get('id'))}")

        with open(DATA_SOURCES_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)

        print(f"数据源配置已保存 ({len(sources)} 个数据源)")

    except Exception as e:
        print(f"保存数据源配置失败: {e}")


def _is_likely_default_data(data: List[Dict]) -> bool:
    """检查数据是否看起来像是默认配置数据"""
    if not data or len(data) == 0:
        return False

    # 检查是否所有数据源都是"未测试"状态和None的last_test
    # 这通常表示是默认数据而不是生产使用后的数据
    all_untested = all(
        item.get("status") == "未测试" and item.get("last_test") is None
        for item in data
    )

    return all_untested


def initialize_data_sources_if_needed():
    """安全的数据源初始化，仅在明确需要时执行"""
    env = os.getenv("RQA_ENV", "development").lower()

    # 检查配置文件是否存在
    if os.path.exists(DATA_SOURCES_CONFIG_FILE):
        try:
            with open(DATA_SOURCES_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if len(data) > 0:
                    print(f"数据源配置已存在，包含 {len(data)} 个数据源")
                    return
        except Exception as e:
            print(f"现有配置文件损坏: {e}")

    # 配置文件不存在或为空，需要初始化
    print("检测到数据源配置缺失，开始安全初始化...")

    if env == "production":
        # 生产环境：不自动初始化，避免覆盖
        print("生产环境：跳过自动初始化，请手动配置数据源")
        print("或者从备份恢复配置文件")
        return
    else:
        # 开发/测试环境：初始化默认数据
        print(f"{env}环境：初始化默认数据源配置")
        try:
            default_sources = _get_default_data_sources()
            save_data_sources(default_sources)
            print(f"已初始化 {len(default_sources)} 个默认数据源")
        except Exception as save_error:
            print(f"初始化默认数据源失败: {save_error}")
            # 不抛出异常，继续启动

# 数据采集调度器后台任务（已迁移到核心服务层，符合架构设计）
# 从 src.core.orchestration.business_process.service_scheduler 导入

# 启动验证函数
async def verify_server_ready(host="localhost", port=8000, max_attempts=10):
    """
    验证服务器是否已就绪（通过健康检查端点）
    
    Args:
        host: 服务器主机地址
        port: 服务器端口
        max_attempts: 最大尝试次数
    
    Returns:
        bool: 服务器是否已就绪
    """
    for i in range(max_attempts):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        logger.info("后端服务验证成功，可以接受请求")
                        return True
        except Exception as e:
            if i < max_attempts - 1:
                await asyncio.sleep(0.5)
            else:
                logger.warning(f"后端服务验证超时（尝试{max_attempts}次），但继续启动: {e}")
    return False

# 应用生命周期管理（使用 lifespan 上下文管理器，替代已弃用的 @app.on_event）
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理（统一调度器架构）

    启动顺序：
    1. 后端服务（FastAPI/uvicorn）已启动并可以接受请求
    2. 等待后端服务完全就绪（健康检查）
    3. 启动统一调度器

    架构设计：
    - 网关层只负责API路由和请求处理
    - 统一调度器管理所有后台任务（数据采集、特征工程、模型训练等）
    - 符合分层架构原则：网关层 → 核心服务层 → 统一调度器
    """
    # 使用 print 确保输出（不依赖日志配置）
    print("=" * 80)
    print("🚀 LIFESPAN 函数开始执行 - 统一调度器架构")
    print("=" * 80)

    logger.info("=== 应用生命周期开始（统一调度器架构）===")
    
    try:
        logger.info("后端服务启动事件触发（FastAPI应用已就绪）")
        
        # 确保后端服务完全启动：等待一小段时间，让服务器完全就绪
        await asyncio.sleep(1)
        
        # 验证后端服务可以响应（健康检查）
        try:
            asyncio.create_task(verify_server_ready())
            logger.info("后端服务已就绪")
        except Exception as e:
            logger.warning(f"后端服务健康检查失败（非关键）: {e}")
        
        # 🚀 启动统一调度器（新架构）
        try:
            from src.core.orchestration.scheduler import get_unified_scheduler
            scheduler = get_unified_scheduler()
            
            logger.info("🔧 启动统一调度器...")
            success = await scheduler.start()
            
            if success:
                logger.info("✅ 统一调度器启动成功")
                print("✅ 统一调度器启动成功")
            else:
                logger.warning("⚠️ 统一调度器启动返回失败状态")
        
        except Exception as scheduler_error:
            logger.error(f"❌ 启动统一调度器失败: {scheduler_error}", exc_info=True)
            print(f"❌ 启动统一调度器失败: {scheduler_error}")
        
        # 🚀 注册数据采集器工作节点（确保任务可以被分配执行）
        try:
            from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
            
            registry = get_unified_worker_registry()
            
            # 注册内置数据采集器
            registry.register_worker(
                worker_type=WorkerType.DATA_COLLECTOR,
                worker_id="builtin_data_collector",
                capabilities=["akshare", "baostock", "tushare", "yfinance"],
                metadata={
                    "version": "1.0.0",
                    "type": "builtin",
                    "max_concurrent_tasks": 5,
                    "description": "内置数据采集器"
                }
            )
            
            logger.info("✅ 内置数据采集器工作节点已注册")
            print("✅ 内置数据采集器工作节点已注册")
            
            # 验证注册
            data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
            logger.info(f"当前数据采集器数量: {len(data_collectors)}")
            
        except Exception as worker_error:
            logger.error(f"❌ 注册数据采集器失败: {worker_error}", exc_info=True)
            print(f"❌ 注册数据采集器失败: {worker_error}")
        
        # 🚀 注册数据采集任务处理器
        try:
            from src.core.orchestration.scheduler import get_unified_scheduler
            
            scheduler = get_unified_scheduler()
            worker_manager = scheduler._worker_manager
            
            # 定义数据采集处理器
            def data_collection_handler(payload: dict):
                """数据采集任务处理器 - 调用实际采集器"""
                source_id = payload.get("source_id", "unknown")
                source_config = payload.get("source_config", {})
                
                logger.info(f"🎯 执行数据采集任务: {source_id}")
                
                try:
                    # 根据数据源类型选择采集器
                    if "akshare" in source_id.lower():
                        # 使用 AKShare 采集器
                        from src.data.collectors.akshare_collector import AKShareCollector
                        
                        collector = AKShareCollector()
                        
                        # 从配置中获取股票池 - 支持多种格式（与baostock保持一致）
                        symbols = []
                        
                        # 1. 检查 symbols 字段（列表或逗号分隔字符串）
                        if "symbols" in source_config and source_config["symbols"]:
                            if isinstance(source_config["symbols"], list):
                                symbols = source_config["symbols"]
                            elif isinstance(source_config["symbols"], str):
                                symbols = [s.strip() for s in source_config["symbols"].split(",") if s.strip()]
                        
                        # 2. 检查 custom_stocks 字段
                        elif "custom_stocks" in source_config and source_config["custom_stocks"]:
                            if isinstance(source_config["custom_stocks"], list):
                                for stock in source_config["custom_stocks"]:
                                    if isinstance(stock, dict) and "code" in stock:
                                        symbols.append(stock["code"])
                                    elif isinstance(stock, str):
                                        symbols.append(stock)
                        
                        # 3. 检查 config.custom_stocks
                        elif "config" in source_config and isinstance(source_config["config"], dict):
                            config = source_config["config"]
                            if "custom_stocks" in config and config["custom_stocks"]:
                                if isinstance(config["custom_stocks"], list):
                                    for stock in config["custom_stocks"]:
                                        if isinstance(stock, dict) and "code" in stock:
                                            symbols.append(stock["code"])
                                        elif isinstance(stock, str):
                                            symbols.append(stock)
                        
                        # 如果没有配置股票池，使用默认的
                        if not symbols:
                            logger.warning(f"⚠️ 数据源 {source_id} 未配置股票池，使用默认股票")
                            symbols = ["000001"]  # 默认采集平安银行
                        
                        logger.info(f"📊 数据源 {source_id} 股票池: {symbols}")
                        
                        start_date = source_config.get("start_date")
                        end_date = source_config.get("end_date")
                        
                        total_records = 0
                        for symbol in symbols:
                            # 采集数据
                            data = collector.collect_stock_data(
                                symbol=symbol,
                                start_date=start_date,
                                end_date=end_date
                            )
                            
                            if data:
                                # 保存到数据库
                                success = collector.save_to_database(data, symbol)
                                if success:
                                    total_records += len(data)
                                    logger.info(f"✅ 股票 {symbol} 数据已保存: {len(data)} 条")
                                else:
                                    logger.error(f"❌ 股票 {symbol} 数据保存失败")
                            else:
                                logger.warning(f"⚠️ 股票 {symbol} 未获取到数据")
                        
                        result = {
                            "source_id": source_id,
                            "status": "success",
                            "records_collected": total_records,
                            "symbols_processed": len(symbols),
                            "timestamp": time.time()
                        }
                        
                    elif "baostock" in source_id.lower():
                        # 使用 Baostock 适配器
                        import asyncio
                        from datetime import datetime, timedelta
                        from src.data.adapters.baostock.baostock_adapter import BaostockAdapter, get_baostock_adapter
                        from src.gateway.web.postgresql_persistence import get_db_connection
                        
                        async def baostock_collect():
                            # 准备配置
                            baostock_config = {
                                "username": source_config.get("username", ""),
                                "password": source_config.get("password", ""),
                                "timeout": source_config.get("timeout", 30)
                            }
                            
                            # 获取适配器实例
                            adapter = get_baostock_adapter(baostock_config)
                            
                            # 连接 Baostock
                            connected = await adapter.connect()
                            if not connected:
                                logger.error("❌ 无法连接到 Baostock")
                                return 0
                            
                            try:
                                # 从配置中获取股票池 - 支持多种格式
                                symbols = []
                                
                                # 1. 检查 symbols 字段（列表或逗号分隔字符串）
                                if "symbols" in source_config and source_config["symbols"]:
                                    if isinstance(source_config["symbols"], list):
                                        symbols = source_config["symbols"]
                                    elif isinstance(source_config["symbols"], str):
                                        symbols = [s.strip() for s in source_config["symbols"].split(",") if s.strip()]
                                
                                # 2. 检查 custom_stocks 字段
                                elif "custom_stocks" in source_config and source_config["custom_stocks"]:
                                    if isinstance(source_config["custom_stocks"], list):
                                        for stock in source_config["custom_stocks"]:
                                            if isinstance(stock, dict) and "code" in stock:
                                                symbols.append(stock["code"])
                                            elif isinstance(stock, str):
                                                symbols.append(stock)
                                
                                # 3. 检查 config.custom_stocks
                                elif "config" in source_config and isinstance(source_config["config"], dict):
                                    config = source_config["config"]
                                    if "custom_stocks" in config and config["custom_stocks"]:
                                        if isinstance(config["custom_stocks"], list):
                                            for stock in config["custom_stocks"]:
                                                if isinstance(stock, dict) and "code" in stock:
                                                    symbols.append(stock["code"])
                                                elif isinstance(stock, str):
                                                    symbols.append(stock)
                                
                                # 如果没有配置股票池，使用默认的
                                if not symbols:
                                    logger.warning(f"⚠️ 数据源 {source_id} 未配置股票池，使用默认股票")
                                    symbols = ["sh.600000"]  # 默认采集浦发银行
                                
                                logger.info(f"📊 数据源 {source_id} 原始股票池: {symbols}")
                                
                                # 转换股票代码格式为 baostock 格式（9位）
                                def convert_to_baostock_symbol(symbol: str) -> str:
                                    """将6位股票代码转换为baostock格式（9位）"""
                                    # 如果已经是9位格式，直接返回
                                    if len(symbol) == 9 and symbol.startswith(('sh.', 'sz.', 'bj.')):
                                        return symbol
                                    
                                    # 如果是6位数字代码，根据规则添加前缀
                                    if len(symbol) == 6 and symbol.isdigit():
                                        # 上海股票：60开头、68开头、69开头
                                        if symbol.startswith(('60', '68', '69')):
                                            return f"sh.{symbol}"
                                        # 深圳股票：00开头、30开头
                                        elif symbol.startswith(('00', '30')):
                                            return f"sz.{symbol}"
                                        # 北京股票：8开头、4开头
                                        elif symbol.startswith(('8', '4')):
                                            return f"bj.{symbol}"
                                        # 默认上海
                                        else:
                                            return f"sh.{symbol}"
                                    
                                    # 其他情况，尝试添加sh.前缀
                                    return f"sh.{symbol}" if not symbol.startswith(('sh.', 'sz.', 'bj.')) else symbol
                                
                                # 转换所有股票代码
                                symbols = [convert_to_baostock_symbol(s) for s in symbols]
                                logger.info(f"📊 数据源 {source_id} 转换后股票池: {symbols}")
                                
                                # 设置日期范围
                                end_date = datetime.now()
                                start_date = end_date - timedelta(days=30)
                                
                                total_records = 0
                                
                                for symbol in symbols:
                                    logger.info(f"🎯 采集股票数据: {symbol}")
                                    
                                    # 采集数据
                                    df = await adapter.get_historical_data(
                                        symbol=symbol,
                                        start_date=start_date,
                                        end_date=end_date,
                                        frequency='d',
                                        adjustflag='3'
                                    )
                                    
                                    if df.empty:
                                        logger.warning(f"⚠️ 股票 {symbol} 未获取到数据")
                                        continue
                                    
                                    # 保存到数据库
                                    records_saved = await save_baostock_data_to_db(df, symbol)
                                    total_records += records_saved
                                    
                                    logger.info(f"✅ 股票 {symbol} 数据已保存: {records_saved} 条")
                                
                                return total_records
                                
                            finally:
                                await adapter.disconnect()
                        
                        async def save_baostock_data_to_db(df, symbol):
                            """保存Baostock数据到数据库"""
                            if df.empty:
                                return 0
                            
                            try:
                                conn = get_db_connection()
                                if not conn:
                                    logger.error("❌ 无法获取数据库连接")
                                    return 0
                                
                                cursor = conn.cursor()
                                
                                # 将9位股票代码转换为6位格式（如：sh.600000 -> 600000）
                                if len(symbol) == 9 and symbol.startswith(('sh.', 'sz.', 'bj.')):
                                    symbol_6digit = symbol[3:]  # 去掉前缀
                                else:
                                    symbol_6digit = symbol
                                
                                # 准备插入语句
                                insert_query = """
                                    INSERT INTO akshare_stock_data (
                                        source_id, symbol, date, open_price, high_price, low_price, close_price, 
                                        volume, amount, data_source
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (source_id, symbol, date, data_type) DO UPDATE SET
                                        open_price = EXCLUDED.open_price,
                                        high_price = EXCLUDED.high_price,
                                        low_price = EXCLUDED.low_price,
                                        close_price = EXCLUDED.close_price,
                                        volume = EXCLUDED.volume,
                                        amount = EXCLUDED.amount,
                                        collected_at = CURRENT_TIMESTAMP
                                """
                                
                                records = []
                                for _, row in df.iterrows():
                                    records.append((
                                        'baostock_stock_a',
                                        symbol_6digit,
                                        row.get('date'),
                                        float(row.get('open', 0)) if pd.notna(row.get('open')) else 0,
                                        float(row.get('high', 0)) if pd.notna(row.get('high')) else 0,
                                        float(row.get('low', 0)) if pd.notna(row.get('low')) else 0,
                                        float(row.get('close', 0)) if pd.notna(row.get('close')) else 0,
                                        int(float(row.get('volume', 0))) if pd.notna(row.get('volume')) else 0,
                                        float(row.get('amount', 0)) if pd.notna(row.get('amount')) else 0,
                                        'baostock'
                                    ))
                                
                                if records:
                                    cursor.executemany(insert_query, records)
                                    conn.commit()
                                
                                cursor.close()
                                conn.close()
                                
                                return len(records)
                                
                            except Exception as e:
                                logger.error(f"❌ 保存数据到数据库失败: {e}")
                                return 0
                        
                        # 运行异步采集
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            total_records = loop.run_until_complete(baostock_collect())
                            loop.close()
                            
                            result = {
                                "source_id": source_id,
                                "status": "success",
                                "records_collected": total_records,
                                "timestamp": time.time()
                            }
                        except Exception as e:
                            logger.error(f"❌ Baostock采集失败: {e}")
                            result = {
                                "source_id": source_id,
                                "status": "failed",
                                "error": str(e),
                                "timestamp": time.time()
                            }
                    else:
                        # 未知数据源类型
                        logger.warning(f"⚠️ 未知数据源类型: {source_id}")
                        result = {
                            "source_id": source_id,
                            "status": "skipped",
                            "reason": "unknown_source_type",
                            "timestamp": time.time()
                        }
                    
                    logger.info(f"✅ 数据采集完成: {source_id}, 记录数: {result.get('records_collected', 0)}")
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ 数据采集失败 {source_id}: {e}", exc_info=True)
                    return {
                        "source_id": source_id,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": time.time()
                    }
            
            # 注册处理器（使用JobType.DATA_COLLECTION的值"data_collection"）
            worker_manager.register_task_handler("data_collection", data_collection_handler)
            logger.info("✅ 数据采集任务处理器已注册 (data_collection)")
            print("✅ 数据采集任务处理器已注册 (data_collection)")
            
        except Exception as handler_error:
            logger.error(f"❌ 注册任务处理器失败: {handler_error}", exc_info=True)
            print(f"❌ 注册任务处理器失败: {handler_error}")
        
        # 🚀 启动自动采集服务（根据环境变量控制）
        try:
            import os
            from src.gateway.web.data_collection_scheduler_manager import get_scheduler_manager
            
            # 通过环境变量控制是否自动启动，默认启用
            auto_start = os.getenv("AUTO_COLLECTION_START_ON_BOOT", "true").lower() == "true"
            
            if auto_start:
                scheduler_manager = get_scheduler_manager()
                success = scheduler_manager.start()
                
                if success:
                    logger.info("✅ 自动采集服务已启动")
                    print("✅ 自动采集服务已启动")
                else:
                    logger.warning("⚠️ 自动采集服务启动失败")
                    print("⚠️ 自动采集服务启动失败")
            else:
                logger.info("ℹ️ 自动采集服务未配置为自动启动（设置 AUTO_COLLECTION_START_ON_BOOT=true 启用）")
                print("ℹ️ 自动采集服务未配置为自动启动（设置 AUTO_COLLECTION_START_ON_BOOT=true 启用）")
                
        except Exception as auto_collection_error:
            logger.error(f"❌ 启动自动采集服务失败: {auto_collection_error}", exc_info=True)
            print(f"❌ 启动自动采集服务失败: {auto_collection_error}")
        
        # 发布应用启动完成事件（向后兼容）
        try:
            from src.core.event_bus import get_event_bus
            from src.core.event_bus.types import EventType
            
            event_bus = get_event_bus()
            event_bus.publish(
                EventType.APPLICATION_STARTUP_COMPLETE,
                {
                    "service_name": "api_server",
                    "service_type": "gateway",
                    "timestamp": time.time(),
                    "source": "gateway.web.api",
                    "scheduler": "unified_scheduler"
                },
                source="gateway.web.api"
            )
            logger.info("✅ 应用启动完成事件已发布")
        except Exception as e:
            logger.warning(f"⚠️ 发布应用启动事件失败（非关键）: {e}")

    except Exception as e:
        logger.error(f"❌ 应用启动过程出错: {e}", exc_info=True)
    
    # 应用运行
    logger.info("应用生命周期：进入运行阶段（yield 之后）")
    yield
    logger.info("应用生命周期：退出运行阶段（服务器关闭）")
    
    # 关闭逻辑（统一调度器架构）
    try:
        logger.info("后端服务正在关闭...")
        
        # 🛑 停止统一调度器
        try:
            from src.infrastructure.orchestration.scheduler import get_unified_scheduler
            scheduler = get_unified_scheduler()
            
            logger.info("🛑 停止统一调度器...")
            success = await scheduler.stop()
            
            if success:
                logger.info("✅ 统一调度器已停止")
            else:
                logger.warning("⚠️ 统一调度器停止返回失败状态")
        
        except Exception as scheduler_error:
            logger.error(f"❌ 停止统一调度器失败: {scheduler_error}")
        
        # 发布应用关闭事件（向后兼容）
        try:
            from src.core.event_bus.core import EventBus
            from src.core.event_bus.types import EventType
            
            event_bus = EventBus()
            if hasattr(event_bus, '_initialized') and event_bus._initialized:
                event_bus.publish(
                    EventType.SERVICE_STOPPED,
                    {
                        "service_name": "api_server",
                        "service_type": "gateway",
                        "timestamp": time.time(),
                        "source": "gateway.web.api",
                        "scheduler": "unified_scheduler"
                    },
                    source="gateway.web.api"
                )
                logger.info("✅ 应用关闭事件已发布")
        except Exception as e:
            logger.warning(f"⚠️ 发布应用关闭事件失败（非关键）: {e}")
            
    except Exception as e:
        logger.warning(f"关闭过程出错: {e}")

try:
    app = FastAPI(
        title="RQA2025 量化交易系统",
        description="RQA2025 量化交易系统API",
        version="1.0.0",
        lifespan=lifespan  # 使用 lifespan 管理应用生命周期（FastAPI 0.93+推荐方式）
    )

    # 验证 lifespan 配置（通过检查 lifespan 函数对象）
    logger.info("FastAPI应用已创建，lifespan参数已传递")
    logger.info(f"lifespan函数对象: {lifespan}")

    print("🚨🚨🚨 FastAPI应用已创建 🚨🚨🚨")

except Exception as e:
    logger.error(f"❌ FastAPI应用创建失败: {e}", exc_info=True)
    # 创建基本应用作为降级
    app = FastAPI(
        title="RQA2025 - 降级模式",
        description="RQA2025 量化交易系统（降级模式）",
        version="1.0.0"
    )
    logger.warning("⚠️ 已创建降级模式应用（无lifespan功能）")
    print("🚨🚨🚨 已创建降级模式应用 🚨🚨🚨")

# 立即测试路由
@app.get("/immediate-test")
async def immediate_test():
    """立即测试路由"""
    print("🎯 IMMEDIATE TEST ROUTE CALLED!")
    return {"message": "立即测试路由工作正常", "timestamp": "2026-01-01"}

# 测试数据源路由
@app.post("/api/v1/test-data-sources")
async def test_data_sources():
    """测试数据源路由"""
    print("🎉🎉🎉 测试数据源路由被调用！🎉🎉🎉")
    # 使用本地的 load_data_sources 函数（已统一使用 data_source_config_manager）
    sources = load_data_sources()
    return {"message": "测试数据源路由工作正常", "total": len(sources), "sources": sources}

# 在应用创建后立即添加数据源路由
# 移除重复的GET路由，使用datasource_routes.py中的POST路由

print("🎯🎯🎯 开始注册立即测试路由...")

print("✅ 立即测试路由注册完成")

print("🔧 配置CORS中间件...")

# 在CORS配置之后立即测试
print("🎯 CORS配置之前检查点")
print(f"🎯 当前应用路由数量: {len(app.routes)}")

# 配置CORS
print("🎯 开始配置CORS中间件...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅ CORS配置完成")
print(f"🎯 CORS配置后应用路由数量: {len(app.routes)}")

# 直接添加导出和导入路由
@app.get("/api/v1/data/sources/export")
async def export_data_sources_config():
    """导出数据源配置"""
    try:
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        config_manager = get_data_source_config_manager()
        config_data = config_manager.export_config()
        logger.info(f"配置导出成功，数据源数量: {len(config_data.get('data_sources', []))}")
        return config_data
    except Exception as e:
        logger.error(f"导出配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@app.post("/api/v1/data/sources/import")
async def import_data_sources_config(config_data: dict):
    """导入数据源配置"""
    try:
        from src.gateway.web.data_source_config_manager import get_data_source_config_manager
        config_manager = get_data_source_config_manager()
        success = config_manager.import_config(config_data)
        if success:
            logger.info(f"配置导入成功，数据源数量: {len(config_data.get('data_sources', []))}")
            return {"success": True, "message": "配置导入成功"}
        else:
            logger.error("配置导入失败")
            raise HTTPException(status_code=400, detail="配置导入失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导入配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")

# 注册路由器
print(f"DEBUG: 注册路由器 - basic_router: {len(basic_router.routes)} 路由")
print(f"DEBUG: 注册路由器 - strategy_router: {len(strategy_router.routes)} 路由")

app.include_router(basic_router)
app.include_router(strategy_router)

# 注册策略执行和优化路由器
if strategy_execution_router:
    try:
        app.include_router(strategy_execution_router, tags=["strategy-execution"])
        print("✅ 策略执行路由器注册成功")
    except Exception as e:
        print(f"❌ 策略执行路由器注册失败: {e}")

if strategy_lifecycle_router:
    try:
        app.include_router(strategy_lifecycle_router, tags=["strategy-lifecycle"])
        print("✅ 策略生命周期路由器注册成功")
    except Exception as e:
        print(f"❌ 策略生命周期路由器注册失败: {e}")

# 注册策略版本管理路由器
if strategy_version_router:
    try:
        app.include_router(strategy_version_router, tags=["strategy-version"])
        print("✅ 策略版本管理路由器注册成功")
    except Exception as e:
        print(f"❌ 策略版本管理路由器注册失败: {e}")

# 注册策略推荐系统路由器
if strategy_recommendation_router:
    try:
        app.include_router(strategy_recommendation_router, tags=["strategy-recommendation"])
        print("✅ 策略推荐系统路由器注册成功")
    except Exception as e:
        print(f"❌ 策略推荐系统路由器注册失败: {e}")

# 注册策略性能监控路由器
if strategy_performance_router:
    try:
        app.include_router(strategy_performance_router, tags=["strategy-performance"])
        print("✅ 策略性能监控路由器注册成功")
    except Exception as e:
        print(f"❌ 策略性能监控路由器注册失败: {e}")

if strategy_optimization_router:
    try:
        app.include_router(strategy_optimization_router, tags=["strategy-optimization"])
        print("✅ 策略优化路由器注册成功")
    except Exception as e:
        print(f"❌ 策略优化路由器注册失败: {e}")

# 注册策略工作流路由器
if strategy_workflow_router:
    try:
        app.include_router(strategy_workflow_router, tags=["strategy-workflow"])
        print("✅ 策略工作流路由器注册成功")
    except Exception as e:
        print(f"❌ 策略工作流路由器注册失败: {e}")

# 注册WebSocket路由器
if websocket_router:
    try:
        app.include_router(websocket_router, tags=["websocket"])
        print("✅ WebSocket路由器注册成功")
    except Exception as e:
        print(f"❌ WebSocket路由器注册失败: {e}")

# 注册数据源路由器
if datasource_router:
    try:
        print(f"DEBUG: 注册路由器 - datasource_router: {len(datasource_router.routes)} 路由")
        app.include_router(datasource_router, tags=["data-sources"])
        print("✅ 数据源路由器注册成功")
    except Exception as e:
        print(f"❌ 数据源路由器注册失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ 数据源路由器为None，跳过注册")

# 注册统一调度器路由（新架构）
try:
    from .scheduler_routes import router as scheduler_router
    app.include_router(scheduler_router)
    print(f"✅ 统一调度器路由注册成功（{len(scheduler_router.routes)} 个端点）")
except Exception as e:
    print(f"❌ 统一调度器路由注册失败: {e}")
    import traceback
    traceback.print_exc()

# 注册数据管理层路由器
if data_management_router:
    try:
        app.include_router(data_management_router, prefix="/api/v1", tags=["data-management"])
        print("✅ 数据管理层路由器注册成功")
    except Exception as e:
        print(f"❌ 数据管理层路由器注册失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ 数据管理层路由器为None，跳过注册")

# 注册特征工程路由器
if feature_engineering_router:
    try:
        app.include_router(feature_engineering_router, prefix="/api/v1", tags=["feature-engineering"])
        print("✅ 特征工程路由器注册成功")
    except Exception as e:
        print(f"❌ 特征工程路由器注册失败: {e}")

# 注册模型训练路由器
if model_training_router:
    try:
        app.include_router(model_training_router, prefix="/api/v1", tags=["model-training"])
        print("✅ 模型训练路由器注册成功")
    except Exception as e:
        print(f"❌ 模型训练路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册模型管理路由器
if model_management_router:
    try:
        app.include_router(model_management_router, prefix="/api/v1", tags=["model-management"])
        print("✅ 模型管理路由器注册成功")
    except Exception as e:
        print(f"❌ 模型管理路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册策略性能评估路由器
if strategy_performance_router:
    try:
        app.include_router(strategy_performance_router, prefix="/api/v1", tags=["strategy-performance"])
        print("✅ 策略性能评估路由器注册成功")
    except Exception as e:
        print(f"❌ 策略性能评估路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册交易信号路由器
if trading_signal_router:
    try:
        app.include_router(trading_signal_router, prefix="/api/v1", tags=["trading-signal"])
        
        # 交易执行路由
        from .trading_execution_routes import router as trading_execution_router
        app.include_router(trading_execution_router, tags=["trading-execution"])
        print("✅ 交易信号路由器注册成功")
        
        # 注册信号监控路由器
        try:
            from .signal_monitoring_api import router as signal_monitoring_router
            app.include_router(signal_monitoring_router)
            print("✅ 信号监控路由器注册成功")
        except Exception as e:
            print(f"❌ 信号监控路由器注册失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 注册移动端API路由器
        try:
            from .mobile_api import router as mobile_router
            app.include_router(mobile_router)
            print("✅ 移动端API路由器注册成功")
        except Exception as e:
            print(f"❌ 移动端API路由器注册失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 交易信号路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册订单路由路由器
if order_routing_router:
    try:
        app.include_router(order_routing_router, prefix="/api/v1", tags=["order-routing"])
        print("✅ 订单路由路由器注册成功")
    except Exception as e:
        print(f"❌ 订单路由路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册风险报告路由器
if risk_reporting_router:
    try:
        app.include_router(risk_reporting_router, prefix="/api/v1", tags=["risk-reporting"])
        print("✅ 风险报告路由器注册成功")
    except Exception as e:
        print(f"❌ 风险报告路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册回测路由器
if backtest_router:
    try:
        app.include_router(backtest_router, prefix="/api/v1", tags=["backtest"])
        print("✅ 回测路由器注册成功")
    except Exception as e:
        print(f"❌ 回测路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册架构状态监控路由器
if architecture_router:
    try:
        app.include_router(architecture_router, tags=["architecture"])
        print("✅ 架构状态监控路由器注册成功")
    except Exception as e:
        print(f"❌ 架构状态监控路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册事件监控路由器
if events_router:
    try:
        app.include_router(events_router, tags=["events"])
        print("✅ 事件监控路由器注册成功")
    except Exception as e:
        print(f"❌ 事件监控路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册策略部署路由器
if deployment_router:
    try:
        app.include_router(deployment_router)
        print("✅ 策略部署路由器注册成功")
    except Exception as e:
        print(f"❌ 策略部署路由器注册失败: {e}")
        import traceback
        traceback.print_exc()

# 注册历史数据采集监控路由器
try:
    from src.gateway.api.historical_collection_monitor_api import router as historical_collection_monitor_router
    app.include_router(historical_collection_monitor_router)
    print("✅ 历史数据采集监控路由器注册成功")
except Exception as e:
    print(f"❌ 历史数据采集监控路由器注册失败: {e}")
    import traceback
    traceback.print_exc()

# 注册历史数据采集WebSocket路由器
try:
    from src.gateway.api.historical_collection_websocket import handle_historical_collection_websocket

    @app.websocket("/ws/historical-collection")
    async def historical_collection_websocket_endpoint(websocket: WebSocket):
        await handle_historical_collection_websocket(websocket)

    print("✅ 历史数据采集WebSocket路由器注册成功")
except Exception as e:
    print(f"❌ 历史数据采集WebSocket路由器注册失败: {e}")
    import traceback
    traceback.print_exc()

# 注册健康检查路由器
try:
    from src.infrastructure.health.api.api_endpoints import health_router
    app.include_router(health_router)
    print("✅ 健康检查路由器注册成功")
except Exception as e:
    print(f"❌ 健康检查路由器注册失败: {e}")
    import traceback
    traceback.print_exc()

# 配置静态文件服务
from fastapi.staticfiles import StaticFiles
print("🎯 配置静态文件服务...")

# 确保静态文件目录存在
static_dir = "web-static"
os.makedirs(static_dir, exist_ok=True)

# 挂载静态文件目录到 /static 路径
app.mount("/static", StaticFiles(directory=static_dir), name="static")
print("✅ 静态文件服务配置成功")

# 添加前端页面路由
@app.get("/feature-engineering-monitor.html")
async def serve_feature_engineering_monitor():
    """提供特征工程监控页面"""
    from fastapi.responses import FileResponse
    return FileResponse("web-static/feature-engineering-monitor.html")

@app.get("/dashboard")
async def serve_dashboard():
    """提供仪表盘页面"""
    from fastapi.responses import FileResponse
    return FileResponse("web-static/dashboard.html")

print("✅ 前端页面路由配置成功")

# 移除重复路由 - 使用datasource_routes.py中的路由
print(f"🎯 添加路由后应用路由数量: {len(app.routes)}")

# 执行路由健康检查
try:
    from .route_health_check import print_routes_health_report
    print_routes_health_report(app)
except Exception as e:
    print(f"⚠️ 路由健康检查执行失败: {e}")
    import traceback
    traceback.print_exc()

# 添加直接的测试路由
@app.post("/api/v1/test-akshare/{source_id}")
async def test_akshare_direct(source_id: str):
    """直接测试AKShare路由"""
    print(f"DIRECT: test_akshare_direct called for {source_id}")
    return {
        "source_id": source_id,
        "success": True,
        "status": f"AKShare直接测试成功 - {source_id}",
        "message": f"AKShare数据源 {source_id} 直接测试成功"
    }

@app.post("/api/v1/debug/test")
async def debug_test():
    """调试测试路由"""
    print("DEBUG: debug_test route called")
    return {"message": "Debug test successful", "timestamp": "2026-01-01"}

@app.get("/api/v1/debug/routes")
async def list_app_routes():
    """列出应用级路由"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": getattr(route, 'methods', []),
                "name": getattr(route, 'name', '')
            })
    return {"routes": routes, "total": len(routes)}

@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus指标端点（标准路径）
    
    Returns:
        Prometheus格式的指标数据
    """
    from fastapi.responses import Response
    try:
        # 这里可以集成实际的Prometheus指标收集
        # 目前返回基本指标
        metrics_data = "# HELP health_status Health check status\n# TYPE health_status gauge\nhealth_status 1\n"
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        return Response(
            content=f"# Error: {str(e)}\n",
            media_type="text/plain",
            status_code=500
        )

@app.post("/api/v1/data/sources/{source_id}/test")
async def test_data_source_override(source_id: str):
    """测试数据源连接 (直接在app中定义)"""
    import time
    from datetime import datetime

    print(f"🚀🚀🚀 test_data_source_override ROUTE TRIGGERED for {source_id} 🚀🚀🚀")

    # 获取数据源配置
    print(f"📋 加载数据源配置...")
    sources = load_data_sources()
    print(f"📋 找到 {len(sources)} 个数据源")

    # 检查是否是AKShare数据源
    is_akshare = "akshare" in source_id.lower() or source_id == "akshare_news_wallstreet"
    print(f"🔍 检查AKShare条件: {'akshare' in source_id.lower()} or {source_id == 'akshare_news_wallstreet'} = {is_akshare}")
    source = None
    for s in sources:
        if s.get("id") == source_id:
            source = s
            break

    print(f"🔍 查找数据源 {source_id}: {'找到' if source else '未找到'}")

    if not source:
        return {
            "source_id": source_id,
            "success": False,
            "status": "数据源不存在",
            "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"数据源 {source_id} 不存在",
            "timestamp": time.time()
        }

    # 特殊处理AKShare数据源
    if "akshare" in source_id.lower() or source_id == "akshare_news_wallstreet":
        print(f"DEBUG: 检测到AKShare数据源: {source_id}，开始真实API测试")
        try:
            # 导入AKShare库
            import akshare
            import asyncio

            # 获取AKShare函数名
            config = source.get("config", {})
            akshare_function = config.get("akshare_function", "")

            if not akshare_function:
                return {
                    "source_id": source_id,
                    "success": False,
                    "status": "配置错误",
                    "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"AKShare数据源 {source_id} 缺少 akshare_function 配置",
                    "timestamp": time.time()
                }

            # 检查函数是否存在
            if not hasattr(akshare, akshare_function):
                return {
                    "source_id": source_id,
                    "success": False,
                    "status": "函数不存在",
                    "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"AKShare函数 {akshare_function} 不存在",
                    "timestamp": time.time()
                }

            # 调用AKShare函数进行真实测试
            print(f"DEBUG: 调用AKShare函数: {akshare_function}")
            akshare_func = getattr(akshare, akshare_function)

            # 根据不同函数传递不同参数
            if akshare_function == "news_economic_baidu":
                # 对于新闻函数，尝试获取少量数据进行测试
                data = await asyncio.to_thread(akshare_func, date="20241107")
            else:
                # 其他函数尝试无参数调用
                data = await asyncio.to_thread(akshare_func)

            # 检查数据是否获取成功
            if data is not None and not data.empty and len(data) > 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 更新数据源状态
                source["last_test"] = current_time
                source["status"] = "连接正常"
                save_data_sources(sources)

                print(f"DEBUG: AKShare数据源 {source_id} 测试成功，获取{len(data)}条数据")

                # 自动采集数据样本用于前端显示
                try:
                    from .api_utils import persist_collected_data
                    from .data_collectors import collect_data_via_data_layer

                    # 采集完整数据集
                    collected_data = await collect_data_via_data_layer(source)
                    if collected_data and len(collected_data) > 0:
                        # 持久化数据
                        metadata = {
                            "collection_timestamp": time.time(),
                            "test_collection": True,
                            "data_count": len(collected_data)
                        }
                        persist_result = await persist_collected_data(source_id, collected_data, metadata, source)
                        print(f"DEBUG: 数据样本采集完成: {len(collected_data)}条记录")
                    else:
                        print("DEBUG: 未采集到数据样本")
                except Exception as sample_error:
                    print(f"DEBUG: 数据样本采集失败: {sample_error}")

                return {
                    "source_id": source_id,
                    "success": True,
                    "status": "连接正常",
                    "last_test": current_time,
                    "message": f"AKShare API测试成功 - 获取{len(data)}条数据",
                    "timestamp": time.time(),
                    "data_sample": data.head(3).to_dict('records') if len(data) > 3 else data.to_dict('records')
                }
            else:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 更新数据源状态
                source["last_test"] = current_time
                source["status"] = "数据获取失败"
                save_data_sources(sources)

                return {
                    "source_id": source_id,
                    "success": False,
                    "status": "数据获取失败",
                    "last_test": current_time,
                    "message": "AKShare API调用成功但未获取到数据",
                    "timestamp": time.time()
                }

        except Exception as e:
            print(f"DEBUG: AKShare数据源 {source_id} 测试异常: {e}")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 更新数据源状态
            source["last_test"] = current_time
            source["status"] = f"连接异常: {str(e)[:50]}"
            save_data_sources(sources)

            return {
                "source_id": source_id,
                "success": False,
                "status": f"连接异常: {str(e)[:50]}",
                "last_test": current_time,
                "message": f"AKShare API测试失败: {str(e)}",
                "timestamp": time.time()
            }

    # 其他数据源使用HTTP连接测试
    try:
        import aiohttp
        source_url = source.get("url", "")
        if not source_url:
            return {
                "source_id": source_id,
                "success": False,
                "status": "配置错误",
                "last_test": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"数据源 {source_id} 缺少URL配置",
                "timestamp": time.time()
            }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.head(source_url, allow_redirects=True) as response:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if response.status < 400:
                    # 更新数据源状态
                    source["last_test"] = current_time
                    source["status"] = f"HTTP {response.status} - 连接正常"
                    save_data_sources(sources)

                    return {
                        "source_id": source_id,
                        "success": True,
                        "status": f"HTTP {response.status} - 连接正常",
                        "last_test": current_time,
                        "message": f"连接测试完成：HTTP {response.status} - 连接正常",
                        "timestamp": time.time()
                    }
                else:
                    # 更新数据源状态
                    source["last_test"] = current_time
                    source["status"] = f"HTTP {response.status} - 服务错误"
                    save_data_sources(sources)

                    return {
                        "source_id": source_id,
                        "success": False,
                        "status": f"HTTP {response.status} - 服务错误",
                        "last_test": current_time,
                        "message": f"连接测试失败：HTTP {response.status} - 服务错误",
                        "timestamp": time.time()
                    }

    except Exception as e:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 更新数据源状态
        source["last_test"] = current_time
        source["status"] = f"连接异常: {str(e)[:50]}"
        save_data_sources(sources)

        return {
            "source_id": source_id,
            "success": False,
            "status": f"连接异常: {str(e)[:50]}",
            "last_test": current_time,
            "message": f"连接测试异常: {str(e)}",
            "timestamp": time.time()
        }

# 移到模块级别定义sample路由
@app.get("/api/v1/data/sources/{source_id}/sample")
async def get_data_source_sample(source_id: str):
    """获取数据源的最新数据样本（优先从PostgreSQL获取）"""
    print(f"🎯 sample路由被调用: {source_id}")
    try:
        import pandas as pd
        from pathlib import Path
        
        # 获取数据源配置以确定数据源类型
        sources = load_data_sources()
        source_config = None
        for s in sources:
            if s.get("id") == source_id:
                source_config = s
                break
        
        if not source_config:
            return {
                "error": f"数据源 {source_id} 不存在",
                "source_id": source_id,
                "sample_count": 0
            }
        
        source_type = source_config.get("type", "")
        print(f"🎯 数据源类型: {source_type}")
        
        # 优先从PostgreSQL获取最新数据
        try:
            from .postgresql_persistence import query_latest_data_from_postgresql
            import asyncio
            
            # 使用asyncio.to_thread在后台线程中执行同步查询，避免阻塞
            pg_data = await asyncio.to_thread(
                query_latest_data_from_postgresql,
                source_id=source_id,
                source_type=source_type,
                limit=10
            )
            
            if pg_data and len(pg_data) > 0:
                print(f"✅ 从PostgreSQL获取到 {len(pg_data)} 条最新数据")
                return {
                    "source_id": source_id,
                    "source_type": source_type,
                    "storage_type": "postgresql",
                    "sample_count": len(pg_data),
                    "total_records": len(pg_data),
                    "data": pg_data,
                    "columns": list(pg_data[0].keys()) if pg_data else [],
                    "message": "数据来自PostgreSQL数据库"
                }
            else:
                print(f"⚠️ PostgreSQL中无数据，尝试从文件系统获取")
        except Exception as pg_error:
            logger.warning(f"从PostgreSQL获取数据失败，尝试文件系统: {pg_error}")
            print(f"⚠️ PostgreSQL查询失败: {pg_error}")
        
        # 如果PostgreSQL没有数据，回退到文件系统
        # 查找该数据源的最新样本文件
        samples_dir = Path("data/samples")
        if not samples_dir.exists():
            return {
                "error": "样本目录不存在，且PostgreSQL中无数据",
                "source_id": source_id,
                "sample_count": 0,
                "storage_type": "none"
            }

        # 查找匹配的数据源样本文件
        pattern = f"{source_id}_*.csv"
        sample_files = list(samples_dir.glob(pattern))

        if not sample_files:
            return {
                "error": "未找到数据样本文件，且PostgreSQL中无数据",
                "source_id": source_id,
                "sample_count": 0,
                "storage_type": "none"
            }

        # 选择最新的样本文件
        latest_file = max(sample_files, key=lambda f: f.stat().st_mtime)

        # 读取CSV文件
        try:
            df = pd.read_csv(latest_file, encoding='utf-8')
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试GBK编码
            df = pd.read_csv(latest_file, encoding='gbk')

        # 转换为字典格式
        sample_data = df.head(10).to_dict('records')  # 返回前10条记录

        print(f"✅ 从文件系统获取到 {len(sample_data)} 条数据")
        return {
            "source_id": source_id,
            "source_type": source_type,
            "storage_type": "file",
            "file_name": latest_file.name,
            "sample_count": len(sample_data),
            "total_records": len(df),
            "last_modified": latest_file.stat().st_mtime,
            "data": sample_data,
            "columns": list(df.columns),
            "message": "数据来自文件系统（PostgreSQL中无数据）"
        }

    except Exception as e:
        logger.error(f"获取数据源样本失败 {source_id}: {e}", exc_info=True)
        return {
            "error": f"获取样本失败: {str(e)}",
            "source_id": source_id,
            "sample_count": 0,
            "storage_type": "error"
        }


@app.post("/api/v1/data/sources/{source_id}/collect")
async def collect_data_source(source_id: str):
    """手动采集数据源数据"""
    print(f"🚀🚀🚀 ROUTE CALLED: {source_id} 🚀🚀🚀")
    try:
        print(f"🚀 TRY BLOCK START: {source_id}")
        from .api_utils import persist_collected_data
        from .data_collectors import collect_data_via_data_layer

        # 获取数据源配置
        sources = load_data_sources()
        source = None
        for s in sources:
            if s.get("id") == source_id:
                source = s
                break

        if not source:
            return {
                "success": False,
                "error": "数据源不存在",
                "source_id": source_id
            }

        # 采集数据
        print(f"🎯 开始采集数据源: {source_id}")
        print(f"🎯 数据源配置: {source}")
        result = await collect_data_via_data_layer(source)
        collected_data = result.get("data", [])
        print(f"🎯 采集结果: {len(collected_data)} 条数据")
        print(f"🎯 第一条数据类型: {type(collected_data[0]) if collected_data else 'N/A'}")
        if collected_data:
            print(f"🎯 第一条数据字段: {list(collected_data[0].keys())}")
            print(f"🎯 发布时间字段类型: {type(collected_data[0].get('发布时间', 'N/A'))}")
            print(f"🎯 内容字段类型: {type(collected_data[0].get('内容', 'N/A'))}")

        if collected_data and len(collected_data) > 0:
            # 持久化数据
            print(f"🎯 开始持久化数据: {len(collected_data)} 条")
            metadata = {
                "collection_timestamp": time.time(),
                "manual_collection": True,
                "data_count": len(collected_data)
            }
            persist_result = await persist_collected_data(source_id, collected_data, metadata, source)
            print(f"🎯 持久化结果: {persist_result}")

            return {
                "success": True,
                "source_id": source_id,
                "data_count": len(collected_data),
                "message": f"成功采集 {len(collected_data)} 条数据",
                "timestamp": time.time()
            }
        else:
            return {
                "success": False,
                "source_id": source_id,
                "data_count": 0,
                "message": "未采集到数据"
            }

    except Exception as e:
        logger.error(f"数据采集失败 {source_id}: {e}")
        return {
            "success": False,
            "error": f"采集失败: {str(e)}",
            "source_id": source_id
        }

@app.get("/test-route")
async def test_route():
    """测试路由是否工作"""
    print("🎯🎯🎯 TEST-ROUTE CALLED! 🎯🎯🎯")
    return {"message": "测试路由工作正常", "timestamp": "2026-01-01"}

@app.post("/test-akshare-direct")
async def test_akshare_direct():
    """直接测试AKShare功能"""
    import asyncio
    import akshare
    from datetime import datetime

    try:
        print("🔥🔥🔥 直接测试AKShare news_economic_baidu 🔥🔥🔥")

        # 调用AKShare函数
        data = await asyncio.to_thread(akshare.news_economic_baidu, date="20241107")

        if data is not None and not data.empty and len(data) > 0:
            result = {
                "success": True,
                "status": "AKShare API测试成功",
                "data_count": len(data),
                "message": f"AKShare API调用成功 - 获取{len(data)}条数据",
                "timestamp": datetime.now().isoformat(),
                "sample_data": data.head(3).to_dict('records')
            }
        else:
            result = {
                "success": False,
                "status": "AKShare API无数据",
                "message": "AKShare API调用成功但未获取数据",
                "timestamp": datetime.now().isoformat()
            }

        print(f"🔥 测试结果: {result['success']} - {result['message']}")
        return result

    except Exception as e:
        print(f"🔥 AKShare测试异常: {e}")
        return {
            "success": False,
            "status": "AKShare API异常",
            "message": f"AKShare API测试失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/super-test")
async def super_test():
    """超级测试路由"""
    try:
        print("🔥🔥🔥 SUPER-TEST ROUTE TRIGGERED! 🔥🔥🔥")
        result = {"message": "超级测试路由工作正常", "timestamp": "2026-01-01"}
        print(f"🔥 返回结果: {result}")
        return result
    except Exception as e:
        print(f"🔥 SUPER-TEST 异常: {e}")
        return {"error": str(e), "timestamp": "2026-01-01"}

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RQA2025 量化交易系统",
        "status": "running",
        "version": "1.0.0",
        "services": ["strategy", "trading", "risk", "data"],
        "timestamp": time.time()
    }

# 健康检查路由已由基础设施层注册，避免重复注册

# 确保健康检查端点总是存在（无论路由导入是否成功）
@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "rqa2025-app",
        "environment": os.getenv("RQA_ENV", "unknown"),
        "timestamp": time.time()
    }

@app.get("/ready")
def readiness_check():
    """就绪检查 - 用于Kubernetes readiness probe"""
    return {
        "status": "ready",
        "service": "rqa2025-app",
        "message": "Service is ready to accept requests",
        "timestamp": time.time()
    }

@app.get("/api/v1/status")
async def system_status():
    """系统状态"""
    return {
        "system": "RQA2025",
        "status": "operational",
        "components": {
            "strategy_service": "healthy",
            "trading_service": "healthy",
            "risk_service": "healthy",
            "data_service": "healthy"
        },
        "uptime": time.time(),
        "version": "1.0.0"
    }

@app.get("/api/v1/strategy/status")
async def strategy_status():
    """策略服务状态"""
    return {
        "service": "strategy",
        "status": "healthy",
        "strategies_count": 0,
        "active_strategies": 0,
        "last_update": time.time()
    }

@app.get("/api/v1/trading/status")
async def trading_status():
    """交易服务状态"""
    return {
        "service": "trading",
        "status": "healthy",
        "active_orders": 0,
        "executed_trades": 0,
        "last_update": time.time()
    }

@app.get("/api/v1/risk/status")
async def risk_status():
    """风险控制服务状态"""
    return {
        "service": "risk",
        "status": "healthy",
        "risk_alerts": 0,
        "compliance_checks": 0,
        "last_update": time.time()
    }

@app.get("/api/v1/data/status")
async def data_status():
    """数据服务状态"""
    sources = load_data_sources()
    return {
        "service": "data",
        "status": "healthy",
        "data_sources": len(sources),
        "active_sources": len([s for s in sources if s.get("enabled", True)]),
        "processed_records": 0,
        "last_update": time.time()
    }


@app.get("/api/v1/data/monitoring/report")
async def get_monitoring_report():
    """获取数据采集监控报告"""
    try:
        from src.infrastructure.monitoring.services.data_collection_monitor import get_data_collection_monitor
        monitor = get_data_collection_monitor()
        report = monitor.get_monitoring_report()
        return report
    except Exception as e:
        logger.error(f"获取监控报告失败: {e}")
        return {
            "error": str(e),
            "timestamp": time.time(),
            "status": "error"
        }

# 移除重复的/test路由

# 删除重复的POST路由，使用datasource_routes.py中的实现

@app.options("/api/v1/data/sources")
async def options_data_sources():
    """处理CORS预检请求"""
    return {"message": "CORS preflight OK"}

@app.get("/api/v1/strategy/conceptions")
async def get_strategy_conceptions():
    """获取策略构思列表"""
    conceptions = []
    try:
        for filename in os.listdir(STRATEGY_CONCEPTION_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(STRATEGY_CONCEPTION_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    conception = json.load(f)
                    # 添加统计信息
                    strategy_id = conception.get('id')
                    if strategy_id:
                        conception['stats'] = await get_strategy_stats(strategy_id)
                    conceptions.append(conception)
    except Exception as e:
        logger.error(f"加载策略构思配置失败: {e}")

    return conceptions


async def get_strategy_stats(strategy_id: str) -> Dict[str, int]:
    """获取策略统计信息（回测次数、优化次数）"""
    stats = {
        'backtest_count': 0,
        'optimization_count': 0
    }
    
    try:
        # 统计回测次数
        backtest_dir = os.path.join(DATA_DIR, 'backtest_results')
        if os.path.exists(backtest_dir):
            for filename in os.listdir(backtest_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(backtest_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if data.get('strategy_id') == strategy_id:
                                stats['backtest_count'] += 1
                    except:
                        pass
        
        # 统计优化次数
        optimization_dir = os.path.join(DATA_DIR, 'optimization_results')
        if os.path.exists(optimization_dir):
            for filename in os.listdir(optimization_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(optimization_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if data.get('strategy_id') == strategy_id:
                                stats['optimization_count'] += 1
                    except:
                        pass
    except Exception as e:
        logger.error(f"获取策略统计信息失败: {e}")
    
    return stats


STRATEGY_CONCEPTION_DIR = "data/strategy_conceptions"

def load_strategy_conceptions() -> List[Dict]:
    """从文件加载策略构思配置"""
    conceptions = []
    try:
        if os.path.exists(STRATEGY_CONCEPTION_DIR):
            for filename in os.listdir(STRATEGY_CONCEPTION_DIR):
                if filename.endswith('.json'):
                    filepath = os.path.join(STRATEGY_CONCEPTION_DIR, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        conception = json.load(f)
                        conceptions.append(conception)
    except Exception as e:
        logger.error(f"加载策略构思配置失败: {e}")

    return conceptions

def save_strategy_conception(conception_data: dict):
    """保存策略构思配置"""
    try:
        strategy_id = conception_data.get("id", f"strategy_{int(time.time())}")
        filename = f"{strategy_id}.json"
        filepath = os.path.join(STRATEGY_CONCEPTION_DIR, filename)

        conception_data["updated_at"] = time.time()
        conception_data["version"] = conception_data.get("version", 1) + 1

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conception_data, f, ensure_ascii=False, indent=2)

        return {"success": True, "strategy_id": strategy_id, "filepath": filepath}

    except Exception as e:
        logger.error(f"保存策略构思配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


def validate_strategy_conception(conception_data: dict) -> dict:
    """验证策略构思配置"""
    errors = []
    warnings = []

    # 基本信息验证
    if not conception_data.get("name"):
        errors.append("策略名称不能为空")

    if not conception_data.get("type"):
        errors.append("策略类型不能为空")

    # 节点验证
    nodes = conception_data.get("nodes", [])
    if not nodes:
        errors.append("策略至少需要一个节点")
    else:
        node_types = [node.get("type") for node in nodes]
        required_types = ["data_source", "trade"]

        for required_type in required_types:
            if required_type not in node_types:
                errors.append(f"缺少必需的节点类型: {required_type}")

        # 检查节点连接性
        connections = conception_data.get("connections", [])
        if len(nodes) > 1 and not connections:
            warnings.append("建议为多个节点建立连接关系")

    # 参数验证
    parameters = conception_data.get("parameters", {})
    for param_name, param_config in parameters.items():
        param_type = param_config.get("type")
        param_value = param_config.get("value", param_config.get("default"))

        if param_type == "number":
            min_val = param_config.get("min")
            max_val = param_config.get("max")

            if min_val is not None and param_value < min_val:
                errors.append(f"参数 {param_name} 值 {param_value} 小于最小值 {min_val}")

            if max_val is not None and param_value > max_val:
                errors.append(f"参数 {param_name} 值 {param_value} 大于最大值 {max_val}")

    # 复杂度评分
    complexity_score = len(nodes) * 0.3 + len(connections) * 0.4 + len(parameters) * 0.3
    complexity_level = "低"
    if complexity_score > 3:
        complexity_level = "中"
    if complexity_score > 6:
        complexity_level = "高"

    # 开发时间估算 (天)
    estimated_days = max(1, int(complexity_score * 2))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "complexity_score": round(complexity_score, 1),
        "complexity_level": complexity_level,
        "estimated_days": estimated_days,
        "node_count": len(nodes),
        "connection_count": len(connections),
        "parameter_count": len(parameters)
    }


@app.get("/api/v1/strategy/conception/templates")
async def get_strategy_conception_templates():
    """获取策略构思模板列表"""
    templates = {
        "trend_following": {
            "name": "趋势跟踪策略",
            "description": "基于技术指标识别市场趋势并跟随交易",
            "parameters": {
                "trend_period": {"type": "number", "default": 20, "min": 5, "max": 100, "label": "趋势周期"},
                "entry_threshold": {"type": "number", "default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001, "label": "入场阈值"},
                "exit_threshold": {"type": "number", "default": 0.01, "min": 0.001, "max": 0.05, "step": 0.001, "label": "出场阈值"}
            },
            "required_nodes": ["data_source", "feature", "trade", "risk"],
            "estimated_complexity": "中",
            "estimated_days": 7
        },
        "mean_reversion": {
            "name": "均值回归策略",
            "description": "利用价格偏离均值的回归特性进行交易",
            "parameters": {
                "lookback_period": {"type": "number", "default": 20, "min": 5, "max": 100, "label": "回望周期"},
                "deviation_threshold": {"type": "number", "default": 2.0, "min": 0.5, "max": 5.0, "step": 0.1, "label": "偏离阈值"},
                "holding_period": {"type": "number", "default": 5, "min": 1, "max": 20, "label": "持仓周期"}
            },
            "required_nodes": ["data_source", "feature", "model", "trade", "risk"],
            "estimated_complexity": "中",
            "estimated_days": 8
        },
        "ml_based": {
            "name": "机器学习策略",
            "description": "使用机器学习算法进行价格预测和交易决策",
            "parameters": {
                "model_type": {"type": "select", "options": ["random_forest", "xgboost", "neural_network"], "default": "random_forest", "label": "模型类型"},
                "training_period": {"type": "number", "default": 252, "min": 30, "max": 1000, "label": "训练周期"},
                "prediction_horizon": {"type": "number", "default": 5, "min": 1, "max": 20, "label": "预测周期"},
                "feature_count": {"type": "number", "default": 10, "min": 3, "max": 50, "label": "特征数量"}
            },
            "required_nodes": ["data_source", "feature", "model", "trade", "risk"],
            "estimated_complexity": "高",
            "estimated_days": 14
        },
        "arbitrage": {
            "name": "套利策略",
            "description": "利用不同市场或相关资产间的价差进行套利",
            "parameters": {
                "spread_threshold": {"type": "number", "default": 0.005, "min": 0.001, "max": 0.05, "step": 0.001, "label": "价差阈值"},
                "max_holding_time": {"type": "number", "default": 300, "min": 60, "max": 3600, "label": "最大持仓时间(秒)"},
                "correlation_threshold": {"type": "number", "default": 0.8, "min": 0.5, "max": 0.99, "step": 0.01, "label": "相关性阈值"}
            },
            "required_nodes": ["data_source", "model", "trade", "risk"],
            "estimated_complexity": "高",
            "estimated_days": 12
        }
    }

    return {
        "templates": templates,
        "count": len(templates),
        "timestamp": time.time()
    }


@app.get("/api/v1/strategy/conceptions/{strategy_id}")
async def get_strategy_conception(strategy_id: str):
    """获取指定的策略构思配置"""
    conceptions = load_strategy_conceptions()

    for conception in conceptions:
        if conception.get("id") == strategy_id:
            return conception

    raise HTTPException(status_code=404, detail=f"策略构思 {strategy_id} 不存在")


@app.post("/api/v1/strategy/conceptions")
async def create_strategy_conception(conception_data: dict):
    """创建新的策略构思"""
    try:
        # 添加基本信息
        if not conception_data.get("id"):
            conception_data["id"] = f"strategy_{int(time.time())}"

        conception_data["created_at"] = time.time()
        conception_data["updated_at"] = time.time()
        conception_data["version"] = 1

        # 保存到文件
        result = save_strategy_conception(conception_data)

        return {
            "success": True,
            "message": "策略构思创建成功",
            "strategy_id": result["strategy_id"],
            "data": conception_data,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"创建策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建失败: {str(e)}")


@app.put("/api/v1/strategy/conceptions/{strategy_id}")
async def update_strategy_conception(strategy_id: str, conception_data: dict):
    """更新策略构思配置"""
    try:
        # 确保ID一致
        conception_data["id"] = strategy_id

        # 保存更新
        result = save_strategy_conception(conception_data)

        return {
            "success": True,
            "message": "策略构思更新成功",
            "strategy_id": strategy_id,
            "data": conception_data,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"更新策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")


@app.delete("/api/v1/strategy/conceptions/{strategy_id}")
async def delete_strategy_conception(strategy_id: str):
    """删除策略构思配置"""
    try:
        filepath = os.path.join(STRATEGY_CONCEPTION_DIR, f"{strategy_id}.json")

        if os.path.exists(filepath):
            os.remove(filepath)
            return {
                "success": True,
                "message": f"策略构思 {strategy_id} 已删除",
                "strategy_id": strategy_id,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=404, detail=f"策略构思 {strategy_id} 不存在")

    except Exception as e:
        logger.error(f"删除策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.post("/api/v1/strategy/conceptions/{strategy_id}/validate")
async def validate_strategy_conception_api(strategy_id: str, conception_data: dict = None):
    """验证策略构思配置"""
    try:
        if conception_data is None:
            # 如果没有提供数据，从已保存的配置中加载
            conception_data = await get_strategy_conception(strategy_id)

        validation_result = validate_strategy_conception(conception_data)

        return {
            "strategy_id": strategy_id,
            "validation": validation_result,
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")


@app.post("/api/v1/strategy/conceptions/validate")
async def validate_strategy_conception_new(conception_data: dict):
    """验证新的策略构思配置（未保存的）"""
    try:
        validation_result = validate_strategy_conception(conception_data)

        return {
            "validation": validation_result,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"验证策略构思失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")




# 数据源相关API已在datasource_routes模块中定义

@app.websocket("/ws/data-sources")
async def websocket_data_sources(websocket: WebSocket):
    """数据源监控WebSocket连接"""
    await websocket_manager.connect(websocket, "data_sources")
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.info(f"Received WebSocket message: {message}")

                # 处理客户端消息
                if message.get("type") == "ping":
                    await websocket_manager.send_to_client(websocket, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.get("type") == "subscribe":
                    # 客户端订阅特定数据源
                    source_ids = message.get("source_ids", [])
                    await websocket_manager.send_to_client(websocket, {
                        "type": "subscribed",
                        "source_ids": source_ids,
                        "timestamp": datetime.now().isoformat(),
                        "message": f"已订阅数据源: {', '.join(source_ids)}"
                    })

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from WebSocket: {data}")
                await websocket_manager.send_to_client(websocket, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(websocket)


@app.websocket("/ws/system")
async def websocket_system(websocket: WebSocket):
    """系统监控WebSocket连接"""
    await websocket_manager.connect(websocket, "system")
    try:
        while True:
            # 系统监控WebSocket保持连接
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket_manager.send_to_client(websocket, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"System WebSocket error: {e}")
        await websocket_manager.disconnect(websocket)


# 广播数据源状态变化的辅助函数
async def broadcast_data_source_change(change_type: str, source_id: str, data: Dict[str, Any] = None):
    """广播数据源状态变化"""
    message = {
        "type": change_type,
        "source_id": source_id,
        "timestamp": datetime.now().isoformat(),
        "data": data or {}
    }

    # 根据变化类型设置消息内容
    if change_type == "data_source_created":
        message["message"] = f"数据源 {source_id} 已创建"
    elif change_type == "data_source_updated":
        message["message"] = f"数据源 {source_id} 已更新"
    elif change_type == "data_source_deleted":
        message["message"] = f"数据源 {source_id} 已删除"
    elif change_type == "data_source_tested":
        status = data.get("status", "unknown") if data else "unknown"
        message["message"] = f"数据源 {source_id} 连接测试完成: {status}"
    elif change_type == "data_source_enabled":
        message["message"] = f"数据源 {source_id} 已启用"
    elif change_type == "data_source_disabled":
        message["message"] = f"数据源 {source_id} 已禁用"

    await websocket_manager.broadcast(message, "data_sources")

# 所有路由已通过@app装饰器直接注册，无需额外注册

# 广播系统状态变化的辅助函数
async def broadcast_system_status(status_type: str, data: Dict[str, Any] = None):
    """广播系统状态变化"""
    message = {
        "type": status_type,
        "timestamp": datetime.now().isoformat(),
        "data": data or {}
    }

    if status_type == "system_health":
        health_status = data.get("status", "unknown") if data else "unknown"
        message["message"] = f"系统健康状态: {health_status}"

    await websocket_manager.broadcast(message, "system")

# 移除错误的路由器注册 - 所有路由已通过@app装饰器直接注册
# app.include_router(router)  # 这行代码导致404错误，因为router未定义

# 测试路由
@app.get("/test/debug")
async def test_debug():
    """测试调试路由"""
    return {"message": "Debug route works", "timestamp": time.time()}

# 路由修复检查（暂时禁用）
print('路由修复检查已禁用')
# 强制修复路由
# for i, route in enumerate(app.routes):
#     if hasattr(route, 'path') and route.path == '/api/v1/data/sources/{source_id}' and hasattr(route, 'methods') and 'GET' in route.methods:
#         old_name = route.endpoint.__name__
#         if old_name != 'get_data_source_api':
#             print(f'修复路由 #{i}: {route.path} -> {old_name} 改为 get_data_source_api')
#             route.endpoint = get_data_source_api
#             print('路由修复完成')
#         else:
#             print(f'路由 #{i} 已经正确: {route.path} -> {old_name}')
#         break
# else:
#     print('未找到需要修复的路由 /api/v1/data/sources/{source_id}')

# print('路由修复检查完成')

# 路由已在 datasource_routes 中正确注册，无需手动添加

# 最后的调试信息
print(f"🎯🎯🎯 API模块加载完成 - 总路由数: {len(app.routes)} 🎯🎯🎯")

# 注意：模型训练任务执行器已在 lifespan 函数中启动
# 重新提交持久化训练任务的逻辑也移至 lifespan 函数

async def monitor_periodic_check(monitor):
    """定期执行数据采集监控检查"""
    while True:
        try:
            await asyncio.sleep(60)  # 每60秒检查一次
            monitor.run_monitoring_cycle()
        except Exception as e:
            logger.error(f"数据采集监控周期执行失败: {e}")
            await asyncio.sleep(60)  # 出错后等待60秒再试

# 调试：显示最后5个路由（仅在开发环境）
if __name__ != "__main__":  # 避免在模块导入时执行
    pass  # 可以在这里添加调试代码，如果需要的话

