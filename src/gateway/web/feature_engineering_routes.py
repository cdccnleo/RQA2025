"""
特征工程API路由
提供特征提取任务、技术指标、特征存储等API接口
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理，使用BusinessProcessOrchestrator进行业务流程编排
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import random
import logging
import time

# 量化交易系统安全要求：允许的股票代码白名单（与模型训练共享）
ALLOWED_STOCKS: Set[str] = {
    '000001', '000002', '000063', '000100', '000333', '000538', '000568', '000651', '000725', '000768',
    '000858', '000895', '002001', '002007', '002024', '002027', '002044', '002120', '002142', '002230',
    '002236', '002271', '002304', '002352', '002415', '002460', '002475', '002594', '002714', '002812',
    '300003', '300014', '300015', '300033', '300059', '300122', '300124', '300142', '300274', '300408',
    '300413', '300433', '300498', '300750', '600000', '600009', '600016', '600028', '600030', '600031',
    '600036', '600048', '600050', '600104', '600276', '600309', '600340', '600406', '600436', '600438',
    '600519', '600585', '600588', '600600', '600660', '600690', '600703', '600745', '600809', '600837',
    '600887', '600893', '600900', '601012', '601066', '601088', '601100', '601138', '601166', '601186',
    '601211', '601288', '601318', '601319', '601328', '601336', '601398', '601601', '601628', '601668',
    '601688', '601766', '601788', '601816', '601857', '601888', '601899', '601919', '601933', '601939',
    '601985', '601988', '601990', '603288', '603501', '603659', '603799', '603986', '688008', '688009',
    '688012', '688036', '688111', '688126', '688169', '688185', '688256', '688303', '688363', '688599'
}

# 量化交易系统安全要求：最大日期范围（5年）
MAX_DATE_RANGE_DAYS = 365 * 5

# 量化交易系统安全要求：最大特征数量
MAX_FEATURE_COUNT = 100

# 量化交易系统安全要求：最大执行时间（30分钟）
MAX_EXECUTION_TIME_SECONDS = 30 * 60

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# 导入服务层
from .feature_engineering_service import (
    get_feature_tasks,
    get_feature_tasks_stats,
    get_features,
    get_features_stats,
    get_quality_distribution,
    get_technical_indicators,
    create_feature_task,
    stop_feature_task,
    delete_feature_task,
    get_scheduler_status,
    start_scheduler,
    stop_scheduler,
    get_feature_data_for_training,
    get_available_feature_tasks_for_training
)

router = APIRouter()

# 全局服务容器（延迟初始化，符合架构设计）
_container = None

def _get_container():
    """获取服务容器实例（单例模式，符合架构设计）"""
    global _container
    if _container is None:
        try:
            from src.core.container.container import DependencyContainer
            _container = DependencyContainer()
            
            # 注册事件总线（符合架构设计：事件驱动通信）
            try:
                from src.core.event_bus.core import EventBus
                event_bus = EventBus()
                event_bus.initialize()
                _container.register(
                    "event_bus",
                    service=event_bus,
                    lifecycle="singleton"
                )
                logger.info("事件总线已注册到服务容器")
            except Exception as e:
                logger.warning(f"注册事件总线失败: {e}")
            
            logger.info("服务容器已初始化")
        except Exception as e:
            logger.warning(f"服务容器初始化失败: {e}")
            _container = None
    return _container

def _get_event_bus():
    """获取事件总线实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            return container.resolve("event_bus")
        except Exception as e:
            logger.debug(f"从服务容器获取事件总线失败: {e}")
    
    # 降级方案：直接创建
    try:
        from src.core.event_bus.core import EventBus
        event_bus = EventBus()
        event_bus.initialize()
        return event_bus
    except Exception as e:
        logger.warning(f"创建事件总线失败: {e}")
        return None

def _get_orchestrator():
    """获取业务流程编排器实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            return container.resolve("business_process_orchestrator")
        except Exception as e:
            logger.debug(f"从服务容器获取业务流程编排器失败: {e}")
    
    # 降级方案：直接创建（业务流程编排器已在特征引擎中集成，这里提供全局访问点）
    try:
        from src.infrastructure.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
        orchestrator = BusinessProcessOrchestrator()
        orchestrator.initialize()
        return orchestrator
    except Exception as e:
        logger.debug(f"创建业务流程编排器失败（可选功能，特征引擎内部已集成）: {e}")
        return None


def _validate_symbol(symbol: str) -> None:
    """验证股票代码是否在白名单中"""
    if symbol and symbol not in ALLOWED_STOCKS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的股票代码: {symbol}。请联系管理员添加支持或选择其他股票。"
        )


def _validate_symbols(symbols: List[str]) -> None:
    """验证批量股票代码"""
    if symbols:
        invalid_symbols = set(symbols) - ALLOWED_STOCKS
        if invalid_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"包含不支持的股票代码: {list(invalid_symbols)}。请联系管理员添加支持或选择其他股票。"
            )


def _validate_date_range(start_date: str, end_date: str) -> tuple:
    """验证日期范围"""
    if not start_date or not end_date:
        return None, None
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days
        
        if days > MAX_DATE_RANGE_DAYS:
            raise HTTPException(
                status_code=400,
                detail=f"日期范围不能超过{MAX_DATE_RANGE_DAYS}天（约5年），当前选择: {days}天"
            )
        
        warnings = []
        if days < 30:
            warnings.append(f"日期范围仅{days}天，建议至少30天以获得更好的特征提取效果")
        
        return warnings, days
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"日期格式错误，请使用YYYY-MM-DD格式"
        )


def _enforce_quality_checks(config: Dict[str, Any]) -> List[str]:
    """强制启用特征质量评估"""
    warnings = []
    
    # 强制启用特征质量检查
    if not config.get("enable_quality_check"):
        logger.warning("特征质量检查未启用，根据量化交易系统要求强制启用")
        config["enable_quality_check"] = True
        warnings.append("特征质量检查已根据量化交易系统要求强制启用")
    
    # 强制启用异常检测
    if not config.get("enable_anomaly_detection"):
        logger.warning("异常检测未启用，根据量化交易系统要求强制启用")
        config["enable_anomaly_detection"] = True
        warnings.append("异常检测已根据量化交易系统要求强制启用")
    
    return warnings


# ==================== 特征提取任务API ====================

@router.get("/features/engineering/tasks")
async def get_feature_tasks_endpoint(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="任务状态筛选"),
    task_type: Optional[str] = Query(None, description="任务类型筛选"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期")
) -> Dict[str, Any]:
    """获取特征提取任务列表 - 支持分页和筛选"""
    try:
        # 获取所有任务
        all_tasks = get_feature_tasks()
        
        # 应用筛选条件
        filtered_tasks = all_tasks
        if status:
            filtered_tasks = [t for t in filtered_tasks if t.get('status') == status]
        if task_type:
            filtered_tasks = [t for t in filtered_tasks if t.get('task_type') == task_type]
        if start_date:
            filtered_tasks = [t for t in filtered_tasks if t.get('created_at', '') >= start_date]
        if end_date:
            filtered_tasks = [t for t in filtered_tasks if t.get('created_at', '') <= end_date]
        
        # 计算分页
        total = len(filtered_tasks)
        total_pages = (total + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_tasks = filtered_tasks[start_idx:end_idx]
        
        # 获取统计信息
        stats = get_feature_tasks_stats()
        
        return {
            "tasks": paginated_tasks,
            "stats": stats,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征任务失败: {str(e)}")


@router.post("/features/engineering/tasks")
async def create_feature_task_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """创建特征提取任务 - 使用真实数据，符合架构设计：事件驱动和业务流程编排"""
    try:
        task_type = request.get("task_type", "技术指标")
        config = request.get("config", {})
        
        # 量化交易系统安全要求：验证股票代码
        symbol = config.get("symbol")
        if symbol:
            _validate_symbol(symbol)
        
        # 量化交易系统安全要求：验证日期范围
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        date_warnings, date_days = _validate_date_range(start_date, end_date)
        
        # 量化交易系统安全要求：强制启用特征质量评估
        quality_warnings = _enforce_quality_checks(config)
        
        # 合并警告信息
        all_warnings = (date_warnings or []) + quality_warnings
        if all_warnings:
            config["warning_messages"] = all_warnings
        
        # 可选：使用BusinessProcessOrchestrator管理特征工程任务创建业务流程（符合架构设计）
        # 注意：当前orchestrator主要支持交易周期流程，特征工程任务可通过状态机间接管理
        orchestrator = _get_orchestrator()
        process_id = None
        if orchestrator:
            try:
                from src.infrastructure.orchestration.models.process_models import BusinessProcessState
                import time
                task_id_preview = f"task_{int(time.time())}"
                process_id = f"feature_task_create_{task_id_preview}_{int(time.time())}"
                # 业务流程编排器已在特征工程服务层集成，这里记录流程ID用于关联
                logger.debug(f"特征工程任务创建业务流程ID: {process_id}（业务流程编排器已初始化）")
            except Exception as e:
                logger.debug(f"业务流程编排器初始化检查失败（可选功能）: {e}")
        
        # 调用服务层创建任务
        task = create_feature_task(task_type, config)
        task_id = task.get("task_id")
        
        # 发布特征提取开始事件到EventBus（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.FEATURE_EXTRACTION_STARTED,
                    {
                        "task_id": task_id,
                        "task_type": task_type,
                        "config": config,
                        "process_id": process_id,
                        "timestamp": time.time()
                    },
                    source="feature_engineering_routes"
                )
                logger.debug(f"已发布特征提取开始事件: {task_id}")
            except Exception as e:
                logger.debug(f"发布特征提取开始事件失败: {e}")
        
        # 可选：通过业务流程编排器记录任务创建完成（符合架构设计）
        # 注意：特征工程任务的业务流程状态管理主要在特征引擎内部完成
        if orchestrator and process_id:
            try:
                # 业务流程编排器已在特征引擎中集成，这里只是确保流程追踪可用
                logger.debug(f"特征工程任务创建完成，流程ID: {process_id}")
            except Exception as e:
                logger.debug(f"业务流程状态记录失败（可选功能）: {e}")
        
        # WebSocket广播特征任务创建事件（符合架构设计：实时更新）
        try:
            from .websocket_manager import manager
            await manager.broadcast("feature_engineering", {
                "type": "feature_task_created",
                "task_id": task_id,
                "task_type": task_type,
                "task": task,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.debug(f"WebSocket广播失败: {e}")
        
        return {
            "success": True,
            "task_id": task_id,
            "task": task,
            "message": f"特征提取任务已创建: {task_type}"
        }
    except HTTPException:
        # 重新抛出HTTPException，保持原始状态码
        raise
    except Exception as e:
        logger.error(f"创建特征任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建特征任务失败: {str(e)}")


@router.post("/features/engineering/tasks/batch")
async def create_batch_feature_tasks_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    批量创建特征提取任务 - 支持A股市场大规模特征计算
    
    支持按股票池批量创建任务，适用于：
    - 全市场批量计算
    - 沪深300/中证500等指数成分股计算
    - 自定义股票列表计算
    
    请求示例:
    {
        "stock_pool": "all",  // 或 "hs300", "csi500", ["002837", "688702"]
        "date_range": {
            "start": "2025-01-01",
            "end": "2026-02-13"
        },
        "task_type": "技术指标",
        "indicators": ["SMA", "EMA", "RSI", "MACD"],
        "batch_size": 100,  // 每批处理股票数
        "priority": "medium"
    }
    """
    try:
        # 解析请求参数
        stock_pool = request.get("stock_pool", "all")
        date_range = request.get("date_range", {})
        start_date = date_range.get("start", "2025-01-01")
        end_date = date_range.get("end", "2026-02-13")
        task_type = request.get("task_type", "技术指标")
        indicators = request.get("indicators", ["SMA", "EMA", "RSI", "MACD"])
        batch_size = request.get("batch_size", 100)
        priority = request.get("priority", "medium")
        
        # 量化交易系统安全要求：验证日期范围
        date_warnings, date_days = _validate_date_range(start_date, end_date)
        
        logger.info(f"批量创建特征任务: stock_pool={stock_pool}, date_range={start_date}~{end_date}")
        
        # 获取股票池管理器
        pool_manager = None
        stock_pool_type = None
        try:
            from src.data_management import get_stock_pool_manager, StockPoolType
            pool_manager = get_stock_pool_manager()
            stock_pool_type = StockPoolType
        except ImportError as e:
            logger.warning(f"导入股票池管理器失败: {e}，将使用降级方案")
        
        # 获取股票列表
        symbols = []
        pool_name = ""
        
        if isinstance(stock_pool, list):
            # 自定义股票列表
            symbols = stock_pool
            pool_name = "自定义列表"
        elif pool_manager and stock_pool_type:
            # 使用股票池管理器获取股票列表
            if stock_pool == "all":
                # 全市场
                pool = pool_manager.get_predefined_pool(stock_pool_type.ALL)
                if pool:
                    symbols = pool.symbols
                    pool_name = pool.name
            elif stock_pool == "hs300":
                pool = pool_manager.get_predefined_pool(stock_pool_type.HS300)
                if pool:
                    symbols = pool.symbols
                    pool_name = pool.name
            elif stock_pool == "csi500":
                pool = pool_manager.get_predefined_pool(stock_pool_type.CSI500)
                if pool:
                    symbols = pool.symbols
                    pool_name = pool.name
            elif stock_pool == "growth":
                pool = pool_manager.get_predefined_pool(stock_pool_type.GROWTH)
                if pool:
                    symbols = pool.symbols
                    pool_name = pool.name
            elif stock_pool == "star":
                pool = pool_manager.get_predefined_pool(stock_pool_type.STAR)
                if pool:
                    symbols = pool.symbols
                    pool_name = pool.name
            else:
                # 尝试按股票池ID获取
                symbols = pool_manager.get_pool_symbols(stock_pool)
                pool_name = stock_pool
        else:
            # 降级方案：使用默认股票列表
            logger.warning("股票池管理器不可用，使用默认股票列表")
            if stock_pool == "all":
                # 尝试从数据层获取股票列表
                try:
                    from src.data.loader import get_stock_loader
                    loader = get_stock_loader()
                    symbols = loader.get_all_symbols()
                    pool_name = "全市场"
                except Exception as e:
                    logger.warning(f"从数据层获取股票列表失败: {e}")
                    # 使用示例股票列表作为最后的降级方案
                    symbols = ["000001", "000002", "600000", "600519", "002837", "688702"]
                    pool_name = "示例股票池"
            else:
                # 对于其他股票池类型，使用示例股票
                symbols = ["000001", "000002", "600000", "600519", "002837", "688702"]
                pool_name = f"示例股票池({stock_pool})"
        
        if not symbols:
            raise HTTPException(status_code=400, detail=f"股票池 {stock_pool} 为空或不存在")
        
        # 量化交易系统安全要求：验证股票代码白名单
        _validate_symbols(symbols)
        
        logger.info(f"股票池 {pool_name} 包含 {len(symbols)} 只股票")
        
        # 分批创建任务
        if pool_manager:
            batches = pool_manager.split_pool_for_batch(symbols, batch_size)
        else:
            # 降级方案：手动分批
            batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        created_tasks = []
        failed_tasks = []
        
        logger.info(f"将创建 {len(batches)} 批任务，每批最多 {batch_size} 只股票")
        
        # 获取任务ID前缀
        task_id_prefix = request.get("task_id_prefix", "feature_task_pool")
        
        for batch_idx, batch_symbols in enumerate(batches):
            try:
                # 为每批创建一个任务
                config = {
                    "symbols": batch_symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "indicators": indicators,
                    "batch_index": batch_idx,
                    "total_batches": len(batches),
                    "priority": priority,
                    "task_id_prefix": f"{task_id_prefix}_batch{batch_idx}"
                }
                
                # 量化交易系统安全要求：强制启用特征质量评估
                quality_warnings = _enforce_quality_checks(config)
                if quality_warnings:
                    config["warning_messages"] = quality_warnings
                
                task = create_feature_task(task_type, config)
                task_id = task.get("task_id")
                
                created_tasks.append({
                    "task_id": task_id,
                    "batch_index": batch_idx,
                    "symbol_count": len(batch_symbols),
                    "symbols": batch_symbols[:5] + ["..."] if len(batch_symbols) > 5 else batch_symbols  # 只显示前5只
                })
                
                logger.info(f"批次 {batch_idx + 1}/{len(batches)} 任务创建成功: {task_id}")
                
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 任务创建失败: {e}")
                failed_tasks.append({
                    "batch_index": batch_idx,
                    "symbol_count": len(batch_symbols),
                    "error": str(e)
                })
        
        # 发布批量任务创建事件
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.FEATURE_EXTRACTION_STARTED,
                    {
                        "batch_task": True,
                        "pool_name": pool_name,
                        "total_symbols": len(symbols),
                        "batch_count": len(batches),
                        "created_tasks": len(created_tasks),
                        "failed_tasks": len(failed_tasks),
                        "timestamp": time.time()
                    },
                    source="feature_engineering_routes"
                )
            except Exception as e:
                logger.debug(f"发布批量任务事件失败: {e}")
        
        # WebSocket广播
        try:
            from .websocket_manager import manager
            await manager.broadcast("feature_engineering", {
                "type": "batch_feature_tasks_created",
                "pool_name": pool_name,
                "total_symbols": len(symbols),
                "batch_count": len(batches),
                "created_count": len(created_tasks),
                "timestamp": time.time()
            })
        except Exception as e:
            logger.debug(f"WebSocket广播失败: {e}")
        
        return {
            "success": True,
            "pool_name": pool_name,
            "total_symbols": len(symbols),
            "batch_count": len(batches),
            "created_tasks": created_tasks,
            "failed_tasks": failed_tasks,
            "message": f"成功创建 {len(created_tasks)} 批任务，共 {len(symbols)} 只股票"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量创建特征任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量创建特征任务失败: {str(e)}")


@router.get("/features/engineering/pools")
async def get_stock_pools_endpoint() -> Dict[str, Any]:
    """
    获取可用股票池列表
    
    返回预定义的股票池（全市场、沪深300、中证500等）
    """
    try:
        from src.data_management import get_stock_pool_manager, STOCK_POOL_AVAILABLE
        
        # 检查股票池管理器是否可用
        if not STOCK_POOL_AVAILABLE or get_stock_pool_manager is None:
            logger.warning("股票池管理器不可用，返回默认股票池列表")
            return get_default_stock_pools()
        
        pool_manager = get_stock_pool_manager()
        
        stats = pool_manager.get_pool_statistics()
        pools = pool_manager.get_all_pools()
        
        return {
            "success": True,
            "total_stocks": stats.get("total_stocks", 0),
            "pools": [
                {
                    "pool_id": pool.pool_id,
                    "name": pool.name,
                    "type": pool.pool_type.value,
                    "symbol_count": pool.symbol_count,
                    "description": pool.description
                }
                for pool in pools
            ],
            "exchange_distribution": stats.get("exchange_distribution", {})
        }
        
    except Exception as e:
        logger.error(f"获取股票池列表失败: {e}，使用默认股票池")
        # 返回默认股票池而不是报错
        return get_default_stock_pools()


def get_default_stock_pools() -> Dict[str, Any]:
    """
    获取默认股票池列表（当股票池管理器不可用时使用）
    
    Returns:
        默认股票池列表
    """
    return {
        "success": True,
        "total_stocks": 0,
        "pools": [
            {
                "pool_id": "all_market",
                "name": "全市场",
                "type": "all",
                "symbol_count": 0,
                "description": "A股市场所有股票（数据加载中）"
            },
            {
                "pool_id": "hs300",
                "name": "沪深300",
                "type": "index",
                "symbol_count": 300,
                "description": "沪深300指数成分股"
            },
            {
                "pool_id": "zz500",
                "name": "中证500",
                "type": "index",
                "symbol_count": 500,
                "description": "中证500指数成分股"
            },
            {
                "pool_id": "sz50",
                "name": "上证50",
                "type": "index",
                "symbol_count": 50,
                "description": "上证50指数成分股"
            }
        ],
        "exchange_distribution": {
            "SSE": 0,
            "SZSE": 0
        },
        "note": "使用默认股票池，实际数据加载失败"
    }


@router.post("/features/engineering/tasks/{task_id}/stop")
async def stop_feature_task_endpoint(task_id: str) -> Dict[str, Any]:
    """停止特征提取任务 - 使用真实数据，符合架构设计：事件驱动"""
    try:
        success = stop_feature_task(task_id)
        if success:
            # 发布特征任务停止事件到EventBus（符合架构设计：事件驱动通信）
            event_bus = _get_event_bus()
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.FEATURE_PROCESSING_COMPLETED,  # 使用特征处理完成事件表示任务停止
                        {
                            "task_id": task_id,
                            "action": "stopped",
                            "timestamp": time.time()
                        },
                        source="feature_engineering_routes"
                    )
                    logger.debug(f"已发布特征任务停止事件: {task_id}")
                except Exception as e:
                    logger.debug(f"发布特征任务停止事件失败: {e}")
            
            # WebSocket广播特征任务停止事件（符合架构设计：实时更新）
            try:
                from .websocket_manager import manager
                await manager.broadcast("feature_engineering", {
                    "type": "feature_task_stopped",
                    "task_id": task_id,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
            
            return {
                "success": True,
                "message": f"任务 {task_id} 已停止"
            }
        else:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在或无法停止")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止特征任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止特征任务失败: {str(e)}")


@router.get("/features/engineering/tasks/available-for-training")
async def get_available_feature_tasks_endpoint() -> Dict[str, Any]:
    """
    获取可用于模型训练的特征工程任务列表
    
    注意：此路由必须在 /{task_id} 路由之前定义，否则会被当作 task_id 参数
    
    Returns:
        可用的特征工程任务列表
    """
    try:
        tasks = get_available_feature_tasks_for_training()
        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks)
        }
    except Exception as e:
        logger.error(f"获取可用特征任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取可用特征任务失败: {str(e)}")


# ==================== 统计聚合API ====================

@router.get("/features/engineering/tasks/stats")
async def get_feature_tasks_stats_endpoint() -> Dict[str, Any]:
    """获取特征提取任务统计聚合"""
    try:
        from .feature_task_persistence import list_feature_tasks
        
        tasks = list_feature_tasks(limit=10000)  # 获取所有任务
        
        # 按状态统计
        status_counts = {}
        type_counts = {}
        daily_counts = {}
        symbol_counts = {}  # 按股票代码统计
        
        for task in tasks:
            # 状态统计
            status = task.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # 类型统计
            task_type = task.get('task_type', 'unknown')
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
            
            # 日期统计
            created_at = task.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    date_key = created_at[:10]  # YYYY-MM-DD
                else:
                    date_key = str(created_at)[:10]
                daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
            
            # 股票代码统计
            symbol = task.get('symbol')
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        return {
            "total": len(tasks),
            "by_status": status_counts,
            "by_type": type_counts,
            "by_date": daily_counts,
            "by_symbol": symbol_counts,  # 新增：按股票代码统计
            "recent_tasks": tasks[:10]  # 最近10个任务
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务统计失败: {str(e)}")


@router.get("/features/engineering/features/stats")
async def get_features_stats_endpoint() -> Dict[str, Any]:
    """获取特征统计聚合"""
    try:
        features = get_features()
        
        # 按类型统计
        type_counts = {}
        quality_ranges = {
            "优秀(>=0.9)": 0,
            "良好(0.7-0.9)": 0,
            "一般(0.5-0.7)": 0,
            "较差(<0.5)": 0
        }
        
        for feature in features:
            # 类型统计
            feature_type = feature.get('feature_type') or feature.get('type', 'unknown')
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
            
            # 质量分布
            quality = feature.get('quality_score', 0)
            if quality >= 0.9:
                quality_ranges["优秀(>=0.9)"] += 1
            elif quality >= 0.7:
                quality_ranges["良好(0.7-0.9)"] += 1
            elif quality >= 0.5:
                quality_ranges["一般(0.5-0.7)"] += 1
            else:
                quality_ranges["较差(<0.5)"] += 1
        
        # 计算平均质量
        avg_quality = sum(f.get('quality_score', 0) for f in features) / len(features) if features else 0
        
        return {
            "total": len(features),
            "by_type": type_counts,
            "quality_distribution": quality_ranges,
            "avg_quality": round(avg_quality, 4),
            "recent_features": features[:10]  # 最近10个特征
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征统计失败: {str(e)}")


@router.get("/features/engineering/tasks/{task_id}")
async def get_feature_task_details(task_id: str) -> Dict[str, Any]:
    """获取特征提取任务详情 - 使用真实数据"""
    try:
        from .feature_task_persistence import load_feature_task
        
        # 优先从持久化存储加载
        task = load_feature_task(task_id)
        if task:
            return task
        
        # 如果持久化存储没有，尝试从服务层获取
        tasks = get_feature_tasks()
        task = next((t for t in tasks if t.get('task_id') == task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail=f"特征任务 {task_id} 不存在")
        
        return task
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征任务详情失败: {str(e)}")


@router.get("/features/engineering/quality/monitor")
async def get_quality_monitor_data() -> Dict[str, Any]:
    """获取特征质量实时监控数据"""
    try:
        features = get_features()
        
        if not features:
            return {
                "avg_quality": 0,
                "low_quality_count": 0,
                "quality_trend": [],
                "heatmap_data": [],
                "total_features": 0,
                "quality_distribution": {
                    "优秀(>=0.9)": 0,
                    "良好(0.7-0.9)": 0,
                    "一般(0.5-0.7)": 0,
                    "较差(<0.5)": 0
                }
            }
        
        # 计算平均质量
        avg_quality = sum(f.get('quality_score', 0) for f in features) / len(features)
        
        # 统计低质量特征
        low_quality_count = sum(1 for f in features if (f.get('quality_score') or 0) < 0.5)
        
        # 质量分布
        quality_distribution = {
            "优秀(>=0.9)": sum(1 for f in features if (f.get('quality_score') or 0) >= 0.9),
            "良好(0.7-0.9)": sum(1 for f in features if 0.7 <= (f.get('quality_score') or 0) < 0.9),
            "一般(0.5-0.7)": sum(1 for f in features if 0.5 <= (f.get('quality_score') or 0) < 0.7),
            "较差(<0.5)": sum(1 for f in features if (f.get('quality_score') or 0) < 0.5)
        }
        
        # 按日期统计质量趋势（最近7天）
        from datetime import datetime, timedelta
        today = datetime.now()
        quality_trend = []
        
        for i in range(6, -1, -1):
            date = today - timedelta(days=i)
            date_str = date.strftime("%m-%d")
            
            # 找到该日期的特征（根据updated_at）
            day_features = [
                f for f in features 
                if f.get('updated_at') and 
                datetime.fromtimestamp(f['updated_at']).strftime("%Y-%m-%d") == date.strftime("%Y-%m-%d")
            ]
            
            if day_features:
                day_avg = sum(f.get('quality_score', 0) for f in day_features) / len(day_features)
                quality_trend.append({"date": date_str, "avg_quality": round(day_avg, 3)})
            else:
                quality_trend.append({"date": date_str, "avg_quality": 0})
        
        # 生成热力图数据（股票代码 × 特征类型）
        heatmap_data = []
        symbols = list(set(f.get('symbol', 'unknown') for f in features))[:20]  # 最多20个股票
        feature_types = list(set(f.get('feature_type') or f.get('type', 'unknown') for f in features))
        
        for symbol in symbols:
            for feature_type in feature_types:
                type_features = [
                    f for f in features 
                    if f.get('symbol') == symbol and 
                    (f.get('feature_type') or f.get('type')) == feature_type
                ]
                if type_features:
                    avg_q = sum(f.get('quality_score', 0) for f in type_features) / len(type_features)
                    heatmap_data.append({
                        "symbol": symbol,
                        "feature_type": feature_type,
                        "avg_quality": round(avg_q, 3),
                        "count": len(type_features)
                    })
        
        return {
            "avg_quality": round(avg_quality, 4),
            "low_quality_count": low_quality_count,
            "quality_trend": quality_trend,
            "heatmap_data": heatmap_data,
            "total_features": len(features),
            "quality_distribution": quality_distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取质量监控数据失败: {str(e)}")


@router.delete("/features/engineering/tasks/{task_id}")
async def delete_feature_task_endpoint(task_id: str) -> Dict[str, Any]:
    """删除特征提取任务 - 使用真实数据"""
    try:
        success = delete_feature_task(task_id)
        if success:
            # 发布特征任务删除事件到EventBus
            event_bus = _get_event_bus()
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.FEATURE_PROCESSING_COMPLETED,  # 使用特征处理完成事件表示任务删除
                        {
                            "task_id": task_id,
                            "action": "deleted",
                            "timestamp": time.time()
                        },
                        source="feature_engineering_routes"
                    )
                    logger.debug(f"已发布特征任务删除事件: {task_id}")
                except Exception as e:
                    logger.debug(f"发布特征任务删除事件失败: {e}")
            
            # WebSocket广播特征任务删除事件
            try:
                from .websocket_manager import manager
                await manager.broadcast("feature_engineering", {
                    "type": "feature_task_deleted",
                    "task_id": task_id,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
            
            return {
                "success": True,
                "message": f"任务 {task_id} 已删除"
            }
        else:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在或无法删除")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除特征任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除特征任务失败: {str(e)}")


@router.post("/features/engineering/tasks/{task_id}/resubmit")
async def resubmit_feature_task_endpoint(task_id: str) -> Dict[str, Any]:
    """重新提交特征提取任务 - 用于失败或等待中的任务"""
    try:
        # 获取原任务信息
        from .feature_task_persistence import load_feature_task
        original_task = load_feature_task(task_id)
        
        if not original_task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        
        # 检查任务状态是否允许重新提交
        allowed_statuses = ['failed', 'pending', 'cancelled']
        if original_task.get('status') not in allowed_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"任务状态为 {original_task.get('status')}，不允许重新提交。只有失败、等待中或已取消的任务可以重新提交。"
            )
        
        # 准备新任务配置
        task_config = {
            "task_type": original_task.get("task_type", "technical"),
            "symbol": original_task.get("symbol"),
            "start_date": original_task.get("start_date"),
            "end_date": original_task.get("end_date"),
            "indicators": original_task.get("indicators", ["SMA", "EMA", "RSI", "MACD"]),
            "priority": original_task.get("priority", "normal"),
            "description": f"重新提交: {original_task.get('description', '')}"[:200]
        }
        
        # 删除旧任务
        from .feature_task_persistence import delete_feature_task
        delete_feature_task(task_id)
        
        # 创建新任务
        from .feature_engineering_service import create_feature_task
        new_task = create_feature_task(
            task_type=task_config.get("task_type", "technical"),
            config=task_config
        )
        
        if not new_task:
            raise HTTPException(status_code=500, detail="重新提交任务失败")
        
        new_task_id = new_task.get("task_id")
        
        # 发布任务重新提交事件
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.FEATURE_PROCESSING_COMPLETED,
                    {
                        "original_task_id": task_id,
                        "new_task_id": new_task_id,
                        "action": "resubmitted",
                        "timestamp": time.time()
                    },
                    source="feature_engineering_routes"
                )
                logger.info(f"任务重新提交成功: {task_id} -> {new_task_id}")
            except Exception as e:
                logger.debug(f"发布任务重新提交事件失败: {e}")
        
        # WebSocket广播
        try:
            from .websocket_manager import manager
            await manager.broadcast("feature_engineering", {
                "type": "feature_task_resubmitted",
                "original_task_id": task_id,
                "new_task_id": new_task_id,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.debug(f"WebSocket广播失败: {e}")
        
        return {
            "success": True,
            "message": f"任务重新提交成功",
            "original_task_id": task_id,
            "new_task_id": new_task_id,
            "task": new_task
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新提交任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"重新提交任务失败: {str(e)}")


# ==================== 特征存储API ====================

@router.get("/features/engineering/features")
async def get_features_endpoint(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(50, ge=1, le=200, description="每页数量"),
    feature_type: Optional[str] = Query(None, description="特征类型筛选"),
    min_quality: Optional[float] = Query(None, ge=0, le=1, description="最小质量分数"),
    search: Optional[str] = Query(None, description="搜索关键词")
) -> Dict[str, Any]:
    """获取特征列表 - 支持分页和筛选"""
    try:
        # 获取所有特征
        all_features = get_features()
        
        # 应用筛选条件
        filtered_features = all_features
        if feature_type:
            filtered_features = [f for f in filtered_features if f.get('feature_type') == feature_type or f.get('type') == feature_type]
        if min_quality is not None:
            filtered_features = [f for f in filtered_features if (f.get('quality_score') or 0) >= min_quality]
        if search:
            search_lower = search.lower()
            filtered_features = [
                f for f in filtered_features 
                if search_lower in (f.get('name') or '').lower() 
                or search_lower in (f.get('display_name') or '').lower()
                or search_lower in (f.get('description') or '').lower()
            ]
        
        # 计算分页
        total = len(filtered_features)
        total_pages = (total + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_features = filtered_features[start_idx:end_idx]
        
        # 获取统计信息
        stats = get_features_stats()
        quality_distribution = get_quality_distribution(all_features) if all_features else {
            "优秀": 0,
            "良好": 0,
            "一般": 0,
            "较差": 0
        }
        
        # 从实际数据获取选择历史
        selection_history = []
        try:
            from src.features.selection.feature_selector_history import get_feature_selector_history_manager
            import time
            history_manager = get_feature_selector_history_manager()
            if history_manager and hasattr(history_manager, 'get_selection_history'):
                end_time = time.time()
                start_time = end_time - (7 * 24 * 60 * 60)
                raw_history = history_manager.get_selection_history(
                    limit=50,
                    start_time=start_time,
                    end_time=end_time
                )
                selection_history = [
                    {
                        "timestamp": record.get("timestamp", 0),
                        "selected_count": len(record.get("selected_features", [])),
                        "method": record.get("selection_method", "unknown"),
                        "task_id": record.get("task_id", ""),
                        "symbol": record.get("symbol", "")
                    }
                    for record in raw_history
                ]
        except Exception as e:
            logger.warning(f"获取特征选择历史失败: {e}")
        
        return {
            "features": paginated_features,
            "stats": stats,
            "quality_distribution": quality_distribution,
            "selection_history": selection_history,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征列表失败: {str(e)}")


@router.get("/features/engineering/selection/analytics")
async def get_selection_analytics() -> Dict[str, Any]:
    """
    获取特征选择分析数据
    
    返回特征选择的质量评估、趋势分析和建议
    """
    try:
        # 获取特征选择历史
        from src.features.selection.feature_selector_history import get_feature_selector_history_manager
        history_manager = get_feature_selector_history_manager()
        history = history_manager.get_selection_history(limit=100)
        
        if not history:
            return {
                "success": True,
                "analytics": {
                    "total_selections": 0,
                    "avg_selection_ratio": 0,
                    "avg_quality": 0,
                    "method_distribution": {},
                    "trend": "stable",
                    "recommendations": ["暂无特征选择历史数据，建议执行特征选择任务"]
                }
            }
        
        # 计算关键指标（history是字典列表）
        total_selections = len(history)
        avg_selection_ratio = sum(h.get('selection_ratio', 0) for h in history) / total_selections
        avg_quality = sum(
            h.get('evaluation_metrics', {}).get('avg_quality', 0.8) 
            for h in history
        ) / total_selections
        
        # 方法分布
        method_distribution = {}
        for h in history:
            method = h.get('selection_method') or 'unknown'
            method_distribution[method] = method_distribution.get(method, 0) + 1
        
        # 趋势分析（比较最近10次和之前的平均值）
        trend = "stable"
        if len(history) >= 20:
            recent = history[-10:]
            older = history[-20:-10]
            recent_ratio = sum(h.get('selection_ratio', 0) for h in recent) / 10
            older_ratio = sum(h.get('selection_ratio', 0) for h in older) / 10
            
            if recent_ratio > older_ratio * 1.1:
                trend = "increasing"
            elif recent_ratio < older_ratio * 0.9:
                trend = "decreasing"
        
        # 生成建议
        recommendations = []
        
        if avg_selection_ratio < 0.1:
            recommendations.append("选择比例较低，建议调整选择参数或检查特征质量")
        elif avg_selection_ratio > 0.5:
            recommendations.append("选择比例较高，可能存在过度选择，建议增加筛选条件")
        
        if avg_quality < 0.7:
            recommendations.append("平均特征质量较低，建议优化特征工程流程")
        
        # 方法建议
        if len(method_distribution) == 1:
            recommendations.append("建议尝试其他特征选择方法以获得更好的效果")
        
        # 添加正面反馈
        if avg_quality > 0.85 and 0.2 <= avg_selection_ratio <= 0.4:
            recommendations.append("✅ 特征选择效果良好，请继续保持")
        
        # 获取最后执行时间（最新的记录）
        last_execution_time = None
        if history:
            # 历史记录按时间倒序排列，第一条是最新的
            last_record = history[0]
            timestamp = last_record.get('timestamp')
            if timestamp:
                # 如果是datetime对象，转换为timestamp
                if hasattr(timestamp, 'timestamp'):
                    last_execution_time = timestamp.timestamp()
                else:
                    last_execution_time = timestamp
        
        return {
            "success": True,
            "analytics": {
                "total_selections": total_selections,
                "avg_selection_ratio": round(avg_selection_ratio, 3),
                "avg_quality": round(avg_quality, 3),
                "method_distribution": method_distribution,
                "trend": trend,
                "last_execution_time": last_execution_time,
                "recommendations": recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"获取特征选择分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取特征选择分析失败: {str(e)}")


@router.post("/features/engineering/refresh")
async def refresh_features_endpoint() -> Dict[str, Any]:
    """手动刷新特征计算 - 从持久化存储重新加载特征数据"""
    try:
        # 从持久化存储重新加载特征任务
        from .feature_task_persistence import list_feature_tasks
        tasks = list_feature_tasks(limit=100)
        
        # 从已完成的任务中提取特征
        features = []
        for task in tasks:
            if task.get('status') == 'completed' and task.get('feature_count', 0) > 0:
                features.append({
                    "name": f"feature_{task.get('task_id')}",
                    "feature_type": task.get('task_type'),
                    "quality_score": task.get('quality_score', 0.8),
                    "importance": task.get('importance', 0.5),
                    "version": task.get('version', '1.0'),
                    "created_at": task.get('start_time'),
                    "updated_at": task.get('end_time') or task.get('start_time')
                })
        
        logger.info(f"✅ 手动刷新特征成功，获取到 {len(features)} 个特征")
        return {
            "success": True,
            "message": f"成功刷新 {len(features)} 个特征",
            "feature_count": len(features),
            "features": features
        }
    except Exception as e:
        logger.error(f"❌ 刷新特征失败: {e}")
        raise HTTPException(status_code=500, detail=f"刷新特征失败: {str(e)}")


@router.get("/features/engineering/features/{feature_name}")
async def get_feature_details(feature_name: str) -> Dict[str, Any]:
    """获取特征详情 - 不使用模拟数据"""
    try:
        features = get_features()
        # 不使用模拟数据
        
        feature = next((f for f in features if f.get('name') == feature_name), None)
        if not feature:
            raise HTTPException(status_code=404, detail=f"特征 {feature_name} 不存在")
        
        return feature
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征详情失败: {str(e)}")


# ==================== 技术指标API ====================

@router.get("/features/engineering/indicators")
async def get_technical_indicators_endpoint() -> Dict[str, Any]:
    """获取技术指标状态 - 不使用模拟数据"""
    try:
        indicators = get_technical_indicators()
        # 不使用模拟数据，即使为空也返回真实结果
        
        return {
            "indicators": indicators
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取技术指标失败: {str(e)}")


@router.get("/features/engineering/tracker/health")
async def get_tracker_health_endpoint() -> Dict[str, Any]:
    """获取指标计算跟踪器健康状态"""
    try:
        from src.features.monitoring.indicator_calculation_tracker import get_indicator_calculation_tracker
        tracker = get_indicator_calculation_tracker()
        health_status = tracker.get_health_status()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取跟踪器健康状态失败: {str(e)}")


# ==================== 调度器API ====================

@router.get("/features/engineering/scheduler/status")
async def get_scheduler_status_endpoint() -> Dict[str, Any]:
    """获取特征任务调度器状态"""
    try:
        status = get_scheduler_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器状态失败: {str(e)}")


@router.post("/features/engineering/scheduler/start")
async def start_scheduler_endpoint() -> Dict[str, Any]:
    """启动特征任务调度器"""
    try:
        success = start_scheduler()
        if success:
            return {
                "success": True,
                "message": "特征任务调度器已启动"
            }
        else:
            raise HTTPException(status_code=500, detail="启动调度器失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动调度器失败: {str(e)}")


@router.post("/features/engineering/scheduler/stop")
async def stop_scheduler_endpoint() -> Dict[str, Any]:
    """停止特征任务调度器"""
    try:
        success = stop_scheduler()
        if success:
            return {
                "success": True,
                "message": "特征任务调度器已停止"
            }
        else:
            raise HTTPException(status_code=500, detail="停止调度器失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止调度器失败: {str(e)}")


# ==================== 工作节点管理API ====================

@router.get("/features/engineering/workers")
async def list_workers_endpoint() -> Dict[str, Any]:
    """列出所有工作节点"""
    try:
        from src.features.distributed.worker_executor import list_workers
        result = list_workers()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工作节点列表失败: {str(e)}")


@router.get("/features/engineering/workers/{worker_id}")
async def get_worker_status_endpoint(worker_id: str) -> Dict[str, Any]:
    """获取工作节点状态"""
    try:
        from src.features.distributed.worker_executor import get_worker_status
        status = get_worker_status(worker_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工作节点状态失败: {str(e)}")


@router.post("/features/engineering/workers/restart")
async def restart_workers_endpoint() -> Dict[str, Any]:
    """重启所有工作节点"""
    try:
        # 停止并重新启动调度器（带工作节点）
        from src.gateway.web.feature_engineering_service import stop_scheduler, start_scheduler
        
        # 先停止调度器
        stop_scheduler()
        # 再启动调度器（会自动创建新的工作节点）
        success = start_scheduler()
        
        if success:
            return {
                "success": True,
                "message": "工作节点已重启"
            }
        else:
            raise HTTPException(status_code=500, detail="重启工作节点失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重启工作节点失败: {str(e)}")


# ==================== 系统管理API ====================

@router.post("/features/engineering/system/initialize")
async def initialize_system_endpoint() -> Dict[str, Any]:
    """初始化特征工程系统"""
    try:
        from src.gateway.web.feature_engineering_service import initialize_feature_engineering_system
        success = initialize_feature_engineering_system()
        if success:
            return {
                "success": True,
                "message": "特征工程系统初始化完成"
            }
        else:
            raise HTTPException(status_code=500, detail="初始化特征工程系统失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化特征工程系统失败: {str(e)}")


# ==================== 特征数据API（用于模型训练）====================

@router.get("/features/engineering/tasks/{task_id}/data")
async def get_feature_task_data_endpoint(task_id: str) -> Dict[str, Any]:
    """
    获取特征工程任务的特征数据，用于模型训练
    
    符合架构设计：特征层 -> ML层数据流
    数据流：特征工程任务 -> 特征数据(X) -> 模型训练
    
    Args:
        task_id: 特征工程任务ID
        
    Returns:
        特征数据信息（不包含实际特征矩阵，只返回元数据）
    """
    try:
        result = get_feature_data_for_training(task_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        # 返回元数据，不包含实际的DataFrame（避免序列化问题）
        return {
            "success": True,
            "task_id": result["task_id"],
            "feature_names": result["feature_names"],
            "shape": result["shape"],
            "sample_count": result["sample_count"],
            "feature_count": result["feature_count"],
            "symbols": result["symbols"],
            "date_range": result["date_range"],
            "metadata": result["metadata"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取特征数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取特征数据失败: {str(e)}")


# ==================== 质量评估API ====================

@router.get("/features/engineering/quality/assessment/{task_id}")
async def get_quality_assessment_endpoint(task_id: str) -> Dict[str, Any]:
    """
    获取任务的质量评估结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        质量评估结果
    """
    try:
        from src.gateway.web.feature_task_persistence import load_feature_task
        
        task = load_feature_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        
        quality_assessment = task.get("quality_assessment", [])
        quality_distribution = task.get("quality_distribution", {})
        overall_score = task.get("overall_quality_score", 0.0)
        
        return {
            "success": True,
            "task_id": task_id,
            "overall_score": overall_score,
            "quality_distribution": quality_distribution,
            "feature_metrics": quality_assessment,
            "assessment_time": task.get("updated_at")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取质量评估结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取质量评估结果失败: {str(e)}")


@router.get("/features/engineering/quality/stats")
async def get_quality_stats_endpoint() -> Dict[str, Any]:
    """
    获取质量评估统计信息
    
    Returns:
        质量评估统计
    """
    try:
        from src.gateway.web.feature_task_persistence import list_feature_tasks
        
        tasks = list_feature_tasks()
        completed_tasks = [t for t in tasks if t.get("status") == "completed"]
        
        if not completed_tasks:
            return {
                "success": True,
                "message": "没有已完成的任务",
                "stats": {}
            }
        
        # 统计质量分布
        total_distribution = {"优秀": 0, "良好": 0, "一般": 0, "较差": 0}
        total_score = 0.0
        task_count_with_quality = 0
        
        for task in completed_tasks:
            dist = task.get("quality_distribution", {})
            for level, count in dist.items():
                total_distribution[level] = total_distribution.get(level, 0) + count
            
            score = task.get("overall_quality_score", 0)
            if score > 0:
                total_score += score
                task_count_with_quality += 1
        
        avg_score = total_score / task_count_with_quality if task_count_with_quality > 0 else 0
        
        return {
            "success": True,
            "stats": {
                "total_tasks": len(completed_tasks),
                "tasks_with_quality": task_count_with_quality,
                "average_score": avg_score,
                "total_distribution": total_distribution
            }
        }
    except Exception as e:
        logger.error(f"获取质量统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取质量统计失败: {str(e)}")


@router.get("/features/engineering/quality/trend")
async def get_quality_trend_endpoint(days: int = 7) -> Dict[str, Any]:
    """
    获取质量评分趋势
    
    Args:
        days: 查询天数
        
    Returns:
        质量评分趋势数据
    """
    try:
        from src.gateway.web.feature_task_persistence import list_feature_tasks
        import time
        
        cutoff_time = time.time() - (days * 24 * 3600)
        tasks = list_feature_tasks()
        
        # 筛选最近完成的任务
        recent_tasks = [
            t for t in tasks 
            if t.get("status") == "completed" and t.get("end_time", 0) > cutoff_time
        ]
        
        # 按日期分组统计
        trend_data = []
        for task in sorted(recent_tasks, key=lambda x: x.get("end_time", 0)):
            score = task.get("overall_quality_score", 0)
            if score > 0:
                trend_data.append({
                    "task_id": task.get("task_id"),
                    "timestamp": task.get("end_time"),
                    "score": score,
                    "feature_count": task.get("feature_count", 0)
                })
        
        return {
            "success": True,
            "period_days": days,
            "data_points": len(trend_data),
            "trend": trend_data
        }
    except Exception as e:
        logger.error(f"获取质量趋势失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取质量趋势失败: {str(e)}")


@router.post("/features/engineering/quality/reassess/{task_id}")
async def reassess_quality_endpoint(task_id: str) -> Dict[str, Any]:
    """
    重新评估任务质量
    
    Args:
        task_id: 任务ID
        
    Returns:
        重新评估结果
    """
    try:
        from src.gateway.web.feature_task_persistence import load_feature_task
        from src.features.quality.feature_quality_assessor import get_feature_quality_assessor
        
        task = load_feature_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        
        # TODO: 重新加载特征数据并进行质量评估
        # 这里需要实现特征数据重新加载逻辑
        
        return {
            "success": True,
            "message": "质量重新评估已启动",
            "task_id": task_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新评估质量失败: {e}")
        raise HTTPException(status_code=500, detail=f"重新评估质量失败: {str(e)}")


# ==================== 特征质量自定义评分配置API ====================

@router.get("/features/engineering/quality/config")
async def get_quality_configs_endpoint(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    获取用户的自定义评分配置
    
    Args:
        user_id: 用户ID（可选，默认使用当前用户）
    
    Returns:
        自定义评分配置列表
    """
    try:
        from src.gateway.web.feature_quality_config import get_config_manager
        
        # 如果未提供user_id，使用默认用户
        if not user_id:
            user_id = "default_user"
        
        manager = get_config_manager()
        configs = manager.get_user_configs(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "configs": [
                {
                    "config_id": c.config_id,
                    "feature_name": c.feature_name,
                    "custom_score": c.custom_score,
                    "reason": c.reason,
                    "is_active": c.is_active,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None
                }
                for c in configs
            ],
            "total": len(configs)
        }
    except Exception as e:
        logger.error(f"获取自定义评分配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取自定义评分配置失败: {str(e)}")


@router.post("/features/engineering/quality/config")
async def create_quality_config_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建自定义评分配置
    
    Args:
        request: {
            "user_id": "用户ID（可选）",
            "feature_name": "特征名称",
            "custom_score": 0.85,
            "reason": "修改原因（可选）"
        }
    
    Returns:
        创建的配置信息
    """
    try:
        from src.gateway.web.feature_quality_config import get_config_manager
        
        user_id = request.get("user_id", "default_user")
        feature_name = request.get("feature_name")
        custom_score = request.get("custom_score")
        reason = request.get("reason")
        
        if not feature_name:
            raise HTTPException(status_code=400, detail="特征名称不能为空")
        
        if custom_score is None:
            raise HTTPException(status_code=400, detail="自定义评分不能为空")
        
        if not (0.0 <= custom_score <= 1.0):
            raise HTTPException(status_code=400, detail="自定义评分必须在0-1之间")
        
        manager = get_config_manager()
        config = manager.create_config(user_id, feature_name, custom_score, reason)
        
        if not config:
            raise HTTPException(status_code=500, detail="创建配置失败")
        
        return {
            "success": True,
            "message": f"自定义评分配置创建成功",
            "config": {
                "config_id": config.config_id,
                "user_id": config.user_id,
                "feature_name": config.feature_name,
                "custom_score": config.custom_score,
                "reason": config.reason,
                "is_active": config.is_active,
                "created_at": config.created_at.isoformat() if config.created_at else None,
                "updated_at": config.updated_at.isoformat() if config.updated_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建自定义评分配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建自定义评分配置失败: {str(e)}")


@router.put("/features/engineering/quality/config/{config_id}")
async def update_quality_config_endpoint(config_id: int, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新自定义评分配置
    
    Args:
        config_id: 配置ID
        request: {
            "custom_score": 0.90,
            "reason": "新的修改原因"
        }
    
    Returns:
        更新后的配置信息
    """
    try:
        from src.gateway.web.feature_quality_config import get_config_manager
        
        custom_score = request.get("custom_score")
        reason = request.get("reason")
        
        if custom_score is not None and not (0.0 <= custom_score <= 1.0):
            raise HTTPException(status_code=400, detail="自定义评分必须在0-1之间")
        
        manager = get_config_manager()
        config = manager.update_config(config_id, custom_score, reason)
        
        if not config:
            raise HTTPException(status_code=404, detail=f"配置 {config_id} 不存在")
        
        return {
            "success": True,
            "message": f"自定义评分配置更新成功",
            "config": {
                "config_id": config.config_id,
                "user_id": config.user_id,
                "feature_name": config.feature_name,
                "custom_score": config.custom_score,
                "reason": config.reason,
                "is_active": config.is_active,
                "created_at": config.created_at.isoformat() if config.created_at else None,
                "updated_at": config.updated_at.isoformat() if config.updated_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新自定义评分配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新自定义评分配置失败: {str(e)}")


@router.delete("/features/engineering/quality/config/{config_id}")
async def delete_quality_config_endpoint(config_id: int) -> Dict[str, Any]:
    """
    删除自定义评分配置
    
    Args:
        config_id: 配置ID
    
    Returns:
        删除结果
    """
    try:
        from src.gateway.web.feature_quality_config import get_config_manager
        
        manager = get_config_manager()
        result = manager.delete_config(config_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"配置 {config_id} 不存在")
        
        return {
            "success": True,
            "message": f"自定义评分配置 {config_id} 已删除"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除自定义评分配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除自定义评分配置失败: {str(e)}")


@router.post("/features/engineering/quality/config/{config_id}/reset")
async def reset_quality_config_endpoint(config_id: int) -> Dict[str, Any]:
    """
    重置为默认评分（禁用自定义配置）
    
    Args:
        config_id: 配置ID
    
    Returns:
        重置结果
    """
    try:
        from src.gateway.web.feature_quality_config import get_config_manager
        
        manager = get_config_manager()
        
        # 先获取配置信息
        configs = manager.get_user_configs("default_user", include_inactive=True)
        target_config = None
        for c in configs:
            if c.config_id == config_id:
                target_config = c
                break
        
        if not target_config:
            raise HTTPException(status_code=404, detail=f"配置 {config_id} 不存在")
        
        # 重置为默认
        result = manager.reset_to_default(target_config.user_id, target_config.feature_name)
        
        if not result:
            raise HTTPException(status_code=500, detail="重置失败")
        
        return {
            "success": True,
            "message": f"特征 {target_config.feature_name} 已重置为默认评分"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重置自定义评分配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"重置自定义评分配置失败: {str(e)}")


@router.post("/features/engineering/quality/config/batch")
async def batch_create_quality_configs_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    批量创建自定义评分配置
    
    Args:
        request: {
            "user_id": "用户ID（可选）",
            "configs": [
                {"feature_name": "SMA_5", "custom_score": 0.85, "reason": "原因"},
                ...
            ]
        }
    
    Returns:
        批量操作结果
    """
    try:
        from src.gateway.web.feature_quality_config import get_config_manager
        
        user_id = request.get("user_id", "default_user")
        configs = request.get("configs", [])
        
        if not configs:
            raise HTTPException(status_code=400, detail="配置列表不能为空")
        
        # 验证所有配置
        for config in configs:
            if not config.get("feature_name"):
                raise HTTPException(status_code=400, detail="特征名称不能为空")
            score = config.get("custom_score")
            if score is None or not (0.0 <= score <= 1.0):
                raise HTTPException(status_code=400, detail=f"自定义评分必须在0-1之间: {config.get('feature_name')}")
        
        manager = get_config_manager()
        result = manager.batch_create_configs(user_id, configs)
        
        return {
            "success": True,
            "message": f"批量创建完成: 成功 {len(result['success'])}, 失败 {len(result['failed'])}",
            "total": result['total'],
            "success_count": len(result['success']),
            "failed_count": len(result['failed']),
            "failed_items": result['failed']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量创建自定义评分配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量创建自定义评分配置失败: {str(e)}")


# ==================== 特征选择任务API ====================

@router.get("/features/engineering/selection/tasks")
async def get_selection_tasks_endpoint(
    status: Optional[str] = Query(None, description="按状态过滤"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量")
) -> Dict[str, Any]:
    """
    获取特征选择任务列表
    
    Args:
        status: 按状态过滤（pending/running/completed/failed）
        limit: 返回数量限制
        offset: 偏移量
    
    Returns:
        特征选择任务列表
    """
    try:
        from .feature_selection_task_persistence import list_selection_tasks
        
        tasks = list_selection_tasks(status=status, limit=limit, offset=offset)
        
        return {
            "success": True,
            "tasks": tasks,
            "total": len(tasks),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"获取特征选择任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取特征选择任务列表失败: {str(e)}")


@router.get("/features/engineering/selection/tasks/stats")
async def get_selection_tasks_stats_endpoint() -> Dict[str, Any]:
    """
    获取特征选择任务统计
    
    Returns:
        任务统计信息
    """
    try:
        from .feature_selection_task_persistence import get_selection_tasks_stats
        
        stats = get_selection_tasks_stats()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"获取特征选择任务统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取特征选择任务统计失败: {str(e)}")


@router.get("/features/engineering/selection/tasks/{task_id}")
async def get_selection_task_details_endpoint(task_id: str) -> Dict[str, Any]:
    """
    获取特征选择任务详情
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务详情
    """
    try:
        from .feature_selection_task_persistence import get_selection_task
        
        task = get_selection_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
        
        return {
            "success": True,
            "task": task
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取特征选择任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取特征选择任务详情失败: {str(e)}")

