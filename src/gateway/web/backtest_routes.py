"""
策略回测API路由
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理，使用BusinessProcessOrchestrator进行业务流程编排
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, Set
from datetime import datetime
import logging
import time

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from .backtest_service import run_backtest, get_backtest_result, list_backtests

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
    
    # 降级方案：直接创建（业务流程编排器用于管理回测流程）
    try:
        from src.infrastructure.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
        orchestrator = BusinessProcessOrchestrator()
        orchestrator.initialize()
        return orchestrator
    except Exception as e:
        logger.debug(f"创建业务流程编排器失败（可选功能）: {e}")
        return None

def _get_websocket_manager():
    """获取WebSocket管理器实例（用于实时广播）"""
    try:
        from .websocket_manager import ConnectionManager
        # 使用单例模式
        if not hasattr(_get_websocket_manager, "_instance"):
            _get_websocket_manager._instance = ConnectionManager()
        return _get_websocket_manager._instance
    except Exception as e:
        logger.debug(f"获取WebSocket管理器失败: {e}")
        return None


# 量化交易系统安全要求：回测参数限制常量
MAX_DATE_RANGE_DAYS = 365 * 5  # 最大5年
MIN_INITIAL_CAPITAL = 10000  # 最小初始资金1万
MAX_INITIAL_CAPITAL = 10000000  # 最大初始资金1000万
MAX_COMMISSION_RATE = 0.01  # 最大手续费率1%
MAX_SLIPPAGE = 0.01  # 最大滑点1%
MAX_STOP_LOSS = 0.20  # 最大止损20%
MAX_TAKE_PROFIT = 0.50  # 最大止盈50%


class BacktestRequest(BaseModel):
    """回测请求模型 - 符合量化交易系统安全要求"""
    strategy_id: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage: float = 0.001
    market_impact: float = 0.001
    # 量化交易系统风险控制要求：止损止盈设置
    stop_loss: float = 0.05  # 默认5%止损
    take_profit: float = 0.10  # 默认10%止盈
    # 量化交易系统风险控制要求：仓位控制
    max_position_size: float = 0.3  # 默认最大30%仓位
    position_sizing_strategy: str = "fixed"  # 仓位策略: fixed/percent/equal
    # 量化交易系统风险控制要求：风险敞口限制
    max_risk_per_trade: float = 0.02  # 单笔交易最大风险2%
    max_total_risk: float = 0.1  # 总风险敞口10%


class BacktestResponse(BaseModel):
    """回测响应模型"""
    backtest_id: str
    strategy_id: str
    status: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: list
    trades: list
    metrics: Dict[str, Any]
    created_at: str


class ExportRequest(BaseModel):
    """导出请求模型"""
    strategy_id: str
    format: str
    include_metrics: bool = True
    include_trades: bool = True
    include_positions: bool = True
    include_returns: bool = True


@router.post("/backtest/run", response_model=BacktestResponse, tags=["backtest"])
async def run_backtest_endpoint(request: BacktestRequest):
    """
    运行策略回测
    符合架构设计：使用业务流程编排器管理回测流程，使用事件总线发布回测事件，使用WebSocket进行实时广播
    
    Args:
        request: 回测请求参数
    
    Returns:
        回测结果
    """
    try:
        # 使用业务流程编排器管理回测流程（符合架构设计）
        orchestrator = _get_orchestrator()
        process_id = None
        if orchestrator:
            try:
                from src.infrastructure.orchestration.models.process_models import BusinessProcessState
                import time
                backtest_id_preview = f"backtest_{request.strategy_id}_{int(time.time())}"
                process_id = f"backtest_run_{backtest_id_preview}_{int(time.time())}"
                
                # 启动回测业务流程（符合架构设计：业务流程编排）
                try:
                    orchestrator.start_process(
                        process_type="strategy_backtest",
                        process_id=process_id,
                        config={
                            "strategy_id": request.strategy_id,
                            "start_date": request.start_date,
                            "end_date": request.end_date,
                            "initial_capital": request.initial_capital,
                            "commission_rate": request.commission_rate,
                            "slippage": request.slippage,
                            "market_impact": request.market_impact
                        }
                    )
                    logger.info(f"回测业务流程已启动: {process_id}")
                except Exception as e:
                    logger.debug(f"启动回测业务流程失败（可选功能）: {e}")
            except Exception as e:
                logger.debug(f"业务流程编排器初始化检查失败（可选功能）: {e}")
        
        # 发布回测开始事件（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    event_type=EventType.PARAMETER_OPTIMIZATION_STARTED,  # 使用策略优化事件类型（回测属于策略优化）
                    payload={
                        "backtest_id": f"backtest_{request.strategy_id}_{int(time.time())}",
                        "strategy_id": request.strategy_id,
                        "start_date": request.start_date,
                        "end_date": request.end_date,
                        "process_id": process_id
                    }
                )
                logger.info(f"回测开始事件已发布: {request.strategy_id}")
            except Exception as e:
                logger.debug(f"发布回测开始事件失败（可选功能）: {e}")
        
        # 量化交易系统安全要求：验证日期范围
        try:
            start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days
            
            if days > MAX_DATE_RANGE_DAYS:
                raise HTTPException(
                    status_code=400,
                    detail=f"日期范围不能超过{MAX_DATE_RANGE_DAYS}天（约5年），当前选择: {days}天"
                )
            
            if days < 30:
                logger.warning(f"日期范围较短: {days}天，建议至少30天")
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式错误，请使用YYYY-MM-DD格式")
        
        # 量化交易系统安全要求：验证初始资金范围
        if not (MIN_INITIAL_CAPITAL <= request.initial_capital <= MAX_INITIAL_CAPITAL):
            raise HTTPException(
                status_code=400,
                detail=f"初始资金必须在{MIN_INITIAL_CAPITAL}到{MAX_INITIAL_CAPITAL}之间"
            )
        
        # 量化交易系统安全要求：验证交易费用合理性
        if request.commission_rate > MAX_COMMISSION_RATE:
            raise HTTPException(
                status_code=400,
                detail=f"手续费率不能超过{MAX_COMMISSION_RATE}（{MAX_COMMISSION_RATE*100}%）"
            )
        
        if request.slippage > MAX_SLIPPAGE:
            raise HTTPException(
                status_code=400,
                detail=f"滑点不能超过{MAX_SLIPPAGE}（{MAX_SLIPPAGE*100}%）"
            )
        
        # 量化交易系统安全要求：验证风险控制参数
        if request.stop_loss > MAX_STOP_LOSS:
            raise HTTPException(
                status_code=400,
                detail=f"止损比例不能超过{MAX_STOP_LOSS}（{MAX_STOP_LOSS*100}%）"
            )
        
        if request.take_profit > MAX_TAKE_PROFIT:
            raise HTTPException(
                status_code=400,
                detail=f"止盈比例不能超过{MAX_TAKE_PROFIT}（{MAX_TAKE_PROFIT*100}%）"
            )
        
        # 运行回测（调用服务层）
        result = await run_backtest(
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage=request.slippage,
            market_impact=request.market_impact,
            # 量化交易系统风险控制参数
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            max_position_size=request.max_position_size,
            position_sizing_strategy=request.position_sizing_strategy,
            max_risk_per_trade=request.max_risk_per_trade,
            max_total_risk=request.max_total_risk
        )
        
        # 更新业务流程状态（符合架构设计：业务流程编排）
        if orchestrator and process_id:
            try:
                from src.infrastructure.orchestration.models.process_models import BusinessProcessState
                orchestrator.update_process_state(
                    process_id=process_id,
                    new_state=BusinessProcessState.COMPLETED,
                    metadata={
                        "backtest_id": result.get("backtest_id"),
                        "status": result.get("status"),
                        "total_return": result.get("total_return"),
                        "annualized_return": result.get("annualized_return")
                    }
                )
                logger.info(f"回测业务流程状态已更新: {process_id} -> COMPLETED")
            except Exception as e:
                logger.debug(f"更新回测业务流程状态失败（可选功能）: {e}")
        
        # 发布回测完成事件（符合架构设计：事件驱动通信）
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    event_type=EventType.PARAMETER_OPTIMIZATION_COMPLETED,  # 使用策略优化完成事件类型
                    payload={
                        "backtest_id": result.get("backtest_id"),
                        "strategy_id": result.get("strategy_id"),
                        "status": result.get("status"),
                        "total_return": result.get("total_return"),
                        "annualized_return": result.get("annualized_return"),
                        "process_id": process_id
                    }
                )
                logger.info(f"回测完成事件已发布: {result.get('backtest_id')}")
            except Exception as e:
                logger.debug(f"发布回测完成事件失败（可选功能）: {e}")
        
        # WebSocket实时广播回测结果（符合架构设计：实时通信）
        manager = _get_websocket_manager()
        if manager:
            try:
                import time
                await manager.broadcast("backtest_progress", {
                    "type": "backtest_progress",
                    "data": {
                        "backtest_id": result.get("backtest_id"),
                        "strategy_id": result.get("strategy_id"),
                        "status": result.get("status"),
                        "progress": 1.0,
                        "total_return": result.get("total_return"),
                        "annualized_return": result.get("annualized_return")
                    },
                    "timestamp": time.time()
                })
                logger.debug(f"回测结果已广播到WebSocket: {result.get('backtest_id')}")
            except Exception as e:
                logger.debug(f"WebSocket广播回测结果失败（可选功能）: {e}")
        
        # 记录性能数据到性能监控服务
        try:
            from .strategy_performance_service import record_strategy_performance
            performance_metrics = {
                "total_return": result.get("total_return", 0.0),
                "annual_return": result.get("annualized_return", 0.0),
                "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                "max_drawdown": result.get("max_drawdown", 0.0),
                "win_rate": result.get("win_rate", 0.0)
            }
            record_strategy_performance(
                strategy_id=request.strategy_id,
                metrics=performance_metrics,
                period="daily",
                metadata={"source": "backtest", "backtest_id": result.get("backtest_id")}
            )
            logger.info(f"回测性能数据已记录: {request.strategy_id}")
        except Exception as e:
            logger.debug(f"记录回测性能数据失败（可选功能）: {e}")
        
        return BacktestResponse(**result)
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"回测执行失败: {str(e)}")


@router.get("/backtest/{backtest_id}", tags=["backtest"])
async def get_backtest_result_endpoint(backtest_id: str):
    """
    获取回测结果
    
    Args:
        backtest_id: 回测ID
    
    Returns:
        回测结果
    """
    try:
        result = await get_backtest_result(backtest_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"回测结果不存在: {backtest_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取回测结果失败: {str(e)}")


@router.get("/backtest", tags=["backtest"])
async def list_backtests_endpoint(strategy_id: Optional[str] = Query(None, description="策略ID过滤器")):
    """
    列出回测任务
    
    Args:
        strategy_id: 策略ID过滤器
    
    Returns:
        回测任务列表
    """
    try:
        results = await list_backtests(strategy_id)
        return {
            "backtests": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"获取回测列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取回测列表失败: {str(e)}")


@router.delete("/backtest/{backtest_id}", tags=["backtest"])
async def delete_backtest_endpoint(backtest_id: str):
    """
    删除回测结果
    
    Args:
        backtest_id: 回测ID
    
    Returns:
        删除结果
    """
    try:
        from .backtest_persistence import delete_backtest_result
        success = delete_backtest_result(backtest_id)
        if success:
            return {"success": True, "message": f"回测结果已删除: {backtest_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"回测结果不存在或删除失败: {backtest_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除回测结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除回测结果失败: {str(e)}")


# ==================== 模型预测支持的回测API（ML层 -> 策略层数据流）====================

@router.post("/backtest/model", tags=["backtest"])
async def run_backtest_with_model_endpoint(request: Dict[str, Any]):
    """
    使用训练好的模型进行回测
    
    符合架构设计：ML层 -> 策略层数据流
    数据流：训练好的模型 -> 模型预测 -> 交易信号 -> 策略回测
    
    请求示例:
    {
        "model_id": "model_123456",
        "start_date": "2025-01-01",
        "end_date": "2026-02-13",
        "initial_capital": 100000.0,
        "commission_rate": 0.001,
        "slippage": 0.001,
        "prediction_threshold": 0.5
    }
    
    Args:
        request: 回测请求参数
    
    Returns:
        回测结果
    """
    try:
        from .backtest_service import run_backtest_with_model
        
        model_id = request.get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="缺少模型ID")
        
        start_date = request.get("start_date")
        end_date = request.get("end_date")
        
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="缺少开始或结束日期")
        
        # 执行模型驱动的回测
        result = await run_backtest_with_model(
            model_id=model_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=request.get("initial_capital", 100000.0),
            commission_rate=request.get("commission_rate", 0.001),
            slippage=request.get("slippage", 0.001),
            market_impact=request.get("market_impact", 0.001),
            prediction_threshold=request.get("prediction_threshold", 0.5)
        )
        
        return {
            "success": True,
            "backtest": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型回测失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型回测失败: {str(e)}")


@router.get("/backtest/models/available", tags=["backtest"])
async def get_available_models_for_backtest_endpoint():
    """
    获取可用于回测的模型列表
    
    Returns:
        可用模型列表
    """
    try:
        from .backtest_service import get_available_models_for_backtest
        
        models = await get_available_models_for_backtest()
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        logger.error(f"获取可用模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取可用模型列表失败: {str(e)}")

