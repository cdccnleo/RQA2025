"""
策略执行监控路由模块
提供策略执行状态、性能指标、实时监控等API端点
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理，使用BusinessProcessOrchestrator进行业务流程编排
"""

import time
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

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
    
    # 降级方案：直接创建（业务流程编排器用于管理策略执行流程）
    try:
        from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
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


@router.get("/api/v1/strategy/execution/status", summary="获取策略执行状态", description="获取所有策略的当前执行状态，包括运行中、暂停和停止的策略数量，以及每个策略的详细信息。")
async def get_strategy_execution_status():
    """
    获取策略执行状态
    
    Returns:
        dict: 包含策略执行状态的字典，包括：
        - strategies: 策略列表
        - running_count: 运行中策略数量
        - paused_count: 暂停策略数量
        - stopped_count: 停止策略数量
        - total_count: 总策略数量
    """
    try:
        from .strategy_execution_service import get_strategy_execution_status
        return await get_strategy_execution_status()
    except Exception as e:
        logger.error(f"获取策略执行状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/execution/metrics", summary="获取策略执行性能指标", description="获取策略执行的性能指标，包括平均延迟、今日信号数、总交易数等。")
async def get_strategy_execution_metrics():
    """
    获取策略执行性能指标
    
    Returns:
        dict: 包含执行性能指标的字典，包括：
        - avg_latency: 平均处理延迟（毫秒）
        - today_signals: 今日信号数量
        - total_trades: 总交易数量
        - latency_history: 延迟历史数据
        - throughput_history: 吞吐量历史数据
    """
    try:
        from .strategy_execution_service import get_execution_metrics
        return await get_execution_metrics()
    except Exception as e:
        logger.error(f"获取执行指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/execution/{strategy_id}/start", summary="启动策略执行", description="启动指定策略的执行，包括注册到业务流程编排器、发布事件和WebSocket实时广播。")
async def start_strategy_execution(strategy_id: str):
    """
    启动策略执行
    
    Args:
        strategy_id: 策略ID
    
    Returns:
        dict: 包含启动结果的字典，包括：
        - success: 是否成功启动
        - message: 启动结果消息
        - strategy_id: 策略ID
        - timestamp: 启动时间戳
    
    Raises:
        HTTPException: 当策略不存在或启动失败时
    """
    try:
        # 获取业务流程编排器（符合架构设计）
        orchestrator = _get_orchestrator()
        if orchestrator:
            try:
                # 使用业务流程编排器启动策略执行流程
                process_id = orchestrator.start_process(
                    process_type="STRATEGY_EXECUTION",
                    initial_data={"strategy_id": strategy_id, "action": "start"}
                )
                logger.info(f"策略执行流程已启动: process_id={process_id}, strategy_id={strategy_id}")
            except Exception as e:
                logger.debug(f"使用业务流程编排器失败: {e}")
        
        from .strategy_execution_service import start_strategy
        success = await start_strategy(strategy_id)
        if success:
            # 发布执行启动事件（符合架构设计：事件驱动通信）
            event_bus = _get_event_bus()
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.EXECUTION_STARTED,
                        {"strategy_id": strategy_id, "action": "start"},
                        source="strategy_execution_routes"
                    )
                except Exception as e:
                    logger.debug(f"发布执行启动事件失败: {e}")
            
            # WebSocket实时广播（符合架构设计：实时更新）
            manager = _get_websocket_manager()
            if manager:
                try:
                    await manager.broadcast("execution_status", {
                        "type": "execution_status",
                        "action": "started",
                        "strategy_id": strategy_id,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.debug(f"WebSocket广播失败: {e}")
            
            return {
                "success": True,
                "message": f"策略 {strategy_id} 已启动",
                "strategy_id": strategy_id,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在或启动失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动策略执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.post("/api/v1/strategy/execution/{strategy_id}/pause", summary="暂停策略执行", description="暂停指定策略的执行，包括更新业务流程编排器状态、发布事件和WebSocket实时广播。")
async def pause_strategy_execution(strategy_id: str):
    """
    暂停策略执行
    
    Args:
        strategy_id: 策略ID
    
    Returns:
        dict: 包含暂停结果的字典，包括：
        - success: 是否成功暂停
        - message: 暂停结果消息
        - strategy_id: 策略ID
        - timestamp: 暂停时间戳
    
    Raises:
        HTTPException: 当策略不存在时
    """
    try:
        # 获取业务流程编排器（符合架构设计）
        orchestrator = _get_orchestrator()
        if orchestrator:
            try:
                # 更新策略执行流程状态
                orchestrator.update_process_state(
                    process_type="STRATEGY_EXECUTION",
                    process_id=strategy_id,
                    new_state="PAUSED",
                    data={"strategy_id": strategy_id, "action": "pause"}
                )
                logger.info(f"策略执行流程状态已更新: strategy_id={strategy_id}, state=PAUSED")
            except Exception as e:
                logger.debug(f"使用业务流程编排器失败: {e}")
        
        from .strategy_execution_service import pause_strategy
        success = await pause_strategy(strategy_id)
        if success:
            # 发布执行完成事件（符合架构设计：事件驱动通信）
            event_bus = _get_event_bus()
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.EXECUTION_COMPLETED,
                        {"strategy_id": strategy_id, "action": "pause"},
                        source="strategy_execution_routes"
                    )
                except Exception as e:
                    logger.debug(f"发布执行完成事件失败: {e}")
            
            # WebSocket实时广播（符合架构设计：实时更新）
            manager = _get_websocket_manager()
            if manager:
                try:
                    await manager.broadcast("execution_status", {
                        "type": "execution_status",
                        "action": "paused",
                        "strategy_id": strategy_id,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.debug(f"WebSocket广播失败: {e}")
            
            return {
                "success": True,
                "message": f"策略 {strategy_id} 已暂停",
                "strategy_id": strategy_id,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停策略执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"暂停失败: {str(e)}")


@router.delete("/api/v1/strategy/execution/{strategy_id}", summary="删除策略执行状态", description="从执行状态持久化存储中删除策略。")
async def delete_strategy_execution(strategy_id: str):
    """
    删除策略执行状态
    
    Args:
        strategy_id: 策略ID
        
    Returns:
        dict: 删除结果
    """
    try:
        from .execution_persistence import delete_execution_state
        
        success = delete_execution_state(strategy_id)
        if success:
            logger.info(f"策略 {strategy_id} 执行状态已删除")
            return {
                "success": True,
                "message": f"策略 {strategy_id} 已删除",
                "strategy_id": strategy_id
            }
        else:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除策略执行状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/api/v1/strategy/realtime/metrics", summary="获取实时策略处理指标", description="获取实时策略处理指标，包括流处理指标和每个策略的详细指标。")
async def get_realtime_metrics():
    """
    获取实时策略处理指标
    
    Returns:
        dict: 包含实时处理指标的字典，包括：
        - metrics: 流处理指标
        - stream_metrics: 流处理指标（同metrics）
        - strategies: 策略列表及每个策略的指标
        - history: 历史数据（延迟和吞吐量）
    """
    try:
        from .strategy_execution_service import get_realtime_metrics
        return await get_realtime_metrics()
    except Exception as e:
        logger.error(f"获取实时指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/realtime/signals", summary="获取最近信号", description="获取最近的交易信号，优先从交易信号服务获取，若失败则尝试从执行引擎获取。")
async def get_realtime_signals():
    """
    获取最近信号
    
    Returns:
        dict: 包含最近信号的字典，包括：
        - signals: 信号列表，每个信号包含：
          - id: 信号ID
          - strategy_id: 策略ID
          - strategy_name: 策略名称
          - symbol: 标的符号
          - signal_type: 信号类型（buy/sell/unknown）
          - timestamp: 信号时间戳
          - price: 信号价格
          - strength: 信号强度
          - source: 信号来源
    
    Notes:
        - 最多返回最近20个信号
        - 按时间戳降序排序（最新的在前）
        - 不使用模拟数据，即使为空也返回真实结果
    """
    try:
        # 尝试从交易信号服务获取真实信号
        from .trading_signal_service import get_realtime_signals as get_signals
        
        # 获取所有策略的信号
        all_signals = []
        
        # 首先尝试获取全局信号
        signals = get_signals()
        if signals:
            all_signals.extend(signals)
        
        # 从实时引擎获取策略特定的信号
        try:
            from .strategy_execution_service import get_realtime_engine
            
            engine = await get_realtime_engine()
            if engine and hasattr(engine, 'strategies'):
                for strategy_id, strategy in engine.strategies.items():
                    strategy_name = getattr(strategy, 'name', strategy_id)
                    
                    # 获取策略特定的信号
                    strategy_signals = get_signals(strategy_id=strategy_id, strategy_name=strategy_name)
                    if strategy_signals:
                        all_signals.extend(strategy_signals)
                    
                    # 从策略对象中获取信号
                    object_signals = getattr(strategy, 'signals', [])
                    for signal in object_signals[-10:]:  # 每个策略最多取10个信号
                        if isinstance(signal, dict):
                            all_signals.append({
                                "id": signal.get("id", f"{strategy_id}_{len(all_signals)}"),
                                "strategy_id": strategy_id,
                                "strategy_name": strategy_name,
                                "symbol": signal.get("symbol", getattr(strategy, 'symbol', 'UNKNOWN')),
                                "type": signal.get("type", signal.get("signal_type", "unknown")),
                                "timestamp": signal.get("timestamp", time.time()),
                                "price": signal.get("price", 0),
                                "strength": signal.get("strength", 0),
                                "source": "strategy_object"
                            })
                        elif hasattr(signal, '__dict__'):
                            signal_dict = signal.__dict__
                            all_signals.append({
                                "id": signal_dict.get("id", f"{strategy_id}_{len(all_signals)}"),
                                "strategy_id": strategy_id,
                                "strategy_name": strategy_name,
                                "symbol": signal_dict.get("symbol", getattr(strategy, 'symbol', 'UNKNOWN')),
                                "type": signal_dict.get("type", signal_dict.get("signal_type", "unknown")),
                                "timestamp": signal_dict.get("timestamp", time.time()),
                                "price": signal_dict.get("price", 0),
                                "strength": signal_dict.get("strength", 0),
                                "source": "strategy_object"
                            })
        except Exception as e:
            logger.debug(f"从执行引擎获取信号失败: {e}")
        
        # 去重（基于信号ID）
        seen_ids = set()
        unique_signals = []
        for signal in all_signals:
            signal_id = signal.get("id", "")
            if signal_id and signal_id not in seen_ids:
                seen_ids.add(signal_id)
                unique_signals.append(signal)
        
        # 按时间戳排序（最新的在前）
        unique_signals.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # 格式化信号数据以匹配前端期望的格式
        formatted_signals = []
        for signal in unique_signals[:20]:  # 返回最近20个信号
            formatted_signals.append({
                "id": signal.get("id", ""),
                "strategy_id": signal.get("strategy_id", ""),
                "strategy_name": signal.get("strategy_name", signal.get("strategy_id", "未知策略")),
                "symbol": signal.get("symbol", signal.get("symbol", "UNKNOWN")),
                "signal_type": signal.get("type", signal.get("signal_type", "unknown")),
                "timestamp": signal.get("timestamp", time.time()),
                "price": signal.get("price", 0),
                "strength": signal.get("strength", 0),
                "source": signal.get("source", "unknown")
            })
        
        return {"signals": formatted_signals}
    except Exception as e:
        logger.error(f"获取信号失败: {e}")
        # 量化交易系统要求：不使用模拟数据，即使错误也返回空列表
        return {"signals": []}

