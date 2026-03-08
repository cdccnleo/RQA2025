"""
交易执行API路由
提供交易执行流程监控、概览等API接口
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理，使用BusinessProcessOrchestrator进行业务流程编排
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging
import time

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
            
            # 注册业务流程编排器（符合架构设计：业务流程编排）
            try:
                from src.infrastructure.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
                orchestrator = BusinessProcessOrchestrator()
                orchestrator.initialize()
                _container.register(
                    "business_process_orchestrator",
                    service=orchestrator,
                    lifecycle="singleton"
                )
                logger.info("业务流程编排器已注册到服务容器")
            except Exception as e:
                logger.debug(f"注册业务流程编排器失败（可选功能）: {e}")
            
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
    
    # 降级方案：直接创建（业务流程编排器用于管理交易执行流程）
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


@router.get("/api/v1/trading/execution/flow")
async def get_trading_execution_flow() -> Dict[str, Any]:
    """获取交易执行流程监控数据（符合架构设计：使用业务流程编排器管理交易执行流程）"""
    try:
        # 获取业务流程编排器（符合架构设计）
        orchestrator = _get_orchestrator()
        if orchestrator:
            try:
                # 使用业务流程编排器获取交易执行流程状态
                process_id = orchestrator.start_process(
                    process_type="TRADING_EXECUTION",
                    initial_data={"action": "get_execution_flow"}
                )
                logger.info(f"交易执行流程状态查询已启动: process_id={process_id}")
            except Exception as e:
                logger.debug(f"使用业务流程编排器失败: {e}")
        
        # 发布执行流程查询事件（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.EXECUTION_STARTED,
                    {"source": "trading_execution_routes", "action": "get_execution_flow"},
                    source="trading_execution_routes"
                )
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")
        
        from .trading_execution_persistence import get_latest_execution_record
        
        # 从持久化存储获取最新记录
        record = get_latest_execution_record()
        
        if record:
            flow_data = {
                "market_monitoring": record.get("market_monitoring", {}),
                "signal_generation": record.get("signal_generation", {}),
                "risk_check": record.get("risk_check", {}),
                "order_generation": record.get("order_generation", {}),
                "order_routing": record.get("order_routing", {}),
                "execution": record.get("execution", {}),
                "position_management": record.get("position_management", {}),
                "result_feedback": record.get("result_feedback", {}),
                "timestamp": record.get("timestamp", int(time.time()))
            }
            
            # WebSocket实时广播（符合架构设计：实时更新）
            manager = _get_websocket_manager()
            if manager:
                try:
                    await manager.broadcast("trading_execution", {
                        "type": "execution_flow",
                        "data": flow_data,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.debug(f"WebSocket广播失败: {e}")
            
            return flow_data
        
        # 如果没有持久化数据，尝试从实时组件获取
        try:
            from .trading_execution_service import get_execution_flow_data
            flow_data = await get_execution_flow_data()
            if flow_data:
                # 保存到持久化存储
                try:
                    from .trading_execution_persistence import save_execution_record
                    save_execution_record({
                        "record_type": "flow_monitor",
                        "timestamp": int(time.time()),
                        **flow_data
                    })
                except Exception as e:
                    logger.debug(f"保存执行记录失败: {e}")
                
                # WebSocket实时广播（符合架构设计：实时更新）
                manager = _get_websocket_manager()
                if manager:
                    try:
                        await manager.broadcast("trading_execution", {
                            "type": "execution_flow",
                            "data": flow_data,
                            "timestamp": time.time()
                        })
                    except Exception as e:
                        logger.debug(f"WebSocket广播失败: {e}")
                
                return flow_data
        except Exception as e:
            logger.debug(f"从实时组件获取流程数据失败: {e}")
        
        # 返回空数据（不使用硬编码值）
        flow_data = {
            "market_monitoring": {},
            "signal_generation": {},
            "risk_check": {},
            "order_generation": {},
            "order_routing": {},
            "execution": {},
            "position_management": {},
            "result_feedback": {},
            "timestamp": int(time.time()),
            "note": "当前没有可用的执行流程数据"
        }
        
        # WebSocket实时广播（符合架构设计：实时更新）
        manager = _get_websocket_manager()
        if manager:
            try:
                await manager.broadcast("trading_execution", {
                    "type": "execution_flow",
                    "data": flow_data,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
        
        return flow_data
    except Exception as e:
        logger.error(f"获取交易执行流程数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/trading/overview")
async def get_trading_overview() -> Dict[str, Any]:
    """获取交易概览数据"""
    try:
        # 从多个数据源聚合概览数据
        overview = {
            "today_signals": 0,
            "pending_orders": 0,
            "today_trades": 0,
            "portfolio_value": 0.0,
            "timestamp": int(time.time())
        }
        
        # 获取今日信号数
        try:
            from .trading_signal_service import get_signal_stats
            from datetime import datetime
            stats = get_signal_stats()
            overview["today_signals"] = stats.get("today_signals", 0)
        except Exception as e:
            logger.debug(f"获取信号统计失败: {e}")
        
        # 获取待处理订单数
        try:
            from .order_routing_service import get_routing_decisions
            decisions = get_routing_decisions()
            overview["pending_orders"] = len([d for d in decisions if d.get("status") == "pending"])
        except Exception as e:
            logger.debug(f"获取订单数据失败: {e}")
        
        # 获取今日交易数
        try:
            from .strategy_execution_service import get_execution_metrics
            metrics = await get_execution_metrics()
            overview["today_trades"] = metrics.get("total_trades", 0)
        except Exception as e:
            logger.debug(f"获取执行指标失败: {e}")
        
        # 获取组合价值（需要从持仓管理或策略执行获取）
        try:
            from .strategy_execution_service import get_strategy_execution_status
            status = await get_strategy_execution_status()
            # 这里需要根据实际业务逻辑计算组合价值
            # 暂时返回0，表示数据不可用
            overview["portfolio_value"] = 0.0
        except Exception as e:
            logger.debug(f"获取组合价值失败: {e}")
        
        return overview
    except Exception as e:
        logger.error(f"获取交易概览失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/trading/signals")
async def get_trading_signals(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    获取交易信号列表
    
    支持按策略ID、交易品种、时间范围筛选，支持分页
    """
    try:
        logger.info(f"获取交易信号列表: strategy_id={strategy_id}, symbol={symbol}, limit={limit}, offset={offset}")
        
        signals = []
        total = 0
        
        # 从信号服务获取信号数据
        try:
            from .trading_signal_service import get_signals
            signals, total = get_signals(
                strategy_id=strategy_id,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                offset=offset
            )
        except Exception as e:
            logger.warning(f"从信号服务获取数据失败: {e}")
            # 降级方案：从持久化存储获取
            try:
                from .trading_execution_persistence import query_execution_records
                records = query_execution_records(
                    record_type="signal",
                    strategy_id=strategy_id,
                    limit=limit,
                    offset=offset
                )
                signals = [r.get("data", {}) for r in records]
                total = len(signals)
            except Exception as e2:
                logger.debug(f"从持久化存储获取信号失败: {e2}")
        
        # WebSocket实时广播
        manager = _get_websocket_manager()
        if manager:
            try:
                await manager.broadcast("trading_execution", {
                    "type": "signals_update",
                    "data": {"count": len(signals), "total": total},
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
        
        return {
            "signals": signals,
            "total": total,
            "limit": limit,
            "offset": offset,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取交易信号列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/trading/orders")
async def get_trading_orders(
    status: Optional[str] = None,
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    获取订单列表
    
    支持按状态、策略ID、交易品种筛选，支持分页
    """
    try:
        logger.info(f"获取订单列表: status={status}, strategy_id={strategy_id}, symbol={symbol}")
        
        orders = []
        total = 0
        
        # 从订单服务获取订单数据
        try:
            from .order_routing_service import get_routing_decisions
            decisions = get_routing_decisions()
            
            # 转换路由决策为订单格式
            orders = []
            for decision in decisions:
                order = {
                    "order_id": decision.get("decision_id"),
                    "strategy_id": decision.get("strategy_id"),
                    "symbol": decision.get("symbol"),
                    "side": decision.get("side"),
                    "quantity": decision.get("quantity"),
                    "order_type": decision.get("order_type", "MARKET"),
                    "status": decision.get("status", "pending"),
                    "timestamp": decision.get("timestamp"),
                    "route": decision.get("route")
                }
                
                # 应用筛选条件
                if status and order["status"] != status:
                    continue
                if strategy_id and order["strategy_id"] != strategy_id:
                    continue
                if symbol and order["symbol"] != symbol:
                    continue
                
                orders.append(order)
            
            total = len(orders)
            # 分页
            orders = orders[offset:offset + limit]
            
        except Exception as e:
            logger.warning(f"从订单服务获取数据失败: {e}")
        
        return {
            "orders": orders,
            "total": total,
            "limit": limit,
            "offset": offset,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取订单列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/trading/trades")
async def get_trading_trades(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    获取成交记录
    
    支持按日期范围、交易品种筛选，支持分页
    """
    try:
        logger.info(f"获取成交记录: start_date={start_date}, end_date={end_date}, symbol={symbol}")
        
        trades = []
        total = 0
        
        # 从执行服务获取成交数据
        try:
            from .strategy_execution_service import get_execution_records
            records = await get_execution_records()
            
            # 转换为成交记录格式
            for record in records:
                trade = {
                    "trade_id": record.get("execution_id"),
                    "order_id": record.get("order_id"),
                    "symbol": record.get("symbol"),
                    "side": record.get("side"),
                    "quantity": record.get("quantity"),
                    "price": record.get("price"),
                    "amount": record.get("amount"),
                    "timestamp": record.get("timestamp"),
                    "exchange": record.get("exchange"),
                    "status": record.get("status", "filled")
                }
                
                # 应用筛选条件
                if symbol and trade["symbol"] != symbol:
                    continue
                
                trades.append(trade)
            
            total = len(trades)
            # 分页
            trades = trades[offset:offset + limit]
            
        except Exception as e:
            logger.warning(f"从执行服务获取成交数据失败: {e}")
        
        return {
            "trades": trades,
            "total": total,
            "limit": limit,
            "offset": offset,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取成交记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/trading/positions")
async def get_trading_positions(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取持仓列表
    
    支持按策略ID、交易品种筛选
    """
    try:
        logger.info(f"获取持仓列表: strategy_id={strategy_id}, symbol={symbol}")
        
        positions = []
        total_value = 0.0
        total_pnl = 0.0
        
        # 从持仓服务获取持仓数据
        try:
            from .strategy_execution_service import get_strategy_execution_status
            status_list = await get_strategy_execution_status()
            
            for status in status_list:
                # 提取持仓信息
                position_info = status.get("position", {})
                
                position = {
                    "position_id": status.get("execution_id"),
                    "strategy_id": status.get("strategy_id"),
                    "symbol": position_info.get("symbol"),
                    "quantity": position_info.get("quantity", 0),
                    "avg_price": position_info.get("avg_price", 0.0),
                    "current_price": position_info.get("current_price", 0.0),
                    "market_value": position_info.get("market_value", 0.0),
                    "unrealized_pnl": position_info.get("unrealized_pnl", 0.0),
                    "realized_pnl": position_info.get("realized_pnl", 0.0),
                    "timestamp": status.get("timestamp")
                }
                
                # 应用筛选条件
                if strategy_id and position["strategy_id"] != strategy_id:
                    continue
                if symbol and position["symbol"] != symbol:
                    continue
                
                positions.append(position)
                total_value += position["market_value"]
                total_pnl += position["unrealized_pnl"]
            
        except Exception as e:
            logger.warning(f"从持仓服务获取数据失败: {e}")
        
        return {
            "positions": positions,
            "total_positions": len(positions),
            "total_value": total_value,
            "total_pnl": total_pnl,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取持仓列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")

