"""
风险控制API路由
提供风险控制流程监控、概览等API接口
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

# 导入告警服务
try:
    from .risk_alerts_service import get_risk_alerts, get_alert_detail
except ImportError:
    logger.warning("风险告警服务导入失败，详细告警API将不可用")
    get_risk_alerts = None
    get_alert_detail = None

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
                from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
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
    
    # 降级方案：直接创建（业务流程编排器用于管理风险控制流程）
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


@router.get("/api/v1/risk/control/overview")
async def get_risk_control_overview() -> Dict[str, Any]:
    """获取风险控制流程概览数据（符合架构设计：使用业务流程编排器管理风险控制流程）"""
    try:
        # 获取业务流程编排器（符合架构设计）
        orchestrator = _get_orchestrator()
        if orchestrator:
            try:
                # 使用业务流程编排器获取风险控制流程状态
                process_id = orchestrator.start_process(
                    process_type="RISK_CONTROL",
                    initial_data={"action": "get_risk_control_overview"}
                )
                logger.info(f"风险控制流程状态查询已启动: process_id={process_id}")
            except Exception as e:
                logger.debug(f"使用业务流程编排器失败: {e}")
        
        # 发布风险控制流程查询事件（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.RISK_CHECK_STARTED,
                    {"source": "risk_control_routes", "action": "get_risk_control_overview"},
                    source="risk_control_routes"
                )
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")
        
        from .risk_control_persistence import get_latest_risk_control_record
        
        # 从持久化存储获取最新记录
        record = get_latest_risk_control_record()
        
        if record:
            overview_data = {
                "realtime_monitoring": record.get("realtime_monitoring", {}),
                "risk_assessment": record.get("risk_assessment", {}),
                "risk_intercept": record.get("risk_intercept", {}),
                "compliance_check": record.get("compliance_check", {}),
                "risk_report": record.get("risk_report", {}),
                "alert_notify": record.get("alert_notify", {}),
                "timestamp": record.get("timestamp", int(time.time()))
            }
            
            # WebSocket实时广播（符合架构设计：实时更新）
            manager = _get_websocket_manager()
            if manager:
                try:
                    await manager.broadcast("risk_control", {
                        "type": "risk_control_overview",
                        "data": overview_data,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.debug(f"WebSocket广播失败: {e}")
            
            return overview_data
        
        # 如果没有持久化数据，尝试从实时组件获取
        try:
            from .risk_control_service import get_risk_control_overview_data
            overview_data = await get_risk_control_overview_data()
            if overview_data:
                # 保存到持久化存储
                try:
                    from .risk_control_persistence import save_risk_control_record
                    save_risk_control_record({
                        "record_type": "overview_monitor",
                        "timestamp": int(time.time()),
                        **overview_data
                    })
                except Exception as e:
                    logger.debug(f"保存风险控制记录失败: {e}")
                
                # WebSocket实时广播（符合架构设计：实时更新）
                manager = _get_websocket_manager()
                if manager:
                    try:
                        await manager.broadcast("risk_control", {
                            "type": "risk_control_overview",
                            "data": overview_data,
                            "timestamp": time.time()
                        })
                    except Exception as e:
                        logger.debug(f"WebSocket广播失败: {e}")
                
                return overview_data
        except Exception as e:
            logger.debug(f"从实时组件获取概览数据失败: {e}")
        
        # 返回空数据（不使用硬编码值）
        overview_data = {
            "realtime_monitoring": {},
            "risk_assessment": {},
            "risk_intercept": {},
            "compliance_check": {},
            "risk_report": {},
            "alert_notify": {},
            "timestamp": int(time.time()),
            "note": "当前没有可用的风险控制流程数据"
        }
        
        # WebSocket实时广播（符合架构设计：实时更新）
        manager = _get_websocket_manager()
        if manager:
            try:
                await manager.broadcast("risk_control", {
                    "type": "risk_control_overview",
                    "data": overview_data,
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
        
        return overview_data
    except Exception as e:
        logger.error(f"获取风险控制流程概览数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/risk/control/heatmap")
async def get_risk_heatmap() -> Dict[str, Any]:
    """获取风险热力图数据"""
    try:
        from .risk_control_service import get_risk_heatmap_data
        heatmap_data = await get_risk_heatmap_data()
        return heatmap_data
    except Exception as e:
        logger.error(f"获取风险热力图数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/risk/control/timeline")
async def get_risk_timeline() -> Dict[str, Any]:
    """获取风险事件时间线数据"""
    try:
        from .risk_control_service import get_risk_timeline_data
        timeline_data = await get_risk_timeline_data()
        return timeline_data
    except Exception as e:
        logger.error(f"获取风险事件时间线数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/risk/control/alerts")
async def get_risk_alerts() -> Dict[str, Any]:
    """获取风险告警数据"""
    try:
        from .risk_control_service import get_risk_alerts_data
        alerts_data = await get_risk_alerts_data()
        return alerts_data
    except Exception as e:
        logger.error(f"获取风险告警数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/risk/control/stages/{stage_id}")
async def get_risk_control_stage(stage_id: str) -> Dict[str, Any]:
    """获取指定步骤的详细信息（符合架构设计：使用业务流程编排器管理流程状态）"""
    try:
        # 更新业务流程编排器的流程状态（符合架构设计）
        orchestrator = _get_orchestrator()
        if orchestrator:
            try:
                from src.core.orchestration.types import BusinessProcessState
                # 根据步骤ID映射到流程状态
                stage_state_mapping = {
                    "monitoring": BusinessProcessState.MONITORING,
                    "assessment": BusinessProcessState.RISK_ASSESSING,
                    "interception": BusinessProcessState.RISK_INTERCEPTING,
                    "compliance": BusinessProcessState.COMPLIANCE_CHECKING,
                    "report": BusinessProcessState.REPORT_GENERATING,
                    "notification": BusinessProcessState.ALERT_NOTIFYING
                }
                
                state = stage_state_mapping.get(stage_id)
                if state:
                    orchestrator.update_process_state(
                        process_type="RISK_CONTROL",
                        new_state=state,
                        context={"stage_id": stage_id, "action": "get_stage_details"}
                    )
                    logger.info(f"风险控制流程状态已更新: stage={stage_id}, state={state}")
            except Exception as e:
                logger.debug(f"更新业务流程编排器状态失败: {e}")
        
        # 发布步骤查询事件（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.RISK_CHECK_STARTED,
                    {"source": "risk_control_routes", "stage_id": stage_id, "action": "get_stage_details"},
                    source="risk_control_routes"
                )
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")
        
        from .risk_control_service import get_risk_control_stage_data
        stage_data = await get_risk_control_stage_data(stage_id)
        return stage_data
    except Exception as e:
        logger.error(f"获取风险控制步骤数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")

