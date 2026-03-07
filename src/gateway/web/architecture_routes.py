"""
架构状态监控API路由
提供21层级架构状态监控API接口
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
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
                logger.debug(f"注册业务流程编排器失败（可选）: {e}")
            
            logger.info("服务容器初始化成功")
        except Exception as e:
            logger.error(f"服务容器初始化失败: {e}")
            return None
    return _container


def _get_event_bus():
    """通过服务容器获取事件总线实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            event_bus = container.resolve("event_bus")
            return event_bus
        except Exception as e:
            logger.debug(f"从容器解析事件总线失败: {e}")
            return None
    return None


def _get_orchestrator():
    """通过服务容器获取业务流程编排器实例（符合架构设计）"""
    container = _get_container()
    if container:
        try:
            orchestrator = container.resolve("business_process_orchestrator")
            return orchestrator
        except Exception as e:
            logger.debug(f"从容器解析业务流程编排器失败（可选）: {e}")
            return None
    return None


def _get_websocket_manager():
    """获取WebSocket管理器实例（符合架构设计）"""
    try:
        from .websocket_manager import manager
        return manager
    except Exception as e:
        logger.debug(f"获取WebSocket管理器失败: {e}")
        return None


@router.get("/api/v1/architecture/status")
async def get_architecture_status() -> Dict[str, Any]:
    """
    获取21层级架构的整体状态
    返回所有层级的状态、健康度、指标等
    """
    try:
        from .architecture_service import get_architecture_overview
        
        # 发布架构状态查询事件
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.SYSTEM_STATUS_CHECKED,
                    {"source": "architecture_routes", "action": "get_architecture_status"},
                    source="architecture_routes"
                )
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")
        
        # 获取架构概览数据
        overview_data = await get_architecture_overview()
        
        # WebSocket广播（如果可用）
        websocket_manager = _get_websocket_manager()
        if websocket_manager:
            try:
                await websocket_manager.broadcast("architecture_status", {
                    "type": "architecture_status",
                    "data": overview_data,
                    "timestamp": int(time.time())
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
        
        return overview_data
    except Exception as e:
        logger.error(f"获取架构状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取架构状态失败: {str(e)}")


@router.get("/api/v1/architecture/layers/{layer_id}/status")
async def get_layer_status_endpoint(layer_id: str) -> Dict[str, Any]:
    """
    获取指定层级的状态
    
    支持的层级ID：
    - 核心业务层（4层）：strategy, trading, risk, features
    - 核心支撑层（4层）：data, ml, infrastructure, streaming
    - 辅助支撑层（9层）：core, monitoring, optimization, gateway, adapter, 
                        automation, resilience, testing, utils
    - 其他层级（4层）：distributed, async, mobile, boundary
    """
    try:
        from .architecture_service import get_layer_status
        
        # 发布层级状态查询事件
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.SYSTEM_STATUS_CHECKED,
                    {"source": "architecture_routes", "action": "get_layer_status", "layer_id": layer_id},
                    source="architecture_routes"
                )
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")
        
        # 获取层级状态
        layer_status = await get_layer_status(layer_id)
        
        # WebSocket广播（如果可用）
        websocket_manager = _get_websocket_manager()
        if websocket_manager:
            try:
                await websocket_manager.broadcast("layer_status", {
                    "type": "layer_status",
                    "layer_id": layer_id,
                    "data": layer_status,
                    "timestamp": int(time.time())
                })
            except Exception as e:
                logger.debug(f"WebSocket广播失败: {e}")
        
        return layer_status
    except Exception as e:
        logger.error(f"获取层级状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取层级状态失败: {str(e)}")


# ==================== 为前端兼容性提供的分层API端点 ====================

@router.get("/api/v1/strategy/status")
async def get_strategy_status() -> Dict[str, Any]:
    """获取策略服务层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("strategy")


@router.get("/api/v1/trading/status")
async def get_trading_status() -> Dict[str, Any]:
    """获取交易执行层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("trading")


@router.get("/api/v1/risk/status")
async def get_risk_status() -> Dict[str, Any]:
    """获取风险控制层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("risk")


@router.get("/api/v1/features/status")
async def get_features_status() -> Dict[str, Any]:
    """获取特征分析层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("features")


@router.get("/api/v1/data/status")
async def get_data_status() -> Dict[str, Any]:
    """获取数据管理层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("data")


@router.get("/api/v1/ml/status")
async def get_ml_status() -> Dict[str, Any]:
    """获取机器学习层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("ml")


@router.get("/api/v1/infrastructure/status")
async def get_infrastructure_status() -> Dict[str, Any]:
    """获取基础设施层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("infrastructure")


@router.get("/api/v1/streaming/status")
async def get_streaming_status() -> Dict[str, Any]:
    """获取流处理层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("streaming")


@router.get("/api/v1/core/status")
async def get_core_status() -> Dict[str, Any]:
    """获取核心服务层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("core")


@router.get("/api/v1/monitoring/status")
async def get_monitoring_status() -> Dict[str, Any]:
    """获取监控层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("monitoring")


@router.get("/api/v1/optimization/status")
async def get_optimization_status() -> Dict[str, Any]:
    """获取优化层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("optimization")


@router.get("/api/v1/gateway/status")
async def get_gateway_status() -> Dict[str, Any]:
    """获取网关层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("gateway")


@router.get("/api/v1/adapter/status")
async def get_adapter_status() -> Dict[str, Any]:
    """获取适配器层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("adapter")


@router.get("/api/v1/automation/status")
async def get_automation_status() -> Dict[str, Any]:
    """获取自动化层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("automation")


@router.get("/api/v1/resilience/status")
async def get_resilience_status() -> Dict[str, Any]:
    """获取弹性层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("resilience")


@router.get("/api/v1/testing/status")
async def get_testing_status() -> Dict[str, Any]:
    """获取测试层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("testing")


@router.get("/api/v1/utils/status")
async def get_utils_status() -> Dict[str, Any]:
    """获取工具层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("utils")


@router.get("/api/v1/distributed/status")
async def get_distributed_status() -> Dict[str, Any]:
    """获取分布式协调器状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("distributed")


@router.get("/api/v1/async/status")
async def get_async_status() -> Dict[str, Any]:
    """获取异步处理器状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("async")


@router.get("/api/v1/mobile/status")
async def get_mobile_status() -> Dict[str, Any]:
    """获取移动端层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("mobile")


@router.get("/api/v1/boundary/status")
async def get_boundary_status() -> Dict[str, Any]:
    """获取业务边界层状态（兼容前端LAYER_CONFIG）"""
    from .architecture_service import get_layer_status
    return await get_layer_status("boundary")

