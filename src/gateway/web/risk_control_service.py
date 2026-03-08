"""
风险控制服务层
提供风险控制流程数据获取功能
符合架构设计：使用统一适配器工厂访问风险控制层组件，使用EventBus进行事件通信，使用ServiceContainer进行依赖管理
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import time

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# 全局服务容器（延迟初始化，符合架构设计）
_container = None

# 全局适配器工厂实例（符合架构设计：统一适配器工厂）
_adapter_factory = None
_risk_adapter = None

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
                logger.debug(f"注册业务流程编排器失败（可选）: {e}")
            
            logger.info("服务容器初始化成功")
        except Exception as e:
            logger.error(f"服务容器初始化失败: {e}")
            return None
    return _container

def _get_adapter_factory():
    """获取统一适配器工厂实例（符合架构设计）"""
    global _adapter_factory
    if _adapter_factory is None:
        try:
            from src.infrastructure.integration.business_adapters import get_unified_adapter_factory
            _adapter_factory = get_unified_adapter_factory()
            logger.info("统一适配器工厂已获取")
        except Exception as e:
            logger.warning(f"获取统一适配器工厂失败: {e}")
            _adapter_factory = None
    return _adapter_factory

def _get_risk_adapter():
    """获取风险控制层适配器（符合架构设计：通过统一适配器工厂访问风险控制层）"""
    global _risk_adapter
    if _risk_adapter is None:
        try:
            from src.infrastructure.integration.unified_business_adapters import BusinessLayerType
            factory = _get_adapter_factory()
            if factory:
                _risk_adapter = factory.get_adapter(BusinessLayerType.RISK)
                if _risk_adapter:
                    logger.info("风险控制层适配器已获取（通过统一适配器工厂）")
                else:
                    logger.warning("风险控制层适配器获取失败，将使用降级方案")
            else:
                logger.warning("统一适配器工厂不可用，将使用降级方案")
        except Exception as e:
            logger.warning(f"获取风险控制层适配器失败: {e}")
            _risk_adapter = None
    return _risk_adapter

def _get_adapter():
    """获取风险控制层适配器实例（符合架构设计：优先通过统一适配器工厂获取）"""
    # 优先通过统一适配器工厂获取（符合架构设计）
    adapter = _get_risk_adapter()
    if adapter:
        return adapter
    
    # 降级方案：通过服务容器获取（向后兼容）
    container = _get_container()
    if container:
        try:
            adapter = container.resolve("risk_adapter")
            if adapter:
                logger.info("通过服务容器获取风险控制层适配器（降级方案）")
                return adapter
        except Exception as e:
            logger.debug(f"从容器解析风险控制层适配器失败: {e}")
    
    # 最终降级方案：直接实例化（符合架构设计：降级处理）
    try:
        from src.infrastructure.integration.adapters.risk_adapter import RiskLayerAdapter
        adapter = RiskLayerAdapter()
        logger.info("直接实例化风险控制层适配器（最终降级方案）")
        return adapter
    except Exception as e:
        logger.debug(f"直接实例化风险控制层适配器失败: {e}")
        return None


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


# 定义6个步骤与流程状态的映射关系（符合架构设计：流程状态机）
step_state_mapping = {
    "realtime_monitoring": "MONITORING",  # 实时监测对应监控状态
    "risk_assessment": "RISK_CHECKING",  # 风险评估对应风险检查状态
    "risk_intercept": "RISK_CHECKING",  # 风险拦截对应风险检查状态
    "compliance_check": "RISK_CHECKING",  # 合规检查对应风险检查状态
    "risk_report": "MONITORING",  # 风险报告对应监控状态
    "alert_notify": "MONITORING"  # 告警通知对应监控状态
}


async def get_risk_control_overview_data() -> Optional[Dict[str, Any]]:
    """
    获取风险控制流程概览数据
    从各个组件收集流程监控数据
    符合架构设计：通过RiskLayerAdapter访问风险控制层组件，发布事件
    """
    try:
        # 发布风险控制流程数据获取开始事件
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.RISK_CHECK_STARTED,
                    {"source": "risk_control_service", "action": "get_overview_data"},
                    source="risk_control_service"
                )
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")
        
        overview_data = {
            "realtime_monitoring": {},
            "risk_assessment": {},
            "risk_intercept": {},
            "compliance_check": {},
            "risk_report": {},
            "alert_notify": {}
        }
        
        # 通过适配器访问风险控制层组件（符合架构设计）
        adapter = _get_adapter()
        if not adapter:
            logger.warning("风险控制层适配器不可用，返回空数据")
            return overview_data
        
        # 获取适配器的基础设施桥接器（用于降级服务，符合架构设计）
        infrastructure_bridge = None
        try:
            if hasattr(adapter, '_service_bridges') and adapter._service_bridges:
                infrastructure_bridge = adapter._service_bridges.get('risk_infrastructure_bridge')
        except Exception as e:
            logger.debug(f"获取基础设施桥接器失败: {e}")
        
        # 尝试获取业务流程编排器（用于流程状态管理，符合架构设计）
        orchestrator = _get_orchestrator()
        
        # 辅助函数：使用降级服务机制获取数据（符合架构设计：降级服务）
        def get_with_fallback(operation_name: str, primary_func, fallback_func=None):
            """使用降级服务机制执行操作"""
            if infrastructure_bridge and hasattr(infrastructure_bridge, 'execute_with_fallback'):
                try:
                    return infrastructure_bridge.execute_with_fallback(
                        operation_name,
                        primary_func,
                        fallback_func
                    )
                except Exception as e:
                    logger.debug(f"降级服务执行失败 {operation_name}: {e}")
            # 如果没有降级机制，直接执行主函数
            try:
                return primary_func()
            except Exception as e:
                logger.debug(f"主服务执行失败 {operation_name}: {e}")
                if fallback_func:
                    try:
                        return fallback_func()
                    except Exception:
                        pass
            return None
        
        # 1. 实时监测数据 - 通过适配器获取风险监控器（支持降级服务）
        try:
            def get_monitor_primary():
                return adapter.get_risk_monitor()
            
            risk_monitor = get_with_fallback(
                "获取风险监控器",
                get_monitor_primary
            )
            
            if risk_monitor:
                overview_data["realtime_monitoring"] = {
                    "status": "active",
                    "coverage": 100,  # TODO: 从监控器获取真实覆盖率
                    "latency": 0,  # TODO: 从监控器获取真实延迟
                    "alerts_count": 0  # TODO: 从监控器获取真实告警数
                }
            else:
                overview_data["realtime_monitoring"] = {
                    "status": "数据不可用",
                    "coverage": 0,
                    "latency": 0,
                    "alerts_count": 0
                }
        except Exception as e:
            logger.debug(f"获取实时监测数据失败: {e}")
        
        # 2. 风险评估数据 - 通过适配器获取风险计算器
        try:
            def get_calculator_primary():
                return adapter.get_risk_calculator()
            
            risk_calculator = get_with_fallback(
                "获取风险计算器",
                get_calculator_primary
            )
            
            if risk_calculator:
                overview_data["risk_assessment"] = {
                    "latency": 0,  # TODO: 从计算器获取真实延迟
                    "risk_level": "低",  # TODO: 从计算器获取真实风险等级
                    "accuracy": 100  # TODO: 从计算器获取真实准确性
                }
                # 发布风险评估完成事件
                if event_bus:
                    try:
                        from src.core.event_bus.types import EventType
                        event_bus.publish(
                            EventType.RISK_ASSESSMENT_COMPLETED,
                            {"risk_level": overview_data["risk_assessment"].get("risk_level", "未知")},
                            source="risk_control_service"
                        )
                    except Exception:
                        pass
            else:
                overview_data["risk_assessment"] = {
                    "latency": 0,
                    "risk_level": "数据不可用",
                    "accuracy": 0
                }
        except Exception as e:
            logger.debug(f"获取风险评估数据失败: {e}")
        
        # 3. 风险拦截数据
        try:
            def get_manager_primary():
                return adapter.get_risk_manager()
            
            risk_manager = get_with_fallback(
                "获取风险管理器",
                get_manager_primary
            )
            
            if risk_manager:
                overview_data["risk_intercept"] = {
                    "intercept_rate": 0,  # TODO: 从管理器获取真实拦截率
                    "latency": 0,  # TODO: 从管理器获取真实延迟
                    "intercept_type": "数据不可用"  # TODO: 从管理器获取真实拦截类型
                }
                # 发布风险拦截事件
                if event_bus:
                    try:
                        from src.core.event_bus.types import EventType
                        event_bus.publish(
                            EventType.RISK_INTERCEPTED,
                            {"intercept_rate": overview_data["risk_intercept"].get("intercept_rate", 0)},
                            source="risk_control_service"
                        )
                    except Exception:
                        pass
            else:
                overview_data["risk_intercept"] = {
                    "intercept_rate": 0,
                    "latency": 0,
                    "intercept_type": "数据不可用"
                }
        except Exception as e:
            logger.debug(f"获取风险拦截数据失败: {e}")
        
        # 4. 合规检查数据
        try:
            overview_data["compliance_check"] = {
                "latency": 0,
                "compliance_rate": 100,  # TODO: 从合规检查器获取真实合规率
                "violations_count": 0  # TODO: 从合规检查器获取真实违规数
            }
            # 发布合规检查完成事件
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.COMPLIANCE_CHECK_COMPLETED,
                        {"compliance_rate": overview_data["compliance_check"].get("compliance_rate", 100)},
                        source="risk_control_service"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"获取合规检查数据失败: {e}")
        
        # 5. 风险报告数据
        try:
            overview_data["risk_report"] = {
                "report_generation_time": 0,  # TODO: 从报告生成器获取真实生成时间
                "report_count": 0,  # TODO: 从报告生成器获取真实报告数
                "report_type_distribution": "数据不可用"  # TODO: 从报告生成器获取真实类型分布
            }
            # 发布风险报告生成事件
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.RISK_REPORT_GENERATED,
                        {"report_count": overview_data["risk_report"].get("report_count", 0)},
                        source="risk_control_service"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"获取风险报告数据失败: {e}")
        
        # 6. 告警通知数据 - 通过适配器获取告警系统
        try:
            def get_alert_system_primary():
                return adapter.get_alert_system()
            
            alert_system = get_with_fallback(
                "获取告警系统",
                get_alert_system_primary
            )
            
            if alert_system:
                overview_data["alert_notify"] = {
                    "alert_count": 0,  # TODO: 从告警系统获取真实告警数
                    "response_time": 0,  # TODO: 从告警系统获取真实响应时间
                    "resolution_rate": 100  # TODO: 从告警系统获取真实处理率
                }
                # 发布告警触发事件
                if event_bus:
                    try:
                        from src.core.event_bus.types import EventType
                        event_bus.publish(
                            EventType.ALERT_TRIGGERED,
                            {"alert_count": overview_data["alert_notify"].get("alert_count", 0)},
                            source="risk_control_service"
                        )
                    except Exception:
                        pass
            else:
                overview_data["alert_notify"] = {
                    "alert_count": 0,
                    "response_time": 0,
                    "resolution_rate": 0
                }
        except Exception as e:
            logger.debug(f"获取告警通知数据失败: {e}")
        
        # 如果业务流程编排器可用，尝试获取流程状态（符合架构设计：使用流程状态机）
        if orchestrator:
            try:
                # 获取当前流程状态（使用状态机）
                current_state = orchestrator.get_current_state()
                if current_state:
                    state_value = current_state.value if hasattr(current_state, 'value') else str(current_state)
                    # 将流程状态信息添加到概览数据中
                    overview_data["process_state"] = {
                        "current_state": state_value,
                        "state_machine_active": True,
                        "state_machine_type": "BusinessProcessStateMachine"
                    }
                    
                    # 根据当前状态，确定6个步骤的执行状态（符合架构设计：流程状态机）
                    for step_name, mapped_state in step_state_mapping.items():
                        if step_name in overview_data:
                            # 判断当前步骤是否应该处于活跃状态
                            is_active = (
                                state_value == mapped_state or
                                (state_value == "MONITORING" and step_name in ["realtime_monitoring", "risk_report", "alert_notify"]) or
                                (state_value == "RISK_CHECKING" and step_name in ["risk_assessment", "risk_intercept", "compliance_check"])
                            )
                            if isinstance(overview_data[step_name], dict):
                                overview_data[step_name]["process_state"] = mapped_state
                                overview_data[step_name]["is_active"] = is_active
            except Exception as e:
                logger.debug(f"从业务流程编排器获取流程状态失败（可选）: {e}")
        
        return overview_data
    except Exception as e:
        logger.error(f"获取风险控制流程概览数据失败: {e}")
        return None


async def get_risk_heatmap_data() -> Dict[str, Any]:
    """获取风险热力图数据"""
    try:
        return {
            "heatmap": [],
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取风险热力图数据失败: {e}")
        return {"heatmap": [], "timestamp": int(time.time())}


async def get_risk_timeline_data() -> Dict[str, Any]:
    """获取风险事件时间线数据"""
    try:
        return {
            "timeline": [],
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取风险事件时间线数据失败: {e}")
        return {"timeline": [], "timestamp": int(time.time())}


async def get_risk_alerts_data() -> Dict[str, Any]:
    """获取风险告警数据"""
    try:
        adapter = _get_adapter()
        alert_system = None
        if adapter:
            try:
                alert_system = adapter.get_alert_system()
            except Exception as e:
                logger.debug(f"获取告警系统失败: {e}")
        
        return {
            "alerts": [],
            "alert_count": 0,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取风险告警数据失败: {e}")
        return {"alerts": [], "alert_count": 0, "timestamp": int(time.time())}


async def get_risk_control_stage_data(stage_id: str) -> Dict[str, Any]:
    """获取指定步骤的详细信息"""
    try:
        overview_data = await get_risk_control_overview_data()
        if not overview_data:
            return {"stage_id": stage_id, "data": {}, "timestamp": int(time.time())}
        
        # 根据stage_id映射到对应的步骤数据
        stage_mapping = {
            "monitoring": "realtime_monitoring",
            "assessment": "risk_assessment",
            "interception": "risk_intercept",
            "compliance": "compliance_check",
            "report": "risk_report",
            "notification": "alert_notify"
        }
        
        step_key = stage_mapping.get(stage_id, "realtime_monitoring")
        stage_data = overview_data.get(step_key, {})
        
        return {
            "stage_id": stage_id,
            "data": stage_data,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取风险控制步骤数据失败: {e}")
        return {"stage_id": stage_id, "data": {}, "timestamp": int(time.time())}
