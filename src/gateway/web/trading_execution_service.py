"""
交易执行服务层
提供交易执行流程数据获取功能
符合架构设计：使用统一适配器工厂访问交易层组件，使用EventBus进行事件通信，使用ServiceContainer进行依赖管理
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
_trading_adapter = None

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

def _get_adapter_factory():
    """获取统一适配器工厂实例（符合架构设计）"""
    global _adapter_factory
    if _adapter_factory is None:
        try:
            from src.core.integration.business_adapters import get_unified_adapter_factory
            _adapter_factory = get_unified_adapter_factory()
            logger.info("统一适配器工厂已获取")
        except Exception as e:
            logger.warning(f"获取统一适配器工厂失败: {e}")
            _adapter_factory = None
    return _adapter_factory

def _get_trading_adapter():
    """获取交易层适配器（符合架构设计：通过统一适配器工厂访问交易层）"""
    global _trading_adapter
    if _trading_adapter is None:
        try:
            from src.core.integration.unified_business_adapters import BusinessLayerType
            factory = _get_adapter_factory()
            if factory:
                _trading_adapter = factory.get_adapter(BusinessLayerType.TRADING)
                if _trading_adapter:
                    logger.info("交易层适配器已获取（通过统一适配器工厂）")
                else:
                    logger.warning("交易层适配器获取失败，将使用降级方案")
            else:
                logger.warning("统一适配器工厂不可用，将使用降级方案")
        except Exception as e:
            logger.warning(f"获取交易层适配器失败: {e}")
            _trading_adapter = None
    return _trading_adapter

def _get_adapter():
    """获取交易层适配器实例（符合架构设计：优先通过统一适配器工厂获取）"""
    # 优先通过统一适配器工厂获取（符合架构设计）
    adapter = _get_trading_adapter()
    if adapter:
        return adapter
    
    # 降级方案：通过服务容器获取（向后兼容）
    container = _get_container()
    if container:
        try:
            adapter = container.resolve("trading_adapter")
            if adapter:
                logger.info("通过服务容器获取交易层适配器（降级方案）")
                return adapter
        except Exception as e:
            logger.debug(f"从容器解析交易层适配器失败: {e}")
    
    # 最终降级方案：直接实例化（符合架构设计：降级处理）
    try:
        from src.core.integration.adapters.trading_adapter import TradingLayerAdapter
        adapter = TradingLayerAdapter()
        logger.info("直接实例化交易层适配器（最终降级方案）")
        return adapter
    except Exception as e:
        logger.debug(f"直接实例化交易层适配器失败: {e}")
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


async def get_execution_flow_data() -> Optional[Dict[str, Any]]:
    """
    获取交易执行流程数据
    从各个组件收集流程监控数据
    符合架构设计：通过TradingLayerAdapter访问交易层组件，发布事件
    """
    try:
        # 发布执行流程数据获取开始事件
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.EXECUTION_STARTED,
                    {"source": "trading_execution_service", "action": "get_flow_data"},
                    source="trading_execution_service"
                )
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")
        
        flow_data = {
            "market_monitoring": {},
            "signal_generation": {},
            "risk_check": {},
            "order_generation": {},
            "order_routing": {},
            "execution": {},
            "position_management": {},
            "result_feedback": {}
        }
        
        # 通过适配器访问交易层组件（符合架构设计）
        adapter = _get_adapter()
        if not adapter:
            logger.warning("交易层适配器不可用，返回空数据")
            return flow_data
        
        # 获取适配器的基础设施桥接器（用于降级服务，符合架构设计）
        infrastructure_bridge = None
        try:
            if hasattr(adapter, '_service_bridges') and adapter._service_bridges:
                infrastructure_bridge = adapter._service_bridges.get('trading_infrastructure_bridge')
        except Exception as e:
            logger.debug(f"获取基础设施桥接器失败: {e}")
        
        # 尝试获取业务流程编排器（用于流程状态管理，符合架构设计）
        orchestrator = _get_orchestrator()
        
        # 定义8个步骤与流程状态的映射关系（符合架构设计：流程状态机）
        step_state_mapping = {
            "market_monitoring": "MONITORING",  # 市场监控对应监控状态
            "signal_generation": "SIGNAL_GENERATING",  # 信号生成
            "risk_check": "RISK_CHECKING",  # 风险检查
            "order_generation": "ORDER_GENERATING",  # 订单生成
            "order_routing": "ORDER_ROUTING",  # 智能路由
            "execution": "EXECUTING",  # 成交执行
            "result_feedback": "MONITORING",  # 结果反馈
            "position_management": "MONITORING"  # 持仓管理
        }
        
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
        
        # 1. 市场监控数据 - 通过适配器获取监控系统（支持降级服务）
        try:
            def get_monitoring_primary():
                return adapter.get_monitoring_system()
            
            def get_monitoring_fallback():
                if infrastructure_bridge:
                    return infrastructure_bridge.get_monitoring()
                return None
            
            monitoring_system = get_with_fallback(
                "获取监控系统",
                get_monitoring_primary,
                get_monitoring_fallback
            )
            
            if monitoring_system:
                # 尝试获取市场监控指标
                # 注意：实际实现需要根据监控系统的API获取真实数据
                flow_data["market_monitoring"] = {
                    "status": "active",
                    "latency": 0,  # TODO: 从监控系统获取真实延迟
                    "quality": 100  # TODO: 从监控系统获取真实质量
                }
            else:
                flow_data["market_monitoring"] = {
                    "status": "数据不可用",
                    "latency": 0,
                    "quality": 0
                }
        except Exception as e:
            logger.debug(f"获取市场监控数据失败: {e}")
        
        # 2. 信号生成数据 - 通过适配器或服务获取
        try:
            from .trading_signal_service import get_signal_stats
            signal_stats = get_signal_stats()
            flow_data["signal_generation"] = {
                "frequency": signal_stats.get("today_signals", 0) / 3600.0 if signal_stats.get("today_signals", 0) > 0 else 0,
                "quality": signal_stats.get("accuracy", 0.0) * 100 if signal_stats.get("accuracy") else 0.0,
                "distribution": "数据不可用"  # 需要从信号服务获取
            }
            # 发布信号生成事件
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.SIGNALS_GENERATED,
                        {"count": signal_stats.get("today_signals", 0)},
                        source="trading_execution_service"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"获取信号生成数据失败: {e}")
        
        # 3. 风险检查数据 - 通过适配器获取风险组件
        try:
            # 适配器可能没有直接的风险管理器接口，暂时返回空
            flow_data["risk_check"] = {
                "latency": 0,
                "intercept_rate": 0,
                "threshold_status": "数据不可用"
            }
            # 发布风险检查完成事件
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.RISK_CHECK_COMPLETED,
                        {"status": "completed"},
                        source="trading_execution_service"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"获取风险检查数据失败: {e}")
        
        # 4. 订单生成数据 - 通过适配器获取订单管理器（支持降级服务）
        try:
            def get_order_manager_primary():
                return adapter.get_order_manager()
            
            order_manager = get_with_fallback(
                "获取订单管理器",
                get_order_manager_primary
            )
            if order_manager:
                # 尝试获取订单生成统计
                flow_data["order_generation"] = {
                    "rate": 0,  # 需要从订单管理器获取
                    "type_distribution": "数据不可用",
                    "status_flow": "数据不可用"
                }
            # 发布订单生成事件
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.ORDERS_GENERATED,
                        {"count": 0},
                        source="trading_execution_service"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"获取订单生成数据失败: {e}")
        
        # 5. 订单路由数据 - 通过服务获取
        try:
            from .order_routing_service import get_routing_stats
            routing_stats = get_routing_stats()
            flow_data["order_routing"] = {
                "latency": routing_stats.get("avg_latency", 0.0),
                "success_rate": routing_stats.get("success_rate", 0.0) * 100 if routing_stats.get("success_rate") else 0.0,
                "path_analysis": "数据不可用"
            }
        except Exception as e:
            logger.debug(f"获取订单路由数据失败: {e}")
        
        # 6. 执行数据 - 通过适配器获取执行引擎（支持降级服务）
        try:
            def get_execution_engine_primary():
                return adapter.get_execution_engine()
            
            execution_engine = get_with_fallback(
                "获取执行引擎",
                get_execution_engine_primary
            )
            if execution_engine:
                flow_data["execution"] = {
                    "success_rate": 0,  # 需要从执行引擎获取
                    "latency": 0,  # 需要从执行引擎获取
                    "slippage": 0  # 需要从执行引擎获取
                }
            # 发布执行完成事件
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.EXECUTION_COMPLETED,
                        {"status": "completed"},
                        source="trading_execution_service"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"获取执行数据失败: {e}")
        
        # 7. 持仓管理数据 - 通过适配器获取投资组合管理器（支持降级服务）
        try:
            def get_portfolio_manager_primary():
                return adapter.get_portfolio_manager()
            
            portfolio_manager = get_with_fallback(
                "获取投资组合管理器",
                get_portfolio_manager_primary
            )
            if portfolio_manager:
                flow_data["position_management"] = {
                    "changes": 0,  # 需要从投资组合管理器获取
                    "risk": "数据不可用",
                    "pnl": 0  # 需要从投资组合管理器获取
                }
            # 发布持仓更新事件
            if event_bus:
                try:
                    from src.core.event_bus.types import EventType
                    event_bus.publish(
                        EventType.POSITION_UPDATED,
                        {"timestamp": int(time.time())},
                        source="trading_execution_service"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"获取持仓管理数据失败: {e}")
        
        # 8. 结果反馈数据
        try:
            flow_data["result_feedback"] = {
                "latency": 0,
                "confirmation_rate": 0,
                "abnormal_orders": 0
            }
        except Exception as e:
            logger.debug(f"获取结果反馈数据失败: {e}")
        
        # 如果业务流程编排器可用，尝试获取流程状态（符合架构设计：使用流程状态机）
        if orchestrator:
            try:
                # 获取当前流程状态（使用状态机）
                current_state = orchestrator.get_current_state()
                if current_state:
                    state_value = current_state.value if hasattr(current_state, 'value') else str(current_state)
                    # 将流程状态信息添加到流程数据中
                    flow_data["process_state"] = {
                        "current_state": state_value,
                        "state_machine_active": True,
                        "state_machine_type": "BusinessProcessStateMachine"
                    }
                    
                    # 根据当前状态，确定8个步骤的执行状态（符合架构设计：流程状态机）
                    # 将每个步骤的状态映射到流程状态机的状态
                    for step_name, mapped_state in step_state_mapping.items():
                        if step_name in flow_data:
                            # 判断当前步骤是否应该处于活跃状态
                            is_active = (
                                state_value == mapped_state or
                                (state_value == "EXECUTING" and step_name in ["execution", "result_feedback", "position_management"]) or
                                (state_value == "MONITORING" and step_name in ["market_monitoring", "result_feedback", "position_management"])
                            )
                            if isinstance(flow_data[step_name], dict):
                                flow_data[step_name]["process_state"] = mapped_state
                                flow_data[step_name]["is_active"] = is_active
                
                # 获取运行中的流程指标
                try:
                    process_metrics = orchestrator.get_process_metrics()
                    if process_metrics:
                        # 将流程指标合并到流程数据中
                        flow_data["process_metrics"] = process_metrics
                        
                        # 获取状态历史（如果可用）
                        try:
                            state_history = orchestrator.state_machine.get_state_history() if hasattr(orchestrator, 'state_machine') else None
                            if state_history:
                                flow_data["process_state"]["state_history"] = [
                                    {
                                        "from_state": h.get("from_state", {}).get("value") if isinstance(h.get("from_state"), dict) else str(h.get("from_state", "")),
                                        "to_state": h.get("to_state", {}).get("value") if isinstance(h.get("to_state"), dict) else str(h.get("to_state", "")),
                                        "timestamp": h.get("timestamp"),
                                        "duration": h.get("duration")
                                    }
                                    for h in state_history[-10:]  # 只保留最近10次状态转换
                                ]
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"从业务流程编排器获取流程状态失败（可选）: {e}")
        
        return flow_data
    except Exception as e:
        logger.error(f"获取执行流程数据失败: {e}")
        return None

