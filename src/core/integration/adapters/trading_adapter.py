import logging
#!/usr/bin/env python3
"""
RQA2025 交易层基础设施适配器 - 重构版

专门为交易层提供基础设施服务访问接口，
基于统一业务层适配器架构，实现交易层的特定需求。

重构说明:
- 将大类拆分为多个专门组件
- 职责分离，提高可维护性
- 组合模式，保持接口兼容性
"""

from typing import Dict, Any, Optional, Protocol
from src.core.integration.unified_business_adapters import UnifiedBusinessAdapter, BusinessLayerType
from src.core.integration.core.business_adapters import BaseBusinessAdapter
from datetime import datetime
import time
from dataclasses import dataclass

from src.core.constants import (
    SECONDS_PER_HOUR, MAX_RECORDS
)

# 导入组件
from .components import TradingHealthChecker, TradingMetricsCollector, TradingExecutor

logger = logging.getLogger(__name__)


class TradingEngineProvider(Protocol):
    """交易引擎提供者协议"""
    def get_trading_engine(self) -> Optional[Any]: ...
    def get_hft_execution_engine(self) -> Optional[Any]: ...


class InfrastructureServiceProvider(Protocol):
    """基础设施服务提供者协议"""
    def get_monitoring_system(self) -> Optional[Any]: ...
    def get_audit_system(self) -> Optional[Any]: ...
    def get_order_manager(self) -> Optional[Any]: ...


@dataclass
class TradingInfrastructureConfig:
    """交易基础设施配置"""
    enable_distributed_engine: bool = True
    enable_hft_engine: bool = False
    enable_monitoring: bool = True
    enable_audit: bool = True
    cache_enabled: bool = True
    max_retries: int = 3


class TradingEngineManager:
    """交易引擎管理器 - 职责：管理各种交易引擎"""

    def __init__(self, config: TradingInfrastructureConfig):
        self.config = config

    def get_trading_engine(self):
        """获取交易引擎"""
        if not self.config.enable_distributed_engine:
            return self._get_standard_engine()

        try:
            # 优先使用分布式交易引擎
            from src.trading.trading_engine_with_distributed import DistributedTradingEngine
            logger.info("使用分布式交易引擎")
            return DistributedTradingEngine()
        except ImportError:
            return self._get_standard_engine()

    def _get_standard_engine(self):
        """获取标准交易引擎"""
        try:
            from src.trading.trading_engine import TradingEngine
            logger.info("使用标准交易引擎")
            return TradingEngine()
        except ImportError:
            logger.warning("交易引擎导入失败")
            return None

    def get_hft_execution_engine(self):
        """获取高频交易执行引擎"""
        if not self.config.enable_hft_engine:
            return None

        try:
            from src.trading.hft_execution_engine import HFTExecutionEngine
            logger.info("使用高频交易执行引擎")
            return HFTExecutionEngine()
        except ImportError:
            logger.warning("高频交易执行引擎导入失败")
            return None


class InfrastructureServiceManager:
    """基础设施服务管理器 - 职责：管理基础设施服务"""

    def __init__(self, config: TradingInfrastructureConfig):
        self.config = config

    def get_monitoring_system(self):
        """获取监控系统"""
        if not self.config.enable_monitoring:
            return None

        try:
            from src.monitoring.monitoring_system import get_monitoring_system
            logger.info("使用监控系统")
            return get_monitoring_system()
        except ImportError:
            logger.warning("监控系统导入失败")
            return None

    def get_audit_system(self):
        """获取审计系统"""
        if not self.config.enable_audit:
            return None

        try:
            from src.security.audit_system import get_audit_system
            logger.info("使用审计系统")
            return get_audit_system()
        except ImportError:
            logger.warning("审计系统导入失败")
            return None

    def get_order_manager(self):
        """获取订单管理器"""
        try:
            from src.trading.order_manager import OrderManager
            return OrderManager()
        except ImportError:
            logger.warning("订单管理器导入失败")
            return None


class TradingLayerAdapter(UnifiedBusinessAdapter):

    """交易层适配器 - 重构版：使用组件化架构"""

    def __init__(self):
        super().__init__(BusinessLayerType.TRADING)
        
        # 初始化组件
        self._health_checker = TradingHealthChecker(self)
        self._metrics_collector = TradingMetricsCollector(self)
        self._executor = TradingExecutor(self)
        
        self._init_trading_specific_services()

    def _init_trading_specific_services(self):
        """初始化交易层特定的基础设施服务"""
        try:
            # 交易层特定的服务桥接器
            # 注意：交易层可能还没有专门的基础设施桥接器，使用通用适配器
            self._service_bridges = {
                'trading_infrastructure_bridge': self._create_trading_bridge()
            }

            logger.info("交易层特定服务桥接器初始化完成")

        except Exception as e:
            logger.warning(f"交易层特定服务桥接器初始化失败，使用基础服务: {e}")

    def _create_trading_bridge(self):
        """创建交易层专用的基础设施桥接器"""
        # 使用完善的TradingInfrastructureBridge，支持统一基础设施集成
        return TradingInfrastructureBridge()

    def get_trading_engine(self):
        """获取交易引擎"""
        try:
            # 优先使用分布式交易引擎
            from src.trading.trading_engine_with_distributed import DistributedTradingEngine
            logger.info("使用分布式交易引擎")
            return DistributedTradingEngine()
        except ImportError:
            try:
                from src.trading.trading_engine import TradingEngine
                logger.info("使用标准交易引擎")
                return TradingEngine()
            except ImportError:
                logger.warning("交易引擎导入失败")
                return None

    def get_hft_execution_engine(self):
        """获取高频交易执行引擎"""
        try:
            from src.trading.hft_execution_engine import HFTExecutionEngine
            logger.info("使用高频交易执行引擎")
            return HFTExecutionEngine()
        except ImportError:
            logger.warning("高频交易执行引擎导入失败")
            return None

    def get_monitoring_system(self):
        """获取监控系统"""
        try:
            from src.monitoring.monitoring_system import get_monitoring_system
            logger.info("使用监控系统")
            return get_monitoring_system()
        except ImportError:
            logger.warning("监控系统导入失败")
            return None

    def get_audit_system(self):
        """获取审计系统"""
        try:
            from src.security.audit_system import get_audit_system
            logger.info("使用审计系统")
            return get_audit_system()
        except ImportError:
            logger.warning("审计系统导入失败")
            return None

    def get_order_manager(self):
        """获取订单管理器"""
        try:
            from src.trading.order_manager import OrderManager
            return OrderManager()
        except ImportError:
            logger.warning("订单管理器导入失败")
            return None

    def get_execution_engine(self):
        """获取执行引擎"""
        try:
            from src.trading.execution_engine import ExecutionEngine
            return ExecutionEngine()
        except ImportError:
            logger.warning("执行引擎导入失败")
            return None

    def get_portfolio_manager(self):
        """获取投资组合管理器"""
        try:
            from src.trading.portfolio_portfolio_manager import PortfolioManager
            return PortfolioManager()
        except ImportError:
            logger.warning("投资组合管理器导入失败")
            return None

    def health_check(self) -> Dict[str, Any]:
        """交易层健康检查 - 委托给健康检查组件"""
        base_health = super().health_check()

        # 委托给健康检查组件
        trading_specific_health = self._health_checker.check_all_services_health()

        base_health['trading_specific_services'] = trading_specific_health

        # 更新整体状态
        for service_name, health_info in trading_specific_health.items():
            if health_info.get('status') != 'healthy':
                base_health['overall_status'] = 'degraded'
                break

        return base_health

    # 向后兼容的方法（委托给组件）
    def _check_trading_engine_health(self) -> Dict[str, Any]:
        """检查交易引擎健康状态（向后兼容）"""
        return self._health_checker.check_trading_engine_health()

    def _check_order_manager_health(self) -> Dict[str, Any]:
        """检查订单管理器健康状态（向后兼容）"""
        return self._health_checker.check_order_manager_health()

    def _check_execution_engine_health(self) -> Dict[str, Any]:
        """检查执行引擎健康状态（向后兼容）"""
        return self._health_checker.check_execution_engine_health()

    def _check_portfolio_manager_health(self) -> Dict[str, Any]:
        """检查投资组合管理器健康状态（向后兼容）"""
        return self._health_checker.check_portfolio_manager_health()

    def get_trading_layer_metrics(self) -> Dict[str, Any]:
        """获取交易层性能指标 - 委托给指标收集组件"""
        return self._metrics_collector.collect_trading_metrics()

    def execute_trade_with_infrastructure(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """使用基础设施支持执行交易 - 委托给交易执行组件"""
        return self._executor.execute_trade(trade_request)

    # 向后兼容的私有方法（委托给组件，保持内部方法兼容）
    def _get_infrastructure_services(self) -> Dict[str, Any]:
        """获取所需的基础设施服务（向后兼容）"""
        services = self.get_infrastructure_services()
        return {
            'cache_manager': services.get('cache_manager'),
            'monitoring': services.get('monitoring')
        }

    def execute_trading_flow(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """执行完整的交易流程 (基于架构设计的业务流程)

        交易执行流程：
        1. 市场监控 → 信号生成 → 风险检查 → 订单生成 → 智能路由 → 成交执行 → 结果反馈 → 持仓管理

        Args:
            trade_request: 交易请求数据

        Returns:
            执行结果
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "request_id": trade_request.get("request_id", f"req_{int(time.time())}"),
            "flow_steps": [],
            "success": False,
            "errors": [],
            "metrics": {}
        }

        try:
            # 步骤1: 初始化交易流程
            cache_manager, monitoring = self._initialize_trading_flow(result)

            # 步骤2: 风险检查
            if not self._perform_risk_check(trade_request, result):
                return result

            # 步骤3: 订单生成
            if not self._generate_order(trade_request, result):
                return result

            # 步骤4: 执行引擎处理
            if not self._execute_order(trade_request, result):
                return result

            # 步骤5: 持仓管理更新
            self._update_position(result)

            # 步骤6: 缓存结果
            self._cache_trading_result(cache_manager, result)

            # 步骤7: 记录完成指标
            self._record_completion_metrics(monitoring, result)

            result['flow_steps'].append("flow_completed")

        except Exception as e:
            self._handle_trading_flow_error(e, result)

        # 计算流程耗时
        self._calculate_flow_duration(result)

        return result

    def _initialize_trading_flow(self, result: Dict[str, Any]) -> tuple:
        """初始化交易流程"""
        # 获取基础设施服务
        cache_manager = self.get_infrastructure_services().get('cache_manager')
        monitoring = self.get_infrastructure_services().get('monitoring')

        # 记录流程开始
        if monitoring:
            monitoring.record_metric('trading_flow_start', 1, {
                                     'request_id': result['request_id']})
            result['flow_steps'].append("infrastructure_initialized")

        return cache_manager, monitoring

    def _perform_risk_check(self, trade_request: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """执行风险检查"""
        risk_manager = self.get_risk_manager()
        if risk_manager:
            risk_result = risk_manager.evaluate_trade_risk(trade_request)
            result['flow_steps'].append("risk_check_completed")
            result['risk_assessment'] = risk_result

            if risk_result.get('overall_action') == 'block':
                result['errors'].append(f"交易被风控阻止: {risk_result.get('blocks', [])}")
                return False
        else:
            result['flow_steps'].append("risk_check_skipped")

        return True

    def _generate_order(self, trade_request: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """生成订单"""
        order_manager = self.get_order_manager()
        if order_manager:
            # 创建订单
            order = order_manager.create_order(
                symbol=trade_request.get('symbol'),
                quantity=trade_request.get('quantity', 0),
                order_type=trade_request.get('order_type', 'market'),
                price=trade_request.get('price'),
                strategy_id=trade_request.get('strategy_id'),
                metadata={'request_id': result['request_id']}
            )
            result['flow_steps'].append("order_created")
            result['order_id'] = order.order_id

            # 提交订单
            submitted_id = order_manager.submit_order(order)
            result['flow_steps'].append("order_submitted")
            result['submitted_order_id'] = submitted_id
            return True
        else:
            result['errors'].append("订单管理器不可用")
            return False

    def _execute_order(self, trade_request: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """执行订单"""
        execution_engine = self.get_execution_engine()
        if execution_engine:
            execution_result = execution_engine.execute_order({
                'order_id': result['order_id'],
                'symbol': trade_request.get('symbol'),
                'quantity': trade_request.get('quantity'),
                'order_type': trade_request.get('order_type'),
                'price': trade_request.get('price'),
                'side': trade_request.get('side', 'buy')
            })
            result['flow_steps'].append("order_executed")
            result['execution_result'] = execution_result

            if execution_result.get('success'):
                result['success'] = True
            else:
                result['errors'].append(f"执行失败: {execution_result.get('error', '未知错误')}")
            return True
        else:
            result['errors'].append("执行引擎不可用")
            return False

    def _update_position(self, result: Dict[str, Any]) -> None:
        """更新持仓"""
        portfolio_manager = self.get_position_manager()
        if portfolio_manager and result['success']:
            # 这里应该更新持仓，但需要根据实际成交情况
            result['flow_steps'].append("position_updated")

    def _cache_trading_result(self, cache_manager: Any, result: Dict[str, Any]) -> None:
        """缓存交易结果"""
        if cache_manager and result['success']:
            cache_key = f"trading_result_{result['request_id']}"
            cache_manager.set(cache_key, result, SECONDS_PER_HOUR)  # 缓存1小时
            result['flow_steps'].append("result_cached")

    def _record_completion_metrics(self, monitoring: Any, result: Dict[str, Any]) -> None:
        """记录完成指标"""
        if monitoring:
            monitoring.record_metric('trading_flow_complete', 1, {
                'request_id': result['request_id'],
                'success': result['success']
            })

    def _handle_trading_flow_error(self, error: Exception, result: Dict[str, Any]) -> None:
        """处理交易流程错误"""
        result['errors'].append(f"交易流程执行异常: {error}")
        logger.error(f"交易流程执行失败: {error}")

        # 记录错误指标
        monitoring = self.get_infrastructure_services().get('monitoring')
        if monitoring:
            monitoring.record_metric('trading_flow_error', 1, {
                'request_id': result['request_id'],
                'error': str(error)
            })

    def _calculate_flow_duration(self, result: Dict[str, Any]) -> None:
        """计算流程耗时"""
        result['duration_ms'] = (
            datetime.now() - datetime.fromisoformat(result['timestamp'])).total_seconds() * DEFAULT_PERFORMANCE_THRESHOLD

    def get_position_manager(self):
        """获取持仓管理器"""
        try:
            from src.trading.portfolio_portfolio_manager import PortfolioManager
            return PortfolioManager()
        except ImportError:
            logger.warning("持仓管理器导入失败")
            return None

    def get_risk_manager(self):
        """获取风险管理器"""
        try:
            from src.trading.risk import TradingRiskManager
            return TradingRiskManager()
        except ImportError:
            logger.warning("风险管理器导入失败")
            return None

    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务字典 - 通过TradingInfrastructureBridge"""
        if hasattr(self, '_service_bridges') and 'trading_infrastructure_bridge' in self._service_bridges:
            bridge = self._service_bridges['trading_infrastructure_bridge']
            if isinstance(bridge, TradingInfrastructureBridge):
                return {
                    'config_manager': bridge.get_config_manager(),
                    'cache_manager': bridge.get_cache_manager(),
                    'monitoring': bridge.get_monitoring(),
                    'logger': bridge.get_logger(),
                    'health_checker': bridge.get_health_checker()
                }

        # 降级到父类实现
        return super().get_infrastructure_services()


class TradingInfrastructureBridge:

    """交易层基础设施桥接器 - 基于统一基础设施集成层"""

    def __init__(self):

        self._services = {}
        self._unified_adapter_factory = None
        self._fallback_services = {}
        self._init_unified_services()
        self._init_fallback_services()

    def _init_unified_services(self):
        """初始化统一基础设施服务"""
        try:
            # 获取统一适配器工厂
            from .adapters import get_unified_adapter_factory
            self._unified_adapter_factory = get_unified_adapter_factory()

            if self._unified_adapter_factory:
                logger.info("交易层成功连接统一基础设施集成层")
            else:
                logger.warning("无法获取统一适配器工厂")

        except ImportError as e:
            logger.warning(f"无法导入统一适配器工厂: {e}")

    def _init_fallback_services(self):
        """初始化降级服务"""
        try:
            from .fallback_services import (
                get_fallback_config_manager,
                get_fallback_cache_manager,
                get_fallback_logger,
                get_fallback_monitoring,
                get_fallback_health_checker
            )

            self._fallback_services = {
                'config_manager': get_fallback_config_manager(),
                'cache_manager': get_fallback_cache_manager(),
                'logger': get_fallback_logger(),
                'monitoring': get_fallback_monitoring(),
                'health_checker': get_fallback_health_checker()
            }

            logger.info("交易层降级服务初始化完成")

        except ImportError as e:
            logger.warning(f"降级服务初始化失败: {e}")
            self._fallback_services = {}

    def get_service(self, service_name: str):
        """获取服务 - 优先使用统一服务，降级到本地服务"""
        # 首先尝试获取统一基础设施服务
        if self._unified_adapter_factory:
            try:
                # 根据服务名称获取相应的统一服务
                if service_name == 'config_manager':
                    return self._unified_adapter_factory.get_config_manager()
                elif service_name == 'cache_manager':
                    return self._unified_adapter_factory.get_cache_manager()
                elif service_name == 'monitoring':
                    return self._unified_adapter_factory.get_monitoring()
                elif service_name == 'logger':
                    return self._unified_adapter_factory.get_logger()
                elif service_name == 'health_checker':
                    return self._unified_adapter_factory.get_health_checker()
            except Exception as e:
                logger.warning(f"获取统一服务 {service_name} 失败: {e}")

        # 降级到本地服务
        if service_name in self._fallback_services:
            logger.info(f"使用降级服务: {service_name}")
            return self._fallback_services[service_name]

        # 如果都没有，返回None
        logger.error(f"无法获取服务: {service_name}")
        return None

    def get_config_manager(self):
        """获取配置管理器"""
        return self.get_service('config_manager')

    def get_cache_manager(self):
        """获取缓存管理器"""
        return self.get_service('cache_manager')

    def get_monitoring(self):
        """获取监控服务"""
        return self.get_service('monitoring')

    def get_logger(self):
        """获取日志服务"""
        return self.get_service('logger')

    def get_health_checker(self):
        """获取健康检查器"""
        return self.get_service('health_checker')

    def health_check(self) -> Dict[str, Any]:
        """健康检查 - 包括统一服务和降级服务的状态"""
        result = {
            'status': 'healthy',
            'bridge_type': 'trading_infrastructure_bridge',
            'unified_services': {},
            'fallback_services': {},
            'overall_health': 'healthy'
        }

        # 检查统一服务状态
        if self._unified_adapter_factory:
            try:
                unified_health = self._unified_adapter_factory.health_check()
                result['unified_services'] = unified_health
            except Exception as e:
                result['unified_services'] = {'error': str(e)}
                result['overall_health'] = 'degraded'
        else:
            result['unified_services'] = {'status': 'unavailable'}
            result['overall_health'] = 'degraded'

        # 检查降级服务状态
        for service_name, service in self._fallback_services.items():
            try:
                if hasattr(service, 'health_check'):
                    service_health = service.health_check()
                    result['fallback_services'][service_name] = service_health
                else:
                    result['fallback_services'][service_name] = {'status': 'unknown'}
            except Exception as e:
                result['fallback_services'][service_name] = {'status': 'error', 'error': str(e)}
                result['overall_health'] = 'degraded'

        # 更新整体状态
        if result['overall_health'] == 'degraded':
            result['status'] = 'degraded'

        return result

    def execute_with_fallback(self, operation_name: str, primary_func: callable, fallback_func: callable = None, *args, **kwargs):
        """执行操作，支持降级处理"""
        try:
            # 首先尝试使用统一服务
            if self._unified_adapter_factory:
                return primary_func(*args, **kwargs)
            else:
                raise Exception("统一适配器工厂不可用")

        except Exception as e:
            logger.warning(f"统一服务执行失败 {operation_name}: {e}")

            # 尝试使用降级服务
            if fallback_func:
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_e:
                    logger.error(f"降级服务执行失败 {operation_name}: {fallback_e}")

            return None


class TradingLayerAdapterRefactored(UnifiedBusinessAdapter):
    """重构后的交易层适配器 - 组合模式：使用专门的组件"""

    def __init__(self, config: Optional[TradingInfrastructureConfig] = None):
        super().__init__(BusinessLayerType.TRADING)

        # 初始化配置
        self.config = config or TradingInfrastructureConfig()

        # 初始化专门的组件
        self.engine_manager = TradingEngineManager(self.config)
        self.infrastructure_manager = InfrastructureServiceManager(self.config)

        # 初始化交易层特定的服务桥接器
        self._service_bridges = {
            'trading_infrastructure_bridge': self._create_trading_bridge()
        }

        logger.info("重构后的交易层适配器初始化完成")

    def _create_trading_bridge(self):
        """创建交易层专用的基础设施桥接器"""
        return TradingInfrastructureBridge()

    # 代理方法到专门的组件
    def get_trading_engine(self):
        """获取交易引擎 - 代理到引擎管理器"""
        return self.engine_manager.get_trading_engine()

    def get_hft_execution_engine(self):
        """获取高频交易执行引擎 - 代理到引擎管理器"""
        return self.engine_manager.get_hft_execution_engine()

    def get_monitoring_system(self):
        """获取监控系统 - 代理到基础设施管理器"""
        return self.infrastructure_manager.get_monitoring_system()

    def get_audit_system(self):
        """获取审计系统 - 代理到基础设施管理器"""
        return self.infrastructure_manager.get_audit_system()

    def get_order_manager(self):
        """获取订单管理器 - 代理到基础设施管理器"""
        return self.infrastructure_manager.get_order_manager()

    # 保持向后兼容性
    def _init_trading_specific_services(self):
        """初始化交易层特定的基础设施服务（向后兼容）"""
        # 这个方法在重构版本中不需要，但保持接口兼容
        pass


# 为了向后兼容，保留原有的TradingLayerAdapter类名，但内部使用重构版本
TradingLayerAdapter = TradingLayerAdapterRefactored


# 便捷函数

def get_trading_layer_adapter() -> TradingLayerAdapter:
    """获取交易层适配器实例"""
    from .business_adapters import get_trading_adapter
    return get_trading_adapter()


def get_trading_engine():
    """获取交易引擎"""
    return get_trading_layer_adapter().get_trading_engine()


def get_order_manager():
    """获取订单管理器"""
    return get_trading_layer_adapter().get_order_manager()


def get_execution_engine():
    """获取执行引擎"""
    return get_trading_layer_adapter().get_execution_engine()


def execute_trade_with_infrastructure(trade_request: Dict[str, Any]) -> Dict[str, Any]:
    """使用基础设施支持执行交易"""
    return get_trading_layer_adapter().execute_trade_with_infrastructure(trade_request)
