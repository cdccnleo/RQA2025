import logging
#!/usr/bin/env python3
"""
RQA2025 风控层基础设施适配器

专门为风控层提供基础设施服务访问接口，
基于统一业务层适配器架构，实现风控层的特定需求。
"""

from typing import Dict, Any
from src.core.integration.unified_business_adapters import UnifiedBusinessAdapter, BusinessLayerType


logger = logging.getLogger(__name__)


class RiskLayerAdapter(UnifiedBusinessAdapter):

    """风控层适配器"""

    def __init__(self):

        super().__init__(BusinessLayerType.RISK)
        self._init_risk_specific_services()

    def _init_risk_specific_services(self):
        """初始化风控层特定的基础设施服务"""
        try:
            # 风控层特定的服务桥接器
            # 注意：风控层可能还没有专门的基础设施桥接器，使用通用适配器
            self._service_bridges = {
                'risk_infrastructure_bridge': self._create_risk_bridge()
            }

            logger.info("风控层特定服务桥接器初始化完成")

        except Exception as e:
            logger.warning(f"风控层特定服务桥接器初始化失败，使用基础服务: {e}")

    def _create_risk_bridge(self):
        """创建风控层专用的基础设施桥接器"""
        # 这里可以根据风控层的具体需求创建专门的桥接器
        # 暂时返回基础实现
        return RiskInfrastructureBridge()

    def get_risk_manager(self):
        """获取风险管理器"""
        try:
            from src.risk.risk_manager import RiskManager
            return RiskManager()
        except ImportError:
            logger.warning("风险管理器导入失败")
            return None

    def get_risk_monitor(self):
        """获取风险监控器"""
        try:
            from src.risk.realtime_risk_monitor import RealTimeRiskMonitor
            return RealTimeRiskMonitor()
        except ImportError:
            try:
                from src.risk.real_time_risk import RealTimeRiskMonitor
                return RealTimeRiskMonitor()
            except ImportError:
                logger.warning("风险监控器导入失败")
                return None

    def get_risk_calculator(self):
        """获取风险计算器"""
        try:
            from src.risk.risk_calculation_engine import RiskCalculationEngine
            return RiskCalculationEngine()
        except ImportError:
            logger.warning("风险计算器导入失败")
            return None

    def get_alert_system(self):
        """获取告警系统"""
        try:
            from src.risk.alert_system import AlertSystem
            return AlertSystem()
        except ImportError:
            logger.warning("告警系统导入失败")
            return None

    def health_check(self) -> Dict[str, Any]:
        """风控层健康检查"""
        base_health = super().health_check()

        # 添加风控层特定检查
        risk_specific_health = {
            'risk_manager': self._check_risk_manager_health(),
            'risk_monitor': self._check_risk_monitor_health(),
            'risk_calculator': self._check_risk_calculator_health(),
            'alert_system': self._check_alert_system_health()
        }

        base_health['risk_specific_services'] = risk_specific_health

        # 更新整体状态
        for service_name, health_info in risk_specific_health.items():
            if health_info.get('status') != 'healthy':
                base_health['overall_status'] = 'degraded'
                break

        return base_health

    def _check_risk_manager_health(self) -> Dict[str, Any]:
        """检查风险管理器健康状态"""
        manager = self.get_risk_manager()
        if manager and hasattr(manager, 'health_check'):
            try:
                return manager.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif manager:
            return {'status': 'healthy', 'manager_available': True}
        return {'status': 'unknown'}

    def _check_risk_monitor_health(self) -> Dict[str, Any]:
        """检查风险监控器健康状态"""
        monitor = self.get_risk_monitor()
        if monitor and hasattr(monitor, 'health_check'):
            try:
                return monitor.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif monitor:
            return {'status': 'healthy', 'monitor_available': True}
        return {'status': 'unknown'}

    def _check_risk_calculator_health(self) -> Dict[str, Any]:
        """检查风险计算器健康状态"""
        calculator = self.get_risk_calculator()
        if calculator and hasattr(calculator, 'health_check'):
            try:
                return calculator.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif calculator:
            return {'status': 'healthy', 'calculator_available': True}
        return {'status': 'unknown'}

    def _check_alert_system_health(self) -> Dict[str, Any]:
        """检查告警系统健康状态"""
        alert_system = self.get_alert_system()
        if alert_system and hasattr(alert_system, 'health_check'):
            try:
                return alert_system.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif alert_system:
            return {'status': 'healthy', 'alert_system_available': True}
        return {'status': 'unknown'}

    def get_risk_layer_metrics(self) -> Dict[str, Any]:
        """获取风控层性能指标"""
        metrics = {
            'timestamp': '2025 - 01 - 27T10:00:00Z',
            'layer_type': 'risk',
            'infrastructure_metrics': {},
            'risk_metrics': {}
        }

        # 获取基础设施服务指标
        for service_name, service in self.get_infrastructure_services().items():
            if hasattr(service, 'get_metrics'):
                try:
                    metrics['infrastructure_metrics'][service_name] = service.get_metrics()
                except Exception as e:
                    metrics['infrastructure_metrics'][service_name] = {'error': str(e)}

        # 获取风控组件指标
        components = {
            'risk_manager': self.get_risk_manager(),
            'risk_monitor': self.get_risk_monitor(),
            'risk_calculator': self.get_risk_calculator(),
            'alert_system': self.get_alert_system()
        }

        for component_name, component in components.items():
            if component and hasattr(component, 'get_metrics'):
                try:
                    metrics['risk_metrics'][component_name] = component.get_metrics()
                except Exception as e:
                    metrics['risk_metrics'][component_name] = {'error': str(e)}

        return metrics

    def assess_risk_with_infrastructure(self, risk_request: Dict[str, Any]) -> Dict[str, Any]:
        """使用基础设施支持进行风险评估 - 重构版：职责分离"""
        result = self._initialize_risk_result(risk_request)

        try:
            # 获取基础设施服务
            services = self._get_risk_infrastructure_services()

            # 记录风险评估开始
            self._record_risk_start(services, result)

            # 检查缓存避免重复评估
            if self._check_risk_cache(risk_request, services, result):
                return result

            # 执行实际风险评估
            self._execute_risk_assessment(risk_request, services, result)

            # 记录评估完成
            self._record_risk_completion(services)

        except Exception as e:
            self._handle_risk_error(e, result)

        return result

    def _initialize_risk_result(self, risk_request: Dict[str, Any]) -> Dict[str, Any]:
        """初始化风险评估结果对象"""
        return {
            'timestamp': datetime.now().isoformat(),
            'layer_type': 'risk',
            'risk_request': risk_request,
            'assessed': False,
            'infrastructure_used': []
        }

    def _get_risk_infrastructure_services(self) -> Dict[str, Any]:
        """获取风险评估所需的基础设施服务"""
        services = self.get_infrastructure_services()
        return {
            'cache_manager': services.get('cache_manager'),
            'monitoring': services.get('monitoring')
        }

    def _record_risk_start(self, services: Dict[str, Any], result: Dict[str, Any]) -> None:
        """记录风险评估开始"""
        if services['monitoring']:
            services['monitoring'].record_metric('risk_assessment_start', 1, {'layer': 'risk'})
            result['infrastructure_used'].append('monitoring')

    def _check_risk_cache(self, risk_request: Dict[str, Any],
                         services: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """检查风险评估缓存，避免重复评估"""
        if not services['cache_manager']:
            return False

        cache_key = f"risk_{hash(str(risk_request))}"
        cached_result = services['cache_manager'].get(cache_key)

        if cached_result:
            result['cached_result'] = cached_result
            result['assessed'] = True
            result['infrastructure_used'].append('cache')
            return True

        return False

    def _execute_risk_assessment(self, risk_request: Dict[str, Any],
                                services: Dict[str, Any], result: Dict[str, Any]) -> None:
        """执行实际的风险评估逻辑"""
        risk_calculator = self.get_risk_calculator()
        if risk_calculator:
            assessment_result = risk_calculator.assess_risk(risk_request)
            result['assessment_result'] = assessment_result
            result['assessed'] = True

            # 缓存评估结果
            self._cache_risk_result(risk_request, assessment_result, services, result)

    def _cache_risk_result(self, risk_request: Dict[str, Any],
                          assessment_result: Any, services: Dict[str, Any],
                          result: Dict[str, Any]) -> None:
        """缓存风险评估结果"""
        if services['cache_manager']:
            cache_key = f"risk_{hash(str(risk_request))}"
            services['cache_manager'].set(cache_key, assessment_result, 600)  # 缓存10分钟
            result['infrastructure_used'].append('cache')

    def _record_risk_completion(self, services: Dict[str, Any]) -> None:
        """记录风险评估完成"""
        if services['monitoring']:
            services['monitoring'].record_metric('risk_assessment_complete', 1, {'layer': 'risk'})

    def _handle_risk_error(self, error: Exception, result: Dict[str, Any]) -> None:
        """处理风险评估错误"""
        result['error'] = str(error)
        logger.error(f"风险评估失败: {error}")

        # 记录错误指标
        services = self._get_risk_infrastructure_services()
        if services['monitoring']:
            services['monitoring'].record_metric('risk_assessment_error', 1, {
                'layer': 'risk', 'error': str(error)
            })

        return result


class RiskInfrastructureBridge:

    """风控层基础设施桥接器"""

    def __init__(self):

        self._services = {}
        self._init_services()

    def _init_services(self):
        """初始化基础设施服务"""
        # 这里可以根据风控层的具体需求初始化专门的服务

    def get_service(self, service_name: str):
        """获取服务"""
        return self._services.get(service_name)

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy',
            'bridge_type': 'risk_infrastructure_bridge',
            'services': list(self._services.keys())
        }


# 便捷函数

def get_risk_layer_adapter() -> RiskLayerAdapter:
    """获取风控层适配器实例"""
    from .business_adapters import get_risk_adapter
    return get_risk_adapter()


def get_risk_manager():
    """获取风险管理器"""
    return get_risk_layer_adapter().get_risk_manager()


def get_risk_monitor():
    """获取风险监控器"""
    return get_risk_layer_adapter().get_risk_monitor()


def get_risk_calculator():
    """获取风险计算器"""
    return get_risk_layer_adapter().get_risk_calculator()


def assess_risk_with_infrastructure(risk_request: Dict[str, Any]) -> Dict[str, Any]:
    """使用基础设施支持进行风险评估"""
    return get_risk_layer_adapter().assess_risk_with_infrastructure(risk_request)
