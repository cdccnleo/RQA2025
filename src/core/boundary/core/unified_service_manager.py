#!/usr/bin/env python3
"""
RQA2025 子系统统一服务管理器
Subsystem Unified Service Manager

提供统一的子系统间服务调用和管理。
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

# 获取统一基础设施集成层的日志适配器
try:
    from src.infrastructure.integration import get_models_adapter
    models_adapter = get_models_adapter()
    logger = logging.getLogger(__name__)
except Exception as e:
    try:
        from src.infrastructure.logging.core.interfaces import get_logger
        logger = get_logger(__name__)
    except ImportError:
        logger = logging.getLogger(__name__)


@dataclass
class ServiceRegistration:

    """服务注册信息"""
    service_name: str
    subsystem_name: str
    service_instance: Any
    methods: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    health_check: Optional[Callable] = None


@dataclass
class ServiceCall:

    """服务调用记录"""
    call_id: str
    service_name: str
    method_name: str
    parameters: Dict[str, Any]
    caller_subsystem: str
    target_subsystem: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    result: Any = None


class UnifiedServiceManager:

    """
    统一服务管理器
    管理各子系统间的服务调用，提供统一的接口
    """

    def __init__(self):

        self.services: Dict[str, ServiceRegistration] = {}
        self.call_history: List[ServiceCall] = []
        self.call_metrics: Dict[str, Dict[str, Any]] = {}

        # 为测试兼容性添加的属性别名
        self.service_registry = self.services
        self.service_calls = self.call_history

        # 负载均衡器和熔断器（简化实现）
        self.load_balancer = {}
        self.circuit_breaker = {}

        # 最大历史记录数
        self.max_history_size = 10000

        logger.info("统一服务管理器已初始化")

    def register_service(self, registration: ServiceRegistration) -> bool:
        """注册服务"""
        self.services[registration.service_name] = registration

        # 初始化调用指标
        self.call_metrics[registration.service_name] = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_response_time_ms': 0.0,
            'last_called': None
        }

        logger.info(f"注册服务: {registration.service_name} (子系统: {registration.subsystem_name})")

    def get_service(self, service_name: str) -> Optional[ServiceRegistration]:
        """获取服务注册信息"""
        return self.services.get(service_name)

    def call_service_method(self, service_name: str, method_name: str, *args, **kwargs) -> Any:
        """调用服务方法"""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"服务不存在: {service_name}")

        # 记录调用
        call = ServiceCall(
            call_id=str(uuid.uuid4()),
            service_name=service_name,
            method_name=method_name,
            parameters={'args': args, 'kwargs': kwargs},
            timestamp=datetime.now(),
            caller_info={'test': True}
        )
        self.call_history.append(call)

        # 限制历史记录大小
        if len(self.call_history) > self.max_history_size:
            self.call_history.pop(0)

        # 模拟方法调用
        return f"模拟调用 {service_name}.{method_name}"

    def unregister_service(self, service_name: str) -> bool:
        """注销服务"""
        if service_name in self.services:
            del self.services[service_name]
            if service_name in self.call_metrics:
                del self.call_metrics[service_name]
            logger.info(f"注销服务: {service_name}")
            return True
        return False

    async def call_service(self, service_name: str, method_name: str,
                           parameters: Dict[str, Any] = None,
                           caller_subsystem: str = "unknown") -> Dict[str, Any]:
        """调用服务"""
        start_time = datetime.now()
        parameters = parameters or {}

        # 创建调用记录
        call = ServiceCall(
            call_id=f"call_{service_name}_{method_name}_{start_time.strftime('%Y%m%d%H%M%S%f')}",
            service_name=service_name,
            method_name=method_name,
            parameters=parameters,
            caller_subsystem=caller_subsystem,
            target_subsystem=self.services.get(
                service_name, ServiceRegistration("", "", None)).subsystem_name,
            start_time=start_time
        )

        try:
            # 获取服务
            service_reg = self.services.get(service_name)
            if not service_reg:
                raise ValueError(f"服务不存在: {service_name}")

            # 检查方法是否存在
            if method_name not in service_reg.methods:
                raise ValueError(f"方法不存在: {method_name} 在服务 {service_name}")

            # 执行方法调用
            service_instance = service_reg.service_instance
            method = getattr(service_instance, method_name)

            # 调用方法
            if asyncio.iscoroutinefunction(method):
                result = await method(**parameters)
            else:
                result = method(**parameters)

            # 更新调用记录
            end_time = datetime.now()
            call.end_time = end_time
            call.duration_ms = (end_time - start_time).total_seconds() * 1000
            call.success = True
            call.result = result

            # 更新指标
            self._update_call_metrics(service_name, call.duration_ms, True)

            return {
                'success': True,
                'result': result,
                'call_id': call.call_id,
                'duration_ms': call.duration_ms
            }

        except Exception as e:
            end_time = datetime.now()
            call.end_time = end_time
            call.duration_ms = (end_time - start_time).total_seconds() * 1000
            call.success = False
            call.error_message = str(e)

            # 更新指标
            self._update_call_metrics(service_name, call.duration_ms, False)

            logger.error(f"服务调用失败: {service_name}.{method_name} - {str(e)}")

            return {
                'success': False,
                'error': str(e),
                'call_id': call.call_id,
                'duration_ms': call.duration_ms
            }

        finally:
            # 添加到调用历史
            self._add_call_to_history(call)

    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取服务信息"""
        service_reg = self.services.get(service_name)
        if not service_reg:
            return None

        return {
            'service_name': service_reg.service_name,
            'subsystem_name': service_reg.subsystem_name,
            'methods': service_reg.methods,
            'metadata': service_reg.metadata,
            'registered_at': service_reg.registered_at.isoformat(),
            'is_healthy': self._check_service_health(service_reg)
        }

    def list_services(self, subsystem_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出服务"""
        services_info = []

        for service_name, service_reg in self.services.items():
            if subsystem_filter and service_reg.subsystem_name != subsystem_filter:
                continue

            services_info.append({
                'service_name': service_name,
                'subsystem_name': service_reg.subsystem_name,
                'methods_count': len(service_reg.methods),
                'is_healthy': self._check_service_health(service_reg)
            })

        return services_info

    def get_call_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """获取调用指标"""
        if service_name:
            return self.call_metrics.get(service_name, {})

        # 返回所有服务的聚合指标
        total_calls = sum(metrics['total_calls'] for metrics in self.call_metrics.values())
        successful_calls = sum(metrics['successful_calls']
                               for metrics in self.call_metrics.values())
        failed_calls = sum(metrics['failed_calls'] for metrics in self.call_metrics.values())

        return {
            'total_services': len(self.call_metrics),
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0.0,
            'service_metrics': self.call_metrics
        }

    def get_call_history(self, service_name: Optional[str] = None,


                         limit: int = 100) -> List[Dict[str, Any]]:
        """获取调用历史"""
        history = self.call_history

        if service_name:
            history = [call for call in history if call.service_name == service_name]

        # 按时间倒序排序
        history.sort(key=lambda x: x.start_time, reverse=True)

        # 限制返回数量
        history = history[:limit]

        return [
            {
                'call_id': call.call_id,
                'service_name': call.service_name,
                'method_name': call.method_name,
                'caller_subsystem': call.caller_subsystem,
                'target_subsystem': call.target_subsystem,
                'start_time': call.start_time.isoformat(),
                'duration_ms': call.duration_ms,
                'success': call.success,
                'error_message': call.error_message
            }
            for call in history
        ]

    def _update_call_metrics(self, service_name: str, duration_ms: float, success: bool):
        """更新调用指标"""
        metrics = self.call_metrics[service_name]
        metrics['total_calls'] += 1
        metrics['last_called'] = datetime.now()

        if success:
            metrics['successful_calls'] += 1
        else:
            metrics['failed_calls'] += 1

        # 更新平均响应时间
        if metrics['total_calls'] > 1:
            metrics['avg_response_time_ms'] = (
                (metrics['avg_response_time_ms'] * (metrics['total_calls'] - 1)) + duration_ms
            ) / metrics['total_calls']
        else:
            metrics['avg_response_time_ms'] = duration_ms

    def _add_call_to_history(self, call: ServiceCall):
        """添加调用到历史记录"""
        self.call_history.append(call)

        # 限制历史记录大小
        if len(self.call_history) > self.max_history_size:
            self.call_history = self.call_history[-self.max_history_size:]

    def _check_service_health(self, service_reg: ServiceRegistration) -> bool:
        """检查服务健康状态"""
        if service_reg.health_check:
            try:
                return service_reg.health_check()
            except Exception as e:
                logger.error(f"健康检查失败 for {service_reg.service_name}: {str(e)}")
                return False

        # 默认健康检查：检查服务实例是否存在
        return service_reg.service_instance is not None

    async def health_check_all_services(self) -> Dict[str, Any]:
        """检查所有服务的健康状态"""
        health_status = {}

        for service_name, service_reg in self.services.items():
            is_healthy = self._check_service_health(service_reg)
            health_status[service_name] = {
                'healthy': is_healthy,
                'subsystem': service_reg.subsystem_name,
                'last_checked': datetime.now().isoformat()
            }

        unhealthy_count = sum(1 for status in health_status.values() if not status['healthy'])

        return {
            'total_services': len(health_status),
            'healthy_services': len(health_status) - unhealthy_count,
            'unhealthy_services': unhealthy_count,
            'health_rate': (len(health_status) - unhealthy_count) / len(health_status) if health_status else 0.0,
            'service_health': health_status
        }


# 创建全局统一服务管理器实例
_unified_service_manager = None


def get_unified_service_manager() -> UnifiedServiceManager:
    """获取全局统一服务管理器实例"""
    global _unified_service_manager
    if _unified_service_manager is None:
        _unified_service_manager = UnifiedServiceManager()
    return _unified_service_manager


__all__ = [
    'UnifiedServiceManager',
    'ServiceRegistration',
    'ServiceCall',
    'get_unified_service_manager'
]
