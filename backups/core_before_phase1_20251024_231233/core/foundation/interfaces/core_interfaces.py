#!/usr/bin/env python3
"""
RQA2025 核心服务层统一接口定义

基于标准接口模板定义核心服务层所有组件的标准接口
确保与基础设施层和其他层级的接口一致性

作者: AI Assistant
版本: 2.0.0 (基于统一接口标准)
更新时间: 2025年9月29日
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Protocol
from .standard_interface_template import (
    IStatusProvider, IHealthCheckable, ILifecycleManageable,
    IServiceProvider, StandardComponent, ComponentStatus, ComponentHealth
)


class ICoreComponent(IStatusProvider, IHealthCheckable, ILifecycleManageable, Protocol):
    """核心组件统一接口协议

    所有核心服务层组件都必须实现此接口协议，
    确保统一的组件管理和监控能力。
    """

    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务详细信息"""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取组件性能指标"""


class IEventBus(IStatusProvider, IHealthCheckable, Protocol):
    """事件总线接口协议"""

    @abstractmethod
    def publish(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """发布事件"""

    @abstractmethod
    def subscribe(self, event_type: str, handler: callable) -> bool:
        """订阅事件"""

    @abstractmethod
    def unsubscribe(self, event_type: str, handler: callable) -> bool:
        """取消订阅事件"""


class IDependencyContainer(IServiceProvider, IStatusProvider, IHealthCheckable, Protocol):
    """依赖注入容器接口协议"""

    @abstractmethod
    def register(self, name: str, service: Any = None, service_type: Optional[type] = None,
                 factory: Optional[callable] = None, lifecycle: str = "singleton",
                 dependencies: Optional[List[str]] = None, **kwargs) -> bool:
        """注册服务"""

    @abstractmethod
    def resolve(self, name: str) -> Any:
        """解析服务依赖"""

    @abstractmethod
    def create_scope(self) -> Any:
        """创建作用域"""


class IBusinessProcessOrchestrator(IStatusProvider, IHealthCheckable, Protocol):
    """业务流程编排器接口协议"""

    @abstractmethod
    def start_process(self, process_name: str, context: Dict[str, Any]) -> str:
        """启动业务流程"""

    @abstractmethod
    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """获取流程状态"""

    @abstractmethod
    def stop_process(self, process_id: str) -> bool:
        """停止业务流程"""


class ILayerInterface(Protocol):
    """层间接口协议"""

    @abstractmethod
    def communicate_up(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """向上层通信"""

    @abstractmethod
    def communicate_down(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """向下层通信"""

    @abstractmethod
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """验证消息格式"""


# =============================================================================
# 标准实现类
# =============================================================================

class CoreComponent(StandardComponent):
    """核心组件标准实现基类"""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        super().__init__(name, version, description)

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            'component_type': 'core_component',
            'layer': 'core_services',
            **self.get_status_info()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'uptime_seconds': self.get_status_info().get('uptime_seconds', 0),
            'error_count': self.get_status_info().get('error_count', 0),
            'health_checks': 1 if self.get_status_info().get('last_health_check') else 0
        }

    @abstractmethod
    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查的具体逻辑"""


__all__ = [
    # 接口协议
    'ICoreComponent',
    'IEventBus',
    'IDependencyContainer',
    'IBusinessProcessOrchestrator',
    'ILayerInterface',

    # 标准实现
    'CoreComponent',

    # 导入的标准类型
    'ComponentStatus',
    'ComponentHealth',
    'StandardComponent'
]
