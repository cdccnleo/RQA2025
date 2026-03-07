"""
集成层接口模块
"""

# 从foundation.interfaces导入核心接口
try:
    from src.core.foundation.interfaces.core_interfaces import (
        ICoreComponent,
        IEventBus,
        IDependencyContainer,
        IBusinessProcessOrchestrator,
        ILayerInterface
    )
except ImportError:
    # 提供Protocol定义
    from typing import Protocol, Dict, Any, List

    class ICoreComponent(Protocol):
        """核心组件接口"""
        def get_status(self) -> Any: ...
        def health_check(self) -> Dict[str, Any]: ...

    class IEventBus(Protocol):
        """事件总线接口"""
        def publish(self, event_type: str, data: Dict[str, Any]) -> bool: ...

    class IDependencyContainer(Protocol):
        """依赖容器接口"""
        def resolve(self, name: str) -> Any: ...

    class IBusinessProcessOrchestrator(Protocol):
        """业务流程编排器接口"""
        def start_process(self, name: str, context: Dict[str, Any]) -> str: ...

    class ILayerInterface(Protocol):
        """层间接口"""
        def communicate_up(self, message: Dict[str, Any]) -> Dict[str, Any]: ...

    class IServiceComponent(Protocol):
        """服务组件接口"""
        def start(self) -> bool: ...
        def stop(self) -> bool: ...
        def get_status(self) -> Dict[str, Any]: ...
        def health_check(self) -> Dict[str, Any]: ...

    class ILayerComponent(Protocol):
        """层组件接口"""
        def initialize(self) -> bool: ...
        def shutdown(self) -> bool: ...
        def get_layer_info(self) -> Dict[str, Any]: ...

    class IBusinessAdapter(Protocol):
        """业务适配器接口"""
        def adapt_request(self, request: Dict[str, Any]) -> Dict[str, Any]: ...
        def adapt_response(self, response: Dict[str, Any]) -> Dict[str, Any]: ...

    class IAdapterComponent(Protocol):
        """适配器组件接口"""
        def transform(self, data: Any) -> Any: ...
        def validate(self, data: Any) -> bool: ...

    class IServiceBridge(Protocol):
        """服务桥接接口"""
        def connect(self, source: str, target: str) -> bool: ...
        def disconnect(self, source: str, target: str) -> bool: ...
        def transfer(self, data: Any, source: str, target: str) -> Any: ...

# 确保IServiceComponent始终可用
try:
    # 尝试从core_interfaces导入
    from src.core.foundation.interfaces.core_interfaces import IServiceComponent
except ImportError:
    # 如果导入失败，使用本地定义
    from typing import Protocol, Dict, Any

    class IServiceComponent(Protocol):
        """服务组件接口"""
        def start(self) -> bool: ...
        def stop(self) -> bool: ...
        def get_status(self) -> Dict[str, Any]: ...
        def health_check(self) -> Dict[str, Any]: ...

    class ILayerComponent(Protocol):
        """层组件接口"""
        def initialize(self) -> bool: ...
        def shutdown(self) -> bool: ...
        def get_layer_info(self) -> Dict[str, Any]: ...

    class IBusinessAdapter(Protocol):
        """业务适配器接口"""
        def adapt_request(self, request: Dict[str, Any]) -> Dict[str, Any]: ...
        def adapt_response(self, response: Dict[str, Any]) -> Dict[str, Any]: ...

    class IAdapterComponent(Protocol):
        """适配器组件接口"""
        def transform(self, data: Any) -> Any: ...
        def validate(self, data: Any) -> bool: ...

    class IServiceBridge(Protocol):
        """服务桥接接口"""
        def connect(self, source: str, target: str) -> bool: ...
        def disconnect(self, source: str, target: str) -> bool: ...
        def transfer(self, data: Any, source: str, target: str) -> Any: ...

__all__ = [
    'ICoreComponent',
    'IEventBus',
    'IDependencyContainer',
    'IBusinessProcessOrchestrator',
    'ILayerInterface',
    'IServiceComponent',
    'ILayerComponent',
    'IBusinessAdapter',
    'IAdapterComponent',
    'IServiceBridge'
]

