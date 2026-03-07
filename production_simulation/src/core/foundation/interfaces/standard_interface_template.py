"""
标准接口模板模块（别名模块）
提供向后兼容的导入路径

实际实现在不同的接口文件中
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

# 从foundation.base导入ComponentHealth（如果存在）
try:
    from ..base import ComponentHealth as ComponentHealthEnum
except ImportError:
    ComponentHealthEnum = None


class ComponentStatus(Enum):
    """组件状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


# ComponentHealth可以是Enum或dataclass，根据实际使用情况
try:
    from ..base import ComponentHealth as ComponentHealthEnum
    # 同时提供dataclass版本（向后兼容）
    @dataclass
    class ComponentHealth:
        """组件健康状态（dataclass版本）"""
        healthy: bool = True
        status: str = "ok"
        message: str = ""
        timestamp: datetime = None
        details: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now()
            if self.details is None:
                self.details = {}
except ImportError:
    # 如果无法导入，只提供dataclass版本
    @dataclass
    class ComponentHealth:
        """组件健康状态"""
        healthy: bool = True
        status: str = "ok"
        message: str = ""
        timestamp: datetime = None
        details: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now()
            if self.details is None:
                self.details = {}
    
    ComponentHealthEnum = None


class IStatusProvider(ABC):
    """状态提供者接口"""
    
    @abstractmethod
    def get_status(self) -> ComponentStatus:
        """获取组件状态"""
        pass
    
    @abstractmethod
    def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        pass


class IHealthCheckable(ABC):
    """健康检查接口"""
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass


class ILifecycleManageable(ABC):
    """生命周期管理接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化"""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """启动"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """停止"""
        pass


class IServiceProvider(ABC):
    """服务提供者接口"""
    
    @abstractmethod
    def get_service(self, service_name: str) -> Any:
        """获取服务"""
        pass


class StandardComponent(IStatusProvider, IHealthCheckable, ILifecycleManageable):
    """标准组件基类"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.status = ComponentStatus.INITIALIZING
    
    def get_status(self) -> ComponentStatus:
        return self.status
    
    def get_status_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status.value
        }
    
    def health_check(self) -> Dict[str, Any]:
        return {'healthy': self.status == ComponentStatus.RUNNING}
    
    def initialize(self) -> bool:
        self.status = ComponentStatus.RUNNING
        return True
    
    def start(self) -> bool:
        self.status = ComponentStatus.RUNNING
        return True
    
    def stop(self) -> bool:
        self.status = ComponentStatus.STOPPED
        return True


class StandardInterface:
    """标准接口基类"""
    pass


__all__ = [
    'ComponentStatus',
    'ComponentHealth',
    'IStatusProvider',
    'IHealthCheckable',
    'ILifecycleManageable',
    'IServiceProvider',
    'StandardComponent',
    'StandardInterface'
]

