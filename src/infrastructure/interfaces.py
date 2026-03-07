
# 基础接口
#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
"""
基础设施层接口定义

定义基础设施层所有组件的标准接口
"""


class IInfrastructureComponent(ABC):

    """基础设施组件基础接口"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""

# 配置管理接口


class IConfigManagerComponent(IInfrastructureComponent):

    """配置管理器接口"""

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""

    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """设置配置值"""

    @abstractmethod
    def reload_config(self) -> bool:
        """重新加载配置"""

# 缓存管理接口


class ICacheManagerComponent(IInfrastructureComponent):

    """缓存管理器接口"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""

    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""

# 日志管理接口


class ILoggerComponent(IInfrastructureComponent):

    """日志管理器接口"""

    @abstractmethod
    def info(self, message: str, **kwargs):
        """记录信息日志"""

    @abstractmethod
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """记录错误日志"""

    @abstractmethod
    def warning(self, message: str, **kwargs):
        """记录警告日志"""

    @abstractmethod
    def debug(self, message: str, **kwargs):
        """记录调试日志"""

# 安全管理接口


class ISecurityManagerComponent(IInfrastructureComponent):

    """安全管理器接口"""

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """用户认证"""

    @abstractmethod
    def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """用户授权"""

    @abstractmethod
    def encrypt_data(self, data: str) -> str:
        """数据加密"""

    @abstractmethod
    def decrypt_data(self, encrypted_data: str) -> str:
        """数据解密"""

# 错误处理接口


class IErrorHandlerComponent(IInfrastructureComponent):

    """错误处理器接口"""

    @abstractmethod
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理错误"""

    @abstractmethod
    def log_error(self, error: Exception, level: str = "error") -> bool:
        """记录错误"""

# 资源管理接口


class IResourceManagerComponent(IInfrastructureComponent):

    """资源管理器接口"""

    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""

    @abstractmethod
    def allocate_resource(self, resource_type: str, amount: int) -> bool:
        """分配资源"""

    @abstractmethod
    def release_resource(self, resource_type: str, amount: int) -> bool:
        """释放资源"""

# 健康检查接口


class IHealthCheckerComponent(IInfrastructureComponent):

    """健康检查器接口"""

    @abstractmethod
    def perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查"""

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""

# 服务管理接口


class IServiceManagerComponent(IInfrastructureComponent):

    """服务管理器接口"""

    @abstractmethod
    def start_service(self, service_name: str) -> bool:
        """启动服务"""

    @abstractmethod
    def stop_service(self, service_name: str) -> bool:
        """停止服务"""

    @abstractmethod
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """获取服务状态"""


__all__ = []
'IInfrastructureComponent',
'IConfigManagerComponent',
'ICacheManagerComponent',
'ILoggerComponent',
'ISecurityManagerComponent',
'IErrorHandlerComponent',
'IResourceManagerComponent',
'IHealthCheckerComponent',
'IServiceManagerComponent'
