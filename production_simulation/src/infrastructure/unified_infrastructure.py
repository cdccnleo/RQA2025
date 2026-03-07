"""
unified_infrastructure 模块

提供 unified_infrastructure 相关功能和接口。
"""

from enum import Enum


class CacheType(Enum):
    """缓存类型"""
    SMART = "smart"
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"


class ServiceLifecycle(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class InfrastructureManager:
    """基础设施管理器 - 统一管理所有基础设施组件"""

    def __init__(self):
        # 简化初始化，不依赖可能不存在的工厂
        self._services = {}
        self._service_lifecycle = {}

    def get_config_manager(self, manager_type: str = "unified", **kwargs):
        """获取配置管理器"""
        # 返回一个简单的配置管理器模拟
        return {"type": manager_type, "config": kwargs}

    def get_monitor(self, monitor_type: str = "unified", **kwargs):
        """获取监控器"""
        # 返回一个简单的监控器模拟
        return {"type": monitor_type, "config": kwargs}

    def get_cache_manager(self, cache_type: str = "smart", **kwargs):
        """获取缓存管理器"""
        # 返回一个简单的缓存管理器模拟
        return {"type": cache_type, "config": kwargs}

    def get_cache(self, cache_type: str = "smart", **kwargs):
        """获取缓存管理器（别名方法）"""
        return self.get_cache_manager(cache_type, **kwargs)

    def register_service(self, name: str, service, lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON):
        """注册服务"""
        self._services[name] = service
        self._service_lifecycle[name] = lifecycle

    def get_service(self, name: str):
        """获取服务"""
        return self._services.get(name)

    def get_available_config_managers(self):
        """获取可用的配置管理器类型"""
        return ["unified", "basic", "advanced"]

    def get_available_monitors(self):
        """获取可用的监控器类型"""
        return ["unified", "performance", "health"]

    def get_available_cache_managers(self):
        """获取可用的缓存管理器类型"""
        return ["smart", "memory", "redis", "disk", "hybrid"]


# 全局基础设施管理器
_infrastructure_manager = InfrastructureManager()

# 便捷函数


def get_infrastructure_manager() -> InfrastructureManager:
    """获取全局基础设施管理器"""
    return _infrastructure_manager


def get_config_manager(manager_type: str = "unified", **kwargs):
    """获取配置管理器"""
    return _infrastructure_manager.get_config_manager(manager_type, **kwargs)


def get_monitor(monitor_type: str = "unified", **kwargs):
    """获取监控器"""
    return _infrastructure_manager.get_monitor(monitor_type, **kwargs)


def get_cache_manager(cache_type: str = "smart", **kwargs):
    """获取缓存管理器"""
    return _infrastructure_manager.get_cache_manager(cache_type, **kwargs)


def register_infrastructure_service(name: str, service, lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON):
    """注册基础设施服务"""
    _infrastructure_manager.register_service(name, service, lifecycle)


def get_infrastructure_service(name: str):
    """获取基础设施服务"""
    return _infrastructure_manager.get_service(name)


# 导出主要类型和常量
__all__ = [
    # 缓存系统
    'CacheType',

    # 服务生命周期
    'ServiceLifecycle',

    # 基础设施管理器
    'InfrastructureManager',
    'get_infrastructure_manager',
    'get_config_manager',
    'get_monitor',
    'get_cache_manager',
    'register_infrastructure_service',
    'get_infrastructure_service',
]
