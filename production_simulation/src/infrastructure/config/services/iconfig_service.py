"""
iconfig_service 模块

提供 iconfig_service 相关功能和接口。
"""

import logging

from .cache_service import CacheService
from .diff_service import DictDiffService
from .event_service import ConfigEventBus
# EventService别名，保持向后兼容性
EventService = ConfigEventBus
from .unified_hot_reload import UnifiedHotReload
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
#!/usr/bin/env python3
"""
配置服务核心接口定义

Phase 1重构：使用组合模式替代多重继承
定义清晰的服务接口，避免循环依赖
"""


class IConfigService(Protocol):
    """配置服务核心接口

    定义配置管理的基本操作协议
    """

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        ...

    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        ...

    def delete(self, key: str) -> bool:
        """删除配置项"""
        ...

    def exists(self, key: str) -> bool:
        """检查配置项是否存在"""
        ...

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的配置键"""
        ...

    def clear(self) -> bool:
        """清空所有配置"""
        ...


class IConfigStorageService(Protocol):
    """配置存储服务接口

    定义配置持久化相关的操作
    """

    def load(self, source: str) -> Dict[str, Any]:
        """从指定源加载配置"""
        ...

    def save(self, config: Dict[str, Any], target: str) -> bool:
        """保存配置到指定目标"""
        ...

    def reload(self) -> bool:
        """重新加载配置"""
        ...


class IConfigValidationService(Protocol):
    """配置验证服务接口

    定义配置验证相关的操作
    """

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置数据"""
        ...

    def validate_key(self, key: str, value: Any) -> Dict[str, Any]:
        """验证单个配置项"""
        ...


class IConfigMonitoringService(Protocol):
    """配置监控服务接口

    定义配置监控和统计相关的操作
    """

    def record_operation(self, operation: str, key: str = None, duration: float = None):
        """记录配置操作"""
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """获取配置统计信息"""
        ...

    def get_health_status(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        ...


class IConfigEventService(Protocol):
    """配置事件服务接口

    定义配置变更通知相关的操作
    """

    def subscribe(self, event_type: str, callback: callable) -> str:
        """订阅配置事件"""
        ...

    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        ...

    def publish(self, event_type: str, data: Dict[str, Any]):
        """发布配置事件"""
        ...

# 具体实现类的基类


class BaseConfigService(ABC):
    """配置服务基类

    提供通用的初始化和基础功能
    """

    def __init__(self, service_name: str = "config: Dict[str, Any]_service"):
        self._service_name = service_name
        self._initialized = False
        self._start_time = None
        self._lock = None

    def _ensure_initialized(self):
        """确保服务已初始化"""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    @abstractmethod
    def _initialize(self):
        """子类实现具体的初始化逻辑"""

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            "service_name": self._service_name,
            "initialized": self._initialized,
            "start_time": self._start_time,
            "type": self.__class__.__name__
        }

# 服务注册表


class ServiceRegistry:
    """服务注册表

    管理各种配置服务的注册和发现
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._service_types: Dict[str, type] = {}

    def register_service(self, service_type: str, service_class: type):
        """注册服务类型"""
        self._service_types[service_type] = service_class

    def create_service(self, service_type: str, *args, **kwargs) -> Any:
        """创建服务实例"""
        if service_type not in self._service_types:
            raise ValueError(f"未知的服务类型: {service_type}")

        service_class = self._service_types[service_type]
        return service_class(*args, **kwargs)

    def get_service(self, service_name: str) -> Optional[Any]:
        """获取已创建的服务实例"""
        return self._services.get(service_name)

    def register_instance(self, service_name: str, service_instance: Any):
        """注册服务实例"""
        self._services[service_name] = service_instance

    def list_services(self) -> List[str]:
        """列出所有已注册的服务"""
        return list(self._services.keys())

    def list_service_types(self) -> List[str]:
        """列出所有可用的服务类型"""
        return list(self._service_types.keys())


# 全局服务注册表实例
service_registry = ServiceRegistry()

# 延迟导入以避免循环依赖


def _register_existing_services():
    """注册现有的服务"""
    try:
        service_registry.register_service("cache", CacheService)
        service_registry.register_service("event", EventService)
        service_registry.register_service("diff", DictDiffService)
        service_registry.register_service("unified_hot_reload", UnifiedHotReload)
    except ImportError as e:
        # 记录导入错误但不中断程序
        logger = logging.getLogger(__name__)
        logger.warning(f"服务注册失败: {e}")


# 注册现有服务
_register_existing_services()




