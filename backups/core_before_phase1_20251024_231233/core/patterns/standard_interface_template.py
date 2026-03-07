#!/usr/bin/env python3
"""
RQA2025 统一接口标准模板

定义所有架构层级的统一接口标准和规范
确保各层级接口的一致性和可扩展性

作者: AI Assistant
版本: 1.0.0
创建时间: 2025年9月29日
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol, TypeVar, Generic
from datetime import datetime
from enum import Enum

# =============================================================================
# 基础类型定义
# =============================================================================

T = TypeVar('T')


class ComponentStatus(Enum):
    """组件状态枚举"""
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"


class ComponentHealth(Enum):
    """组件健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ServiceLifecycle(Enum):
    """服务生命周期枚举"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

# =============================================================================
# 基础接口协议
# =============================================================================


class IStatusProvider(Protocol):
    """状态提供者接口协议"""

    @property
    def status(self) -> ComponentStatus:
        """获取组件状态"""
        ...

    @property
    def health(self) -> ComponentHealth:
        """获取组件健康状态"""
        ...

    def get_status_info(self) -> Dict[str, Any]:
        """获取详细状态信息"""
        ...


class IHealthCheckable(Protocol):
    """健康检查接口协议"""

    def health_check(self) -> Dict[str, Any]:
        """执行健康检查

        Returns:
            Dict[str, Any]: 健康检查结果，包含以下标准字段:
            - service: 服务名称
            - healthy: 是否健康 (bool)
            - status: 状态字符串 ('healthy', 'unhealthy', 'degraded')
            - timestamp: 检查时间 (ISO格式字符串)
            - version: 服务版本
            - details: 详细状态信息 (可选)
            - issues: 发现的问题列表 (可选)
            - recommendations: 修复建议列表 (可选)
        """
        ...

    @property
    def service_name(self) -> str:
        """服务名称"""
        ...

    @property
    def service_version(self) -> str:
        """服务版本"""
        ...


class ILifecycleManageable(Protocol):
    """生命周期管理接口协议"""

    def initialize(self) -> bool:
        """初始化组件"""
        ...

    def start(self) -> bool:
        """启动组件"""
        ...

    def stop(self) -> bool:
        """停止组件"""
        ...

    def shutdown(self) -> bool:
        """关闭组件"""
        ...

    def restart(self) -> bool:
        """重启组件"""
        ...

# =============================================================================
# 标准组件基类
# =============================================================================


class StandardComponent(ABC):
    """标准组件基类

    所有RQA2025架构组件都应继承此类，
    确保统一的接口和行为标准。
    """

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self._name = name
        self._version = version
        self._description = description
        self._status = ComponentStatus.INITIALIZING
        self._health = ComponentHealth.UNKNOWN
        self._start_time = None
        self._last_health_check = None
        self._error_count = 0
        self._metrics = {}

    @property
    def name(self) -> str:
        """组件名称"""
        return self._name

    @property
    def version(self) -> str:
        """组件版本"""
        return self._version

    @property
    def description(self) -> str:
        """组件描述"""
        return self._description

    @property
    def status(self) -> ComponentStatus:
        """组件状态"""
        return self._status

    @property
    def health(self) -> ComponentHealth:
        """组件健康状态"""
        return self._health

    def set_status(self, status: ComponentStatus) -> None:
        """设置组件状态"""
        self._status = status
        if status == ComponentStatus.RUNNING and not self._start_time:
            self._start_time = datetime.now()

    def set_health(self, health: ComponentHealth) -> None:
        """设置组件健康状态"""
        self._health = health
        self._last_health_check = datetime.now()

    def increment_error_count(self) -> None:
        """增加错误计数"""
        self._error_count += 1

    def get_status_info(self) -> Dict[str, Any]:
        """获取组件状态信息"""
        return {
            'name': self._name,
            'version': self._version,
            'description': self._description,
            'status': self._status.value,
            'health': self._health.value,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'error_count': self._error_count,
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds() if self._start_time else 0,
            'metrics': self._metrics.copy()
        }

    def health_check(self) -> Dict[str, Any]:
        """标准健康检查实现"""
        try:
            # 执行具体的健康检查逻辑（子类实现）
            details = self._perform_health_check()

            # 确定健康状态
            healthy = details.get('healthy', True)
            issues = details.get('issues', [])

            health_status = ComponentHealth.HEALTHY.value
            if issues:
                health_status = ComponentHealth.DEGRADED.value
            if not healthy:
                health_status = ComponentHealth.UNHEALTHY.value

            self.set_health(ComponentHealth(health_status))

            return {
                'service': self._name,
                'healthy': healthy,
                'status': health_status,
                'timestamp': datetime.now().isoformat(),
                'version': self._version,
                'details': details,
                'issues': issues,
                'recommendations': details.get('recommendations', [])
            }

        except Exception as e:
            self.set_health(ComponentHealth.UNHEALTHY)
            self.increment_error_count()

            return {
                'service': self._name,
                'healthy': False,
                'status': ComponentHealth.UNHEALTHY.value,
                'timestamp': datetime.now().isoformat(),
                'version': self._version,
                'error': str(e),
                'issues': [f'健康检查失败: {e}'],
                'recommendations': ['检查组件配置和依赖']
            }

    @abstractmethod
    def _perform_health_check(self) -> Dict[str, Any]:
        """执行具体的健康检查逻辑（子类必须实现）"""

    @property
    def service_name(self) -> str:
        """服务名称（兼容现有接口）"""
        return self._name

    @property
    def service_version(self) -> str:
        """服务版本（兼容现有接口）"""
        return self._version

# =============================================================================
# 标准服务接口
# =============================================================================


class IServiceProvider(Protocol, Generic[T]):
    """服务提供者接口协议"""

    def get_service(self, name: str) -> Optional[T]:
        """获取服务实例"""
        ...

    def has_service(self, name: str) -> bool:
        """检查服务是否存在"""
        ...

    def list_services(self) -> List[str]:
        """列出所有服务"""
        ...


class IConfigProvider(Protocol):
    """配置提供者接口协议"""

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        ...

    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        ...

    def has(self, key: str) -> bool:
        """检查配置是否存在"""
        ...

    def reload(self) -> bool:
        """重新加载配置"""
        ...


class ICacheProvider(Protocol):
    """缓存提供者接口协议"""

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        ...

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        ...

    def clear(self) -> bool:
        """清空缓存"""
        ...

    def has(self, key: str) -> bool:
        """检查缓存是否存在"""
        ...


class ILoggerProvider(Protocol):
    """日志提供者接口协议"""

    def debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""
        ...

    def info(self, message: str, **kwargs) -> None:
        """记录信息日志"""
        ...

    def warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""
        ...

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """记录错误日志"""
        ...

    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误日志"""
        ...

# =============================================================================
# 统一接口标准导出
# =============================================================================


__all__ = [
    # 基础类型
    'ComponentStatus',
    'ComponentHealth',
    'ServiceLifecycle',

    # 接口协议
    'IStatusProvider',
    'IHealthCheckable',
    'ILifecycleManageable',
    'IServiceProvider',
    'IConfigProvider',
    'ICacheProvider',
    'ILoggerProvider',

    # 标准基类
    'StandardComponent',

    # 类型变量
    'T'
]
