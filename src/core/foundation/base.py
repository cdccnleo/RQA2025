#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 核心层基础组件 - 增强版

提供统一的基础组件架构，所有核心服务组件的基础类和接口。

作者: 系统架构师
创建时间: 2025-01-28
版本: 2.1.0

主要特性:
- 统一的组件生命周期管理
- 健康检查和监控
- 配置管理
- 事件驱动架构支持
"""

import time
import uuid
import logging
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum

from src.core.constants import (
    MAX_RECORDS, SECONDS_PER_MINUTE
)
from datetime import datetime


class ComponentStatus(Enum):

    """组件状态枚举"""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ComponentHealth(Enum):

    """组件健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentInfo:

    """组件信息"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    status: ComponentStatus = ComponentStatus.UNKNOWN
    health: ComponentHealth = ComponentHealth.UNKNOWN
    created_time: float = None
    last_updated: float = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.created_time is None:
            self.created_time = time.time()
        if self.last_updated is None:
            self.last_updated = time.time()
        if self.metadata is None:
            self.metadata = {}


class BaseComponent(ABC):

    """组件基类"""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):

        self.name = name
        self.version = version
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._status = ComponentStatus.INITIALIZING
        self._health = ComponentHealth.UNKNOWN
        self._created_time = time.time()
        self._last_updated = time.time()
        self._metadata = {}
        self._initialized = False
        self._started = False

    def initialize(self) -> bool:
        """初始化组件（防止重复初始化）"""
        # 检查是否已经初始化
        if self._initialized:
            self.logger.debug(f"组件 {self.name} 已经初始化，跳过重复初始化")
            return True

        try:
            self.set_status(ComponentStatus.INITIALIZING)
            result = self._initialize_impl()

            if result:
                self._initialized = True
                self.set_status(ComponentStatus.INITIALIZED)
                self.set_health(ComponentHealth.HEALTHY)
                self.logger.info(f"组件 {self.name} 初始化成功")
            else:
                self.set_status(ComponentStatus.ERROR)
                self.set_health(ComponentHealth.UNHEALTHY)
                self.logger.error(f"组件 {self.name} 初始化失败")

            return result

        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.set_health(ComponentHealth.UNHEALTHY)
            self.logger.error(f"组件初始化失败: {e}")
            return False

    def shutdown(self) -> bool:
        """关闭组件"""
        try:
            self.set_status(ComponentStatus.STOPPED)
            self.set_health(ComponentHealth.UNHEALTHY)
            self._started = False
            self.logger.info(f"组件 {self.name} 已关闭")
            return True
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.logger.error(f"组件关闭失败: {e}")
            return False

    def _initialize_impl(self) -> bool:
        """初始化实现（子类可重写）"""
        return True

    def start(self) -> bool:
        """启动组件"""
        if not self._initialized:
            if not self.initialize():
                return False

        try:
            self._status = ComponentStatus.STARTING
            self._last_updated = time.time()

            if self._start_impl():
                self._status = ComponentStatus.RUNNING
                self._started = True
                self.logger.info(f"组件 {self.name} 启动成功")
                return True
            else:
                self._status = ComponentStatus.ERROR
                self.logger.error(f"组件 {self.name} 启动失败")
                return False
        except Exception as e:
            self._status = ComponentStatus.ERROR
            self.logger.error(f"组件 {self.name} 启动异常: {e}")
            return False

    def stop(self) -> bool:
        """停止组件"""
        try:
            self._status = ComponentStatus.STOPPING
            self._last_updated = time.time()

            if self._stop_impl():
                self._status = ComponentStatus.STOPPED
                self._started = False
                self.logger.info(f"组件 {self.name} 停止成功")
                return True
            else:
                self._status = ComponentStatus.ERROR
                self.logger.error(f"组件 {self.name} 停止失败")
                return False
        except Exception as e:
            self._status = ComponentStatus.ERROR
            self.logger.error(f"组件 {self.name} 停止异常: {e}")
            return False

    def _start_impl(self) -> bool:
        """启动实现（子类可重写）"""
        return True

    def _stop_impl(self) -> bool:
        """停止实现（子类可重写）"""
        return True

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def is_started(self) -> bool:
        """检查是否已启动"""
        return self._started

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._status == ComponentStatus.RUNNING

    @property
    def status(self) -> ComponentStatus:
        """获取组件状态"""
        return self._status

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "name": self.name,
            "status": self._status.value,
            "health": self._health.value,
            "version": self.version,
            "initialized": self._initialized,
            "started": self._started
        }

    def get_health(self) -> ComponentHealth:
        """获取健康状态"""
        return self._health

    def set_status(self, status: ComponentStatus):
        """设置状态"""
        self._status = status
        self._last_updated = time.time()

    def set_health(self, health: ComponentHealth):
        """设置健康状态"""
        self._health = health
        self._last_updated = time.time()

    def get_info(self) -> ComponentInfo:
        """获取组件信息"""
        return ComponentInfo(
            name=self.name,
            version=self.version,
            description=self.description,
            status=self._status,
            health=self._health,
            created_time=self._created_time,
            last_updated=self._last_updated,
            metadata=self._metadata.copy()
        )

    def add_metadata(self, key: str, value: Any):
        """添加元数据"""
        self._metadata[key] = value
        self._last_updated = time.time()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self._metadata.get(key, default)

    def health_check(self) -> Dict[str, Any]:
        """健康检查（返回详细信息）"""
        return {
            "status": self._status.value,
            "health": self._health.value,
            "name": self.name,
            "version": self.version,
            "timestamp": time.time()
        }


class BaseService(BaseComponent):

    """服务基类"""

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):

        super().__init__(name, version, description)
        self._dependencies = []
        self._config = {}

    def add_dependency(self, dependency: str):
        """添加依赖"""
        if dependency not in self._dependencies:
            self._dependencies.append(dependency)

    def get_dependencies(self) -> List[str]:
        """获取依赖列表"""
        return self._dependencies.copy()

    def set_config(self, key: str, value: Any):
        """设置配置"""
        self._config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        return self._config.get(key, default)

    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self._config.copy()


# ============================================================================
# 工具函数
# ============================================================================


def generate_id(prefix: str = "") -> str:
    """生成唯一ID"""
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}_{timestamp}_{unique_id}"
    return f"{timestamp}_{unique_id}"


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """验证配置"""
    if not isinstance(config, dict):
        return False
    return all(key in config for key in required_keys)


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """验证必需字段，返回缺失的字段列表"""
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    return missing_fields


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """安全获取字典值，支持嵌套键路径

    Args:
        data: 数据字典
        key: 键名，支持点分隔的嵌套路径，如 'config.database.host'
        default: 默认值

    Returns:
        键对应的值或默认值

    Examples:
        >>> data = {'config': {'database': {'host': 'localhost'}}}
        >>> safe_get(data, 'config.database.host')
        'localhost'
        >>> safe_get(data, 'missing.key', 'default')
        'default'
    """
    if not isinstance(data, dict):
        return default

    # 支持嵌套键路径
    keys = key.split('.')
    current = data

    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default

    return current


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """合并字典（dict2覆盖dict1）"""
    result = dict1.copy() if dict1 else {}
    if dict2:
        result.update(dict2)
    return result


def format_timestamp(timestamp: float) -> str:
    """格式化时间戳"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def calculate_duration(start_time: float, end_time: float = None) -> float:
    """计算持续时间"""
    if end_time is None:
        end_time = time.time()
    return end_time - start_time


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""

    def decorator(func):

        def wrapper(*args, **kwargs):

            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # 指数退避
            raise last_exception
        return wrapper
    return decorator


# ==================== 增强功能 ====================

@dataclass
class ComponentConfig:
    """组件配置"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    auto_start: bool = True
    dependencies: List[str] = field(default_factory=list)
    config_params: Dict[str, Any] = field(default_factory=dict)
    health_check_interval: int = SECONDS_PER_MINUTE  # 秒
    max_restart_attempts: int = 3


@dataclass
class ComponentMetrics:
    """组件指标"""
    component_name: str
    start_time: float = field(default_factory=time.time)
    operation_count: int = 0
    error_count: int = 0
    last_operation_time: Optional[float] = None
    avg_operation_time: float = 0.0
    memory_usage_mb: Optional[float] = None

    @property
    def uptime_seconds(self) -> float:
        """运行时间（秒）"""
        return time.time() - self.start_time

    def record_operation(self, duration: float, success: bool = True):
        """记录操作"""
        self.operation_count += 1
        if not success:
            self.error_count += 1
        self.last_operation_time = time.time()

        # 更新平均操作时间
        if self.operation_count == 1:
            self.avg_operation_time = duration
        else:
            self.avg_operation_time = (
                (self.avg_operation_time * (self.operation_count - 1)) + duration
            ) / self.operation_count


class ComponentEventListener(Protocol):
    """组件事件监听器协议"""

    def on_component_started(self, component_name: str) -> None:
        """组件启动事件"""
        ...

    def on_component_stopped(self, component_name: str) -> None:
        """组件停止事件"""
        ...

    def on_component_error(self, component_name: str, error: Exception) -> None:
        """组件错误事件"""
        ...

    def on_health_changed(self, component_name: str, healthy: bool) -> None:
        """健康状态变更事件"""
        ...


class ComponentRegistry:
    """组件注册表 - 管理所有组件的生命周期"""

    def __init__(self):
        self._components: Dict[str, BaseComponent] = {}
        self._configs: Dict[str, ComponentConfig] = {}
        self._metrics: Dict[str, ComponentMetrics] = {}
        self._listeners: List[ComponentEventListener] = []
        self._logger = logging.getLogger(__name__)

    def register_component(self, component: BaseComponent,
                          config: Optional[ComponentConfig] = None) -> None:
        """注册组件"""
        component_name = component.name

        if component_name in self._components:
            self._logger.warning(f"组件 '{component_name}' 已存在，将被覆盖")

        self._components[component_name] = component
        self._configs[component_name] = config or ComponentConfig(name=component_name)
        self._metrics[component_name] = ComponentMetrics(component_name=component_name)

        # 通知监听器
        for listener in self._listeners:
            try:
                listener.on_component_started(component_name)
            except Exception as e:
                self._logger.error(f"监听器通知失败: {e}")

        self._logger.info(f"组件 '{component_name}' 已注册")

    def unregister_component(self, component_name: str) -> bool:
        """注销组件"""
        if component_name in self._components:
            del self._components[component_name]
            if component_name in self._configs:
                del self._configs[component_name]
            if component_name in self._metrics:
                del self._metrics[component_name]

            # 通知监听器
            for listener in self._listeners:
                try:
                    listener.on_component_stopped(component_name)
                except Exception as e:
                    self._logger.error(f"监听器通知失败: {e}")

            self._logger.info(f"组件 '{component_name}' 已注销")
            return True
        return False

    def get_component(self, component_name: str) -> Optional[BaseComponent]:
        """获取组件"""
        return self._components.get(component_name)

    def get_component_metrics(self, component_name: str) -> Optional[ComponentMetrics]:
        """获取组件指标"""
        return self._metrics.get(component_name)

    def list_components(self) -> List[str]:
        """列出所有组件"""
        return list(self._components.keys())

    def add_event_listener(self, listener: ComponentEventListener) -> None:
        """添加事件监听器"""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_event_listener(self, listener: ComponentEventListener) -> None:
        """移除事件监听器"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def perform_health_check(self) -> Dict[str, bool]:
        """执行所有组件的健康检查"""
        results = {}
        for name, component in self._components.items():
            try:
                healthy = component.health_check()
                results[name] = healthy

                # 通知监听器健康状态变更
                for listener in self._listeners:
                    try:
                        listener.on_health_changed(name, healthy)
                    except Exception as e:
                        self._logger.error(f"健康状态监听器通知失败: {e}")

            except Exception as e:
                self._logger.error(f"组件 '{name}' 健康检查失败: {e}")
                results[name] = False

        return results


# 全局组件注册表实例
_component_registry = ComponentRegistry()


def get_component_registry() -> ComponentRegistry:
    """获取全局组件注册表实例"""
    return _component_registry


def register_global_component(component: BaseComponent,
                             config: Optional[ComponentConfig] = None) -> None:
    """注册全局组件"""
    _component_registry.register_component(component, config)


def get_global_component(component_name: str) -> Optional[BaseComponent]:
    """获取全局组件"""
    return _component_registry.get_component(component_name)
