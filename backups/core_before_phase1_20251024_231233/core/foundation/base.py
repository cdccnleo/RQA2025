#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心层基础组件
提供公共基类和工具函数
"""

import time
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


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
        self._status = ComponentStatus.UNKNOWN
        self._health = ComponentHealth.UNKNOWN
        self._created_time = time.time()
        self._last_updated = time.time()
        self._metadata = {}
        self._initialized = False
        self._started = False

    def initialize(self) -> bool:
        """初始化组件"""
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

    @abstractmethod
    def shutdown(self) -> bool:
        """关闭组件"""

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

    def get_status(self) -> ComponentStatus:
        """获取状态"""
        return self._status

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

    def health_check(self) -> bool:
        """健康检查（子类可重写）"""
        return self._health == ComponentHealth.HEALTHY


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
    """安全获取字典值"""
    return data.get(key, default) if isinstance(data, dict) else default


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
