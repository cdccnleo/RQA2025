#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 核心层基础组件 - 修复版

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
    STARTED = "started"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ComponentHealth(Enum):
    """组件健康状态枚举"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


@dataclass
class ComponentMetadata:
    """组件元数据"""
    name: str
    version: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class IServiceComponent(Protocol):
    """服务组件接口"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """初始化组件"""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """关闭组件"""
        pass


class BaseComponent(IServiceComponent):
    """基础组件类 - 所有核心组件的基础"""
    
    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description
        
        # 状态管理
        self._status = ComponentStatus.UNKNOWN
        self._health = ComponentHealth.UNKNOWN
        
        # 生命周期管理
        self._initialized = False
        self._started = False
        
        # 元数据
        self._metadata = ComponentMetadata(
            name=name,
            version=version,
            description=description
        )
        
        # 日志
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # 统计信息
        self._stats = {}
        
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self._status.value,
            "health": self._health.value,
            "initialized": self._initialized,
            "started": self._started,
            "metadata": {
                "description": self._metadata.description,
                "dependencies": self._metadata.dependencies,
                "tags": self._metadata.tags,
                "created_at": self._metadata.created_at.isoformat(),
                "updated_at": self._metadata.updated_at.isoformat()
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "component": self.name,
            "status": self._status.value,
            "health": self._health.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

    def set_status(self, status: ComponentStatus):
        """设置组件状态"""
        self._status = status
        self._metadata.updated_at = datetime.now()

    def get_status_enum(self) -> ComponentStatus:
        """获取状态枚举"""
        return self._status

    def set_health(self, health: ComponentHealth):
        """设置组件健康状态"""
        self._health = health

    def get_health_enum(self) -> ComponentHealth:
        """获取健康状态枚举"""
        return self._health

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

    @abstractmethod
    def _initialize_impl(self) -> bool:
        """具体的初始化实现，由子类实现"""
        pass

    def start(self) -> bool:
        """启动组件"""
        if not self._initialized:
            if not self.initialize():
                return False
                
        try:
            self.set_status(ComponentStatus.STARTING)
            result = self._start_impl()
            
            if result:
                self._started = True
                self.set_status(ComponentStatus.STARTED)
                self.set_health(ComponentHealth.HEALTHY)
                self.logger.info(f"组件 {self.name} 启动成功")
            else:
                self.set_status(ComponentStatus.ERROR)
                self.set_health(ComponentHealth.UNHEALTHY)
                self.logger.error(f"组件 {self.name} 启动失败")
                
            return result
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.set_health(ComponentHealth.UNHEALTHY)
            self.logger.error(f"组件启动失败: {e}")
            return False

    def _start_impl(self) -> bool:
        """启动实现，默认成功"""
        return True

    def stop(self) -> bool:
        """停止组件"""
        try:
            self.set_status(ComponentStatus.STOPPING)
            result = self._stop_impl()
            
            if result:
                self._started = False
                self.set_status(ComponentStatus.STOPPED)
                self.logger.info(f"组件 {self.name} 停止成功")
            else:
                self.set_status(ComponentStatus.ERROR)
                self.logger.error(f"组件 {self.name} 停止失败")
                
            return result
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.logger.error(f"组件停止失败: {e}")
            return False

    def _stop_impl(self) -> bool:
        """停止实现，默认成功"""
        return True

    def restart(self) -> bool:
        """重启组件"""
        if not self.stop():
            return False
        return self.start()

    def update_metadata(self, **kwargs):
        """更新元数据"""
        for key, value in kwargs.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)
        self._metadata.updated_at = datetime.now()

    def get_metadata(self) -> ComponentMetadata:
        """获取元数据"""
        return self._metadata

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def is_started(self) -> bool:
        """检查是否已启动"""
        return self._started

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()

    def update_stats(self, key: str, value: Any):
        """更新统计信息"""
        self._stats[key] = value

    def reset_stats(self):
        """重置统计信息"""
        self._stats.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}', status={self._status.value}, health={self._health.value})"

    def __str__(self) -> str:
        return f"{self.name} v{self.version} [{self._status.value}/{self._health.value}]"


def generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def get_component_info(component: BaseComponent) -> Dict[str, Any]:
    """获取组件信息"""
    return {
        "name": component.name,
        "version": component.version,
        "status": component.get_status_enum().value,
        "health": component.get_health_enum().value,
        "initialized": component.is_initialized(),
        "started": component.is_started()
    }

