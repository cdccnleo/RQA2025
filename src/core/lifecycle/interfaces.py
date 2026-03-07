#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组件生命周期接口定义

定义组件生命周期管理的标准接口和契约
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class LifecycleState(Enum):
    """生命周期状态枚举"""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LifecycleEvent:
    """生命周期事件"""
    component_id: str
    component_name: str
    event_type: str  # 'initialized', 'started', 'stopped', 'error'
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ILifecycleComponent(ABC):
    """生命周期组件接口"""
    
    @property
    @abstractmethod
    def component_id(self) -> str:
        """获取组件唯一标识"""
        pass
    
    @property
    @abstractmethod
    def component_name(self) -> str:
        """获取组件名称"""
        pass
    
    @property
    @abstractmethod
    def lifecycle_state(self) -> LifecycleState:
        """获取当前生命周期状态"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化组件
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """
        启动组件
        
        Returns:
            bool: 启动是否成功
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        停止组件
        
        Returns:
            bool: 停止是否成功
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        获取组件依赖的其他组件ID列表
        
        Returns:
            List[str]: 依赖的组件ID列表
        """
        pass


class ILifecycleManager(ABC):
    """生命周期管理器接口"""
    
    @abstractmethod
    def register_component(
        self,
        component: ILifecycleComponent,
        dependencies: Optional[List[str]] = None
    ) -> bool:
        """
        注册组件到生命周期管理器
        
        Args:
            component: 要注册的组件
            dependencies: 依赖的组件ID列表
            
        Returns:
            bool: 注册是否成功
        """
        pass
    
    @abstractmethod
    def unregister_component(self, component_id: str) -> bool:
        """
        取消注册组件
        
        Args:
            component_id: 组件ID
            
        Returns:
            bool: 取消注册是否成功
        """
        pass
    
    @abstractmethod
    def get_component(self, component_id: str) -> Optional[ILifecycleComponent]:
        """
        获取组件实例
        
        Args:
            component_id: 组件ID
            
        Returns:
            组件实例，如果不存在则返回None
        """
        pass
    
    @abstractmethod
    def initialize_all(self) -> Dict[str, bool]:
        """
        初始化所有已注册的组件（按依赖顺序）
        
        Returns:
            Dict[str, bool]: 组件ID到初始化结果的映射
        """
        pass
    
    @abstractmethod
    def start_all(self) -> Dict[str, bool]:
        """
        启动所有已注册的组件（按依赖顺序）
        
        Returns:
            Dict[str, bool]: 组件ID到启动结果的映射
        """
        pass
    
    @abstractmethod
    def stop_all(self) -> Dict[str, bool]:
        """
        停止所有已注册的组件（按依赖顺序，逆序）
        
        Returns:
            Dict[str, bool]: 组件ID到停止结果的映射
        """
        pass
    
    @abstractmethod
    def get_component_state(self, component_id: str) -> Optional[LifecycleState]:
        """
        获取组件的生命周期状态
        
        Args:
            component_id: 组件ID
            
        Returns:
            生命周期状态，如果组件不存在则返回None
        """
        pass
    
    @abstractmethod
    def get_all_components(self) -> Dict[str, ILifecycleComponent]:
        """
        获取所有已注册的组件
        
        Returns:
            组件ID到组件实例的映射
        """
        pass