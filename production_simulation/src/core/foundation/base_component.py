#!/usr/bin/env python3
"""
统一组件基类框架

提供所有组件类的基础抽象和公共功能，消除代码重复
创建时间: 2025-11-03
版本: 1.0
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
import logging
from enum import Enum


class ComponentStatus(Enum):
    """组件状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class IComponent(ABC):
    """
    组件基础接口
    
    所有组件类必须实现的核心接口
    """
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """执行组件主要功能"""
        pass


class BaseComponent(IComponent):
    """
    组件基类
    
    提供所有组件的公共功能实现：
    - 日志管理
    - 配置管理
    - 状态管理
    - 错误处理
    - 生命周期管理
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化组件
        
        Args:
            name: 组件名称
            config: 组件配置
        """
        self.name = name
        self.config = config or {}
        self._status = ComponentStatus.UNINITIALIZED
        self._logger = self._setup_logger()
        self._created_at = datetime.now()
        self._initialized_at: Optional[datetime] = None
        self._error: Optional[Exception] = None
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return logger
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取组件信息
        
        Returns:
            包含组件状态、配置等信息的字典
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'status': self._status.value,
            'created_at': self._created_at.isoformat(),
            'initialized_at': self._initialized_at.isoformat() if self._initialized_at else None,
            'config': self.config,
            'error': str(self._error) if self._error else None
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化组件
        
        Args:
            config: 初始化配置
            
        Returns:
            初始化是否成功
        """
        try:
            self.config.update(config)
            
            # 调用子类的初始化逻辑
            if self._do_initialize(config):
                self._status = ComponentStatus.INITIALIZED
                self._initialized_at = datetime.now()
                self._logger.info(f"组件 {self.name} 初始化成功")
                return True
            else:
                self._status = ComponentStatus.ERROR
                self._logger.error(f"组件 {self.name} 初始化失败")
                return False
                
        except Exception as e:
            self._status = ComponentStatus.ERROR
            self._error = e
            self._logger.error(f"组件 {self.name} 初始化异常: {e}")
            return False
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """
        子类实现具体的初始化逻辑
        
        Args:
            config: 初始化配置
            
        Returns:
            初始化是否成功
        """
        # 默认实现：空操作，子类可以覆盖
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """
        执行组件功能
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            执行结果
        """
        if self._status != ComponentStatus.INITIALIZED:
            raise RuntimeError(f"组件 {self.name} 未初始化，无法执行")
        
        try:
            self._status = ComponentStatus.RUNNING
            result = self._do_execute(*args, **kwargs)
            self._status = ComponentStatus.INITIALIZED
            return result
        except Exception as e:
            self._status = ComponentStatus.ERROR
            self._error = e
            self._logger.error(f"组件 {self.name} 执行失败: {e}")
            raise
    
    @abstractmethod
    def _do_execute(self, *args, **kwargs) -> Any:
        """
        子类实现具体的执行逻辑
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            执行结果
        """
        pass
    
    def get_status(self) -> ComponentStatus:
        """获取组件状态"""
        return self._status
    
    def is_initialized(self) -> bool:
        """检查组件是否已初始化"""
        return self._status == ComponentStatus.INITIALIZED
    
    def get_error(self) -> Optional[Exception]:
        """获取最后一次错误"""
        return self._error
    
    def reset(self):
        """重置组件状态"""
        self._status = ComponentStatus.UNINITIALIZED
        self._initialized_at = None
        self._error = None
        self._logger.info(f"组件 {self.name} 已重置")


class ComponentFactory:
    """
    统一组件工厂
    
    负责创建和管理组件实例，替代所有重复的ComponentFactory类
    """
    
    def __init__(self):
        self._components: Dict[str, BaseComponent] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def create_component(
        self, 
        component_type: str, 
        component_class: Type[BaseComponent],
        config: Dict[str, Any]
    ) -> Optional[BaseComponent]:
        """
        创建组件实例
        
        Args:
            component_type: 组件类型标识
            component_class: 组件类
            config: 组件配置
            
        Returns:
            创建的组件实例，失败返回None
        """
        try:
            component = component_class(name=component_type, config=config)
            
            if component.initialize(config):
                self._components[component_type] = component
                self._logger.info(f"成功创建组件: {component_type}")
                return component
            else:
                self._logger.error(f"组件初始化失败: {component_type}")
                return None
                
        except Exception as e:
            self._logger.error(f"创建组件失败 {component_type}: {e}")
            return None
    
    def get_component(self, component_type: str) -> Optional[BaseComponent]:
        """获取已创建的组件"""
        return self._components.get(component_type)
    
    def remove_component(self, component_type: str) -> bool:
        """移除组件"""
        if component_type in self._components:
            del self._components[component_type]
            self._logger.info(f"移除组件: {component_type}")
            return True
        return False
    
    def get_all_components(self) -> Dict[str, BaseComponent]:
        """获取所有组件"""
        return self._components.copy()
    
    def clear(self):
        """清空所有组件"""
        self._components.clear()
        self._logger.info("已清空所有组件")


# 便捷的组件装饰器
def component(name: Optional[str] = None):
    """
    组件装饰器，用于标记组件类
    
    使用示例:
        @component("my_component")
        class MyComponent(BaseComponent):
            pass
    """
    def decorator(cls):
        cls._component_name = name or cls.__name__
        return cls
    return decorator


__all__ = [
    'IComponent',
    'BaseComponent',
    'ComponentFactory',
    'ComponentStatus',
    'component'
]

