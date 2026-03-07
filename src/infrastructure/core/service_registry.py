#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一基础设施服务注册表

负责统一管理所有基础设施服务的创建和获取，支持单例、工厂等多种生命周期模式
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable, Type, Union
from enum import Enum


class ServiceLifecycle(Enum):
    """服务生命周期模式"""
    SINGLETON = "singleton"  # 单例模式
    FACTORY = "factory"  # 工厂模式，每次调用创建新实例
    SCOPED = "scoped"  # 作用域模式（预留）


class InfrastructureServiceRegistry:
    """
    统一基础设施服务注册表（单例模式）
    
    职责：
    - 统一管理所有基础设施服务的注册和获取
    - 支持单例、工厂等多种生命周期模式
    - 确保服务只初始化一次（单例模式）
    - 提供服务的统一访问入口
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化服务注册表"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._services: Dict[str, Any] = {}  # 服务实例缓存（单例模式）
        self._service_factories: Dict[str, Callable[[], Any]] = {}  # 服务工厂函数
        self._service_lifecycles: Dict[str, ServiceLifecycle] = {}  # 服务生命周期模式
        self._service_classes: Dict[str, Type] = {}  # 服务类（用于工厂模式）
        self._initialization_flags: Dict[str, bool] = {}  # 服务初始化标志
        self._registry_lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
        self._initialized = True
        
        self._logger.info("基础设施服务注册表初始化完成")
    
    def register_singleton(
        self,
        service_name: str,
        service_class: Optional[Type] = None,
        factory: Optional[Callable[[], Any]] = None,
        instance: Optional[Any] = None
    ) -> bool:
        """
        注册单例服务
        
        Args:
            service_name: 服务名称
            service_class: 服务类（可选，如果提供则延迟实例化）
            factory: 工厂函数（可选，用于创建服务实例）
            instance: 服务实例（可选，如果提供则直接使用）
            
        Returns:
            bool: 注册是否成功
        """
        with self._registry_lock:
            if service_name in self._services:
                self._logger.warning(f"服务 {service_name} 已注册，跳过重复注册")
                return False
            
            if instance is not None:
                # 直接使用提供的实例
                self._services[service_name] = instance
                self._initialization_flags[service_name] = True
                self._logger.info(f"注册单例服务（实例）: {service_name}")
            elif factory is not None:
                # 使用工厂函数
                self._service_factories[service_name] = factory
                self._service_lifecycles[service_name] = ServiceLifecycle.SINGLETON
                self._initialization_flags[service_name] = False
                self._logger.info(f"注册单例服务（工厂）: {service_name}")
            elif service_class is not None:
                # 使用服务类（延迟实例化）
                self._service_classes[service_name] = service_class
                self._service_lifecycles[service_name] = ServiceLifecycle.SINGLETON
                self._initialization_flags[service_name] = False
                self._logger.info(f"注册单例服务（类）: {service_name}")
            else:
                self._logger.error(f"注册服务 {service_name} 失败：未提供实例、工厂或类")
                return False
            
            return True
    
    def register_factory(
        self,
        service_name: str,
        factory: Callable[[], Any]
    ) -> bool:
        """
        注册工厂服务（每次调用创建新实例）
        
        Args:
            service_name: 服务名称
            factory: 工厂函数
            
        Returns:
            bool: 注册是否成功
        """
        with self._registry_lock:
            if service_name in self._service_factories:
                self._logger.warning(f"服务工厂 {service_name} 已注册，跳过重复注册")
                return False
            
            self._service_factories[service_name] = factory
            self._service_lifecycles[service_name] = ServiceLifecycle.FACTORY
            self._logger.info(f"注册工厂服务: {service_name}")
            
            return True
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        获取服务实例
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务实例，如果不存在则返回None
        """
        with self._registry_lock:
            lifecycle = self._service_lifecycles.get(service_name)
            
            if lifecycle == ServiceLifecycle.SINGLETON:
                # 单例模式：返回缓存的实例或创建新实例
                if service_name in self._services:
                    return self._services[service_name]
                
                # 延迟实例化
                if service_name in self._service_factories:
                    factory = self._service_factories[service_name]
                    try:
                        instance = factory()
                        self._services[service_name] = instance
                        self._initialization_flags[service_name] = True
                        self._logger.debug(f"创建单例服务实例: {service_name}")
                        return instance
                    except Exception as e:
                        self._logger.error(
                            f"创建单例服务实例失败 {service_name}: {e}",
                            exc_info=True
                        )
                        return None
                
                if service_name in self._service_classes:
                    service_class = self._service_classes[service_name]
                    try:
                        instance = service_class()
                        self._services[service_name] = instance
                        self._initialization_flags[service_name] = True
                        self._logger.debug(f"创建单例服务实例: {service_name}")
                        return instance
                    except Exception as e:
                        self._logger.error(
                            f"创建单例服务实例失败 {service_name}: {e}",
                            exc_info=True
                        )
                        return None
                
                self._logger.warning(f"服务 {service_name} 未注册")
                return None
            
            elif lifecycle == ServiceLifecycle.FACTORY:
                # 工厂模式：每次调用创建新实例
                if service_name in self._service_factories:
                    factory = self._service_factories[service_name]
                    try:
                        return factory()
                    except Exception as e:
                        self._logger.error(
                            f"工厂创建服务实例失败 {service_name}: {e}",
                            exc_info=True
                        )
                        return None
                
                self._logger.warning(f"服务工厂 {service_name} 未注册")
                return None
            
            else:
                # 检查是否直接注册了实例
                if service_name in self._services:
                    return self._services[service_name]
                
                self._logger.warning(f"服务 {service_name} 未注册")
                return None
    
    def is_service_registered(self, service_name: str) -> bool:
        """
        检查服务是否已注册
        
        Args:
            service_name: 服务名称
            
        Returns:
            bool: 是否已注册
        """
        with self._registry_lock:
            return (
                service_name in self._services or
                service_name in self._service_factories or
                service_name in self._service_classes
            )
    
    def is_service_initialized(self, service_name: str) -> bool:
        """
        检查服务是否已初始化（仅对单例模式有效）
        
        Args:
            service_name: 服务名称
            
        Returns:
            bool: 是否已初始化
        """
        with self._registry_lock:
            return self._initialization_flags.get(service_name, False)
    
    def unregister_service(self, service_name: str) -> bool:
        """
        取消注册服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            bool: 取消注册是否成功
        """
        with self._registry_lock:
            removed = False
            
            if service_name in self._services:
                del self._services[service_name]
                removed = True
            
            if service_name in self._service_factories:
                del self._service_factories[service_name]
                removed = True
            
            if service_name in self._service_classes:
                del self._service_classes[service_name]
                removed = True
            
            if service_name in self._service_lifecycles:
                del self._service_lifecycles[service_name]
            
            if service_name in self._initialization_flags:
                del self._initialization_flags[service_name]
            
            if removed:
                self._logger.info(f"取消注册服务: {service_name}")
            
            return removed
    
    def get_all_registered_services(self) -> Dict[str, ServiceLifecycle]:
        """
        获取所有已注册的服务
        
        Returns:
            服务名称到生命周期模式的映射
        """
        with self._registry_lock:
            return self._service_lifecycles.copy()
    
    def initialize_all_services(self) -> Dict[str, bool]:
        """
        初始化所有已注册的单例服务
        
        Returns:
            服务名称到初始化结果的映射
        """
        with self._registry_lock:
            results = {}
            
            for service_name, lifecycle in self._service_lifecycles.items():
                if lifecycle == ServiceLifecycle.SINGLETON:
                    if not self._initialization_flags.get(service_name, False):
                        instance = self.get_service(service_name)
                        results[service_name] = instance is not None
                    else:
                        results[service_name] = True
                else:
                    # 工厂模式不需要初始化
                    results[service_name] = True
            
            return results


# 全局单例访问函数
def get_service_registry() -> InfrastructureServiceRegistry:
    """
    获取全局基础设施服务注册表实例
    
    Returns:
        InfrastructureServiceRegistry: 服务注册表实例
    """
    return InfrastructureServiceRegistry()