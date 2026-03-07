#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
适配器生命周期包装器

将适配器包装为生命周期组件，以便集成到生命周期管理器
"""

import logging
from typing import List, Optional, Any
from src.core.lifecycle.interfaces import ILifecycleComponent, LifecycleState


class AdapterLifecycleWrapper(ILifecycleComponent):
    """
    适配器生命周期包装器
    
    将适配器包装为生命周期组件，实现ILifecycleComponent接口
    """
    
    def __init__(
        self,
        adapter_id: str,
        adapter_name: str,
        adapter_instance: Any,
        dependencies: Optional[List[str]] = None
    ):
        """
        初始化适配器包装器
        
        Args:
            adapter_id: 适配器唯一标识
            adapter_name: 适配器名称
            adapter_instance: 适配器实例
            dependencies: 依赖的其他适配器ID列表
        """
        self._adapter_id = adapter_id
        self._adapter_name = adapter_name
        self._adapter_instance = adapter_instance
        self._dependencies = dependencies or []
        self._lifecycle_state = LifecycleState.UNKNOWN
        self._logger = logging.getLogger(__name__)
    
    @property
    def component_id(self) -> str:
        """获取组件唯一标识"""
        return self._adapter_id
    
    @property
    def component_name(self) -> str:
        """获取组件名称"""
        return self._adapter_name
    
    @property
    def lifecycle_state(self) -> LifecycleState:
        """获取当前生命周期状态"""
        return self._lifecycle_state
    
    @property
    def adapter_instance(self) -> Any:
        """获取适配器实例"""
        return self._adapter_instance
    
    def initialize(self) -> bool:
        """
        初始化组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self._lifecycle_state = LifecycleState.INITIALIZING
            
            # 如果适配器有initialize方法，调用它
            if hasattr(self._adapter_instance, 'initialize'):
                result = self._adapter_instance.initialize()
                if result:
                    self._lifecycle_state = LifecycleState.INITIALIZED
                    return True
                else:
                    self._lifecycle_state = LifecycleState.ERROR
                    return False
            
            # 如果没有initialize方法，直接标记为已初始化
            self._lifecycle_state = LifecycleState.INITIALIZED
            return True
            
        except Exception as e:
            self._logger.error(
                f"适配器 {self._adapter_id} 初始化失败: {e}",
                exc_info=True
            )
            self._lifecycle_state = LifecycleState.ERROR
            return False
    
    def start(self) -> bool:
        """
        启动组件
        
        Returns:
            bool: 启动是否成功
        """
        try:
            self._lifecycle_state = LifecycleState.STARTING
            
            # 如果适配器有start方法，调用它
            if hasattr(self._adapter_instance, 'start'):
                result = self._adapter_instance.start()
                if result:
                    self._lifecycle_state = LifecycleState.RUNNING
                    return True
                else:
                    self._lifecycle_state = LifecycleState.ERROR
                    return False
            
            # 如果没有start方法，直接标记为运行中
            self._lifecycle_state = LifecycleState.RUNNING
            return True
            
        except Exception as e:
            self._logger.error(
                f"适配器 {self._adapter_id} 启动失败: {e}",
                exc_info=True
            )
            self._lifecycle_state = LifecycleState.ERROR
            return False
    
    def stop(self) -> bool:
        """
        停止组件
        
        Returns:
            bool: 停止是否成功
        """
        try:
            self._lifecycle_state = LifecycleState.STOPPING
            
            # 如果适配器有stop方法，调用它
            if hasattr(self._adapter_instance, 'stop'):
                result = self._adapter_instance.stop()
                if result:
                    self._lifecycle_state = LifecycleState.STOPPED
                    return True
                else:
                    self._lifecycle_state = LifecycleState.ERROR
                    return False
            
            # 如果没有stop方法，直接标记为已停止
            self._lifecycle_state = LifecycleState.STOPPED
            return True
            
        except Exception as e:
            self._logger.error(
                f"适配器 {self._adapter_id} 停止失败: {e}",
                exc_info=True
            )
            self._lifecycle_state = LifecycleState.ERROR
            return False
    
    def get_dependencies(self) -> List[str]:
        """
        获取组件依赖的其他组件ID列表
        
        Returns:
            List[str]: 依赖的组件ID列表
        """
        return self._dependencies.copy()