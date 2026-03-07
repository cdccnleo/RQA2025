#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局组件生命周期管理器

负责管理所有组件的生命周期，包括初始化、启动、停止等操作
支持依赖关系管理和启动顺序控制
"""

import logging
import threading
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict, deque
from datetime import datetime

from .interfaces import (
    ILifecycleComponent,
    ILifecycleManager,
    LifecycleState,
    LifecycleEvent
)


class ComponentLifecycleManager(ILifecycleManager):
    """
    全局组件生命周期管理器（单例模式）
    
    职责：
    - 统一管理所有组件的生命周期
    - 处理组件依赖关系
    - 控制组件启动和停止顺序
    - 记录生命周期事件
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
        """初始化生命周期管理器"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._components: Dict[str, ILifecycleComponent] = {}
        self._dependencies: Dict[str, List[str]] = defaultdict(list)  # component_id -> [dependency_ids]
        self._reverse_dependencies: Dict[str, List[str]] = defaultdict(list)  # component_id -> [dependent_ids]
        self._lifecycle_events: List[LifecycleEvent] = []
        self._max_events = 1000  # 最多保存1000个事件
        self._logger = logging.getLogger(__name__)
        self._manager_lock = threading.RLock()
        self._initialized = True
        
        self._logger.info("组件生命周期管理器初始化完成")
    
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
        with self._manager_lock:
            component_id = component.component_id
            
            if component_id in self._components:
                self._logger.warning(f"组件 {component_id} 已注册，跳过重复注册")
                return False
            
            self._components[component_id] = component
            
            if dependencies:
                self._dependencies[component_id] = dependencies.copy()
                # 更新反向依赖关系
                for dep_id in dependencies:
                    self._reverse_dependencies[dep_id].append(component_id)
            
            self._logger.info(f"注册组件: {component_id} (依赖: {dependencies or []})")
            self._record_event(
                component_id,
                component.component_name,
                'registered',
                {'dependencies': dependencies or []}
            )
            
            return True
    
    def unregister_component(self, component_id: str) -> bool:
        """
        取消注册组件
        
        Args:
            component_id: 组件ID
            
        Returns:
            bool: 取消注册是否成功
        """
        with self._manager_lock:
            if component_id not in self._components:
                self._logger.warning(f"组件 {component_id} 未注册，无法取消注册")
                return False
            
            component = self._components[component_id]
            
            # 清理依赖关系
            if component_id in self._dependencies:
                for dep_id in self._dependencies[component_id]:
                    if component_id in self._reverse_dependencies.get(dep_id, []):
                        self._reverse_dependencies[dep_id].remove(component_id)
                del self._dependencies[component_id]
            
            # 清理反向依赖关系
            if component_id in self._reverse_dependencies:
                for dependent_id in self._reverse_dependencies[component_id]:
                    if component_id in self._dependencies.get(dependent_id, []):
                        self._dependencies[dependent_id].remove(component_id)
                del self._reverse_dependencies[component_id]
            
            del self._components[component_id]
            
            self._logger.info(f"取消注册组件: {component_id}")
            self._record_event(
                component_id,
                component.component_name,
                'unregistered',
                {}
            )
            
            return True
    
    def get_component(self, component_id: str) -> Optional[ILifecycleComponent]:
        """
        获取组件实例
        
        Args:
            component_id: 组件ID
            
        Returns:
            组件实例，如果不存在则返回None
        """
        with self._manager_lock:
            return self._components.get(component_id)
    
    def initialize_all(self) -> Dict[str, bool]:
        """
        初始化所有已注册的组件（按依赖顺序）
        
        Returns:
            Dict[str, bool]: 组件ID到初始化结果的映射
        """
        with self._manager_lock:
            results = {}
            initialized = set()
            
            # 拓扑排序，确定初始化顺序
            init_order = self._topological_sort()
            
            self._logger.info(f"开始初始化 {len(init_order)} 个组件")
            
            for component_id in init_order:
                component = self._components[component_id]
                
                # 检查依赖是否都已初始化
                deps = self._dependencies.get(component_id, [])
                if not all(dep_id in initialized for dep_id in deps):
                    self._logger.warning(
                        f"组件 {component_id} 的依赖未完全初始化，跳过"
                    )
                    results[component_id] = False
                    continue
                
                try:
                    success = component.initialize()
                    results[component_id] = success
                    
                    if success:
                        initialized.add(component_id)
                        self._record_event(
                            component_id,
                            component.component_name,
                            'initialized',
                            {}
                        )
                        self._logger.info(f"组件 {component_id} 初始化成功")
                    else:
                        self._logger.error(f"组件 {component_id} 初始化失败")
                        self._record_event(
                            component_id,
                            component.component_name,
                            'initialize_failed',
                            {}
                        )
                except Exception as e:
                    self._logger.error(
                        f"组件 {component_id} 初始化异常: {e}",
                        exc_info=True
                    )
                    results[component_id] = False
                    self._record_event(
                        component_id,
                        component.component_name,
                        'initialize_error',
                        {'error': str(e)}
                    )
            
            return results
    
    def start_all(self) -> Dict[str, bool]:
        """
        启动所有已注册的组件（按依赖顺序）
        
        Returns:
            Dict[str, bool]: 组件ID到启动结果的映射
        """
        with self._manager_lock:
            results = {}
            started = set()
            
            # 拓扑排序，确定启动顺序
            start_order = self._topological_sort()
            
            self._logger.info(f"开始启动 {len(start_order)} 个组件")
            
            for component_id in start_order:
                component = self._components[component_id]
                
                # 检查依赖是否都已启动
                deps = self._dependencies.get(component_id, [])
                if not all(dep_id in started for dep_id in deps):
                    self._logger.warning(
                        f"组件 {component_id} 的依赖未完全启动，跳过"
                    )
                    results[component_id] = False
                    continue
                
                try:
                    success = component.start()
                    results[component_id] = success
                    
                    if success:
                        started.add(component_id)
                        self._record_event(
                            component_id,
                            component.component_name,
                            'started',
                            {}
                        )
                        self._logger.info(f"组件 {component_id} 启动成功")
                    else:
                        self._logger.error(f"组件 {component_id} 启动失败")
                        self._record_event(
                            component_id,
                            component.component_name,
                            'start_failed',
                            {}
                        )
                except Exception as e:
                    self._logger.error(
                        f"组件 {component_id} 启动异常: {e}",
                        exc_info=True
                    )
                    results[component_id] = False
                    self._record_event(
                        component_id,
                        component.component_name,
                        'start_error',
                        {'error': str(e)}
                    )
            
            return results
    
    def stop_all(self) -> Dict[str, bool]:
        """
        停止所有已注册的组件（按依赖顺序，逆序）
        
        Returns:
            Dict[str, bool]: 组件ID到停止结果的映射
        """
        with self._manager_lock:
            results = {}
            
            # 逆拓扑排序，确定停止顺序
            stop_order = list(reversed(self._topological_sort()))
            
            self._logger.info(f"开始停止 {len(stop_order)} 个组件")
            
            for component_id in stop_order:
                component = self._components[component_id]
                
                try:
                    success = component.stop()
                    results[component_id] = success
                    
                    if success:
                        self._record_event(
                            component_id,
                            component.component_name,
                            'stopped',
                            {}
                        )
                        self._logger.info(f"组件 {component_id} 停止成功")
                    else:
                        self._logger.error(f"组件 {component_id} 停止失败")
                        self._record_event(
                            component_id,
                            component.component_name,
                            'stop_failed',
                            {}
                        )
                except Exception as e:
                    self._logger.error(
                        f"组件 {component_id} 停止异常: {e}",
                        exc_info=True
                    )
                    results[component_id] = False
                    self._record_event(
                        component_id,
                        component.component_name,
                        'stop_error',
                        {'error': str(e)}
                    )
            
            return results
    
    def get_component_state(self, component_id: str) -> Optional[LifecycleState]:
        """
        获取组件的生命周期状态
        
        Args:
            component_id: 组件ID
            
        Returns:
            生命周期状态，如果组件不存在则返回None
        """
        with self._manager_lock:
            component = self._components.get(component_id)
            if component is None:
                return None
            return component.lifecycle_state
    
    def get_all_components(self) -> Dict[str, ILifecycleComponent]:
        """
        获取所有已注册的组件
        
        Returns:
            组件ID到组件实例的映射
        """
        with self._manager_lock:
            return self._components.copy()
    
    def get_lifecycle_events(
        self,
        component_id: Optional[str] = None,
        limit: int = 100
    ) -> List[LifecycleEvent]:
        """
        获取生命周期事件
        
        Args:
            component_id: 组件ID，如果为None则返回所有事件
            limit: 最多返回的事件数量
            
        Returns:
            生命周期事件列表
        """
        with self._manager_lock:
            events = self._lifecycle_events
            
            if component_id:
                events = [e for e in events if e.component_id == component_id]
            
            return events[-limit:]
    
    def _topological_sort(self) -> List[str]:
        """
        拓扑排序，确定组件的初始化/启动顺序
        
        Returns:
            排序后的组件ID列表
        """
        # Kahn算法实现拓扑排序
        in_degree = {cid: len(self._dependencies.get(cid, [])) for cid in self._components.keys()}
        queue = deque([cid for cid, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            component_id = queue.popleft()
            result.append(component_id)
            
            # 更新依赖此组件的其他组件的入度
            for dependent_id in self._reverse_dependencies.get(component_id, []):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        # 检查是否有循环依赖
        if len(result) != len(self._components):
            remaining = set(self._components.keys()) - set(result)
            self._logger.warning(f"检测到可能的循环依赖，未排序的组件: {remaining}")
            # 将未排序的组件添加到末尾
            result.extend(remaining)
        
        return result
    
    def _record_event(
        self,
        component_id: str,
        component_name: str,
        event_type: str,
        metadata: Dict[str, Any]
    ):
        """记录生命周期事件"""
        event = LifecycleEvent(
            component_id=component_id,
            component_name=component_name,
            event_type=event_type,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self._lifecycle_events.append(event)
        
        # 限制事件数量
        if len(self._lifecycle_events) > self._max_events:
            self._lifecycle_events = self._lifecycle_events[-self._max_events:]


# 全局单例访问函数
def get_lifecycle_manager() -> ComponentLifecycleManager:
    """
    获取全局组件生命周期管理器实例
    
    Returns:
        ComponentLifecycleManager: 生命周期管理器实例
    """
    return ComponentLifecycleManager()