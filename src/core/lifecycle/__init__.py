#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组件生命周期管理模块

提供统一的组件生命周期管理接口和实现
"""

from .interfaces import (
    ILifecycleComponent,
    ILifecycleManager,
    LifecycleState,
    LifecycleEvent
)
from .component_lifecycle_manager import ComponentLifecycleManager, get_lifecycle_manager

__all__ = [
    'ILifecycleComponent',
    'ILifecycleManager',
    'LifecycleState',
    'LifecycleEvent',
    'ComponentLifecycleManager',
    'get_lifecycle_manager'
]