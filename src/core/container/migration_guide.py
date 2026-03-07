#!/usr/bin/env python3
"""
Container组件迁移指南

本文件提供从旧组件文件迁移到新BaseComponent架构的说明

迁移状态：
- ✅ 已创建重构版本：refactored_container_components.py
- 🔄 原始文件将逐步迁移

迁移步骤：
1. 使用refactored_container_components.py中的新组件
2. 更新导入路径
3. 测试验证
4. 替换原始文件

创建时间: 2025-11-03
"""

# 向后兼容导入：从重构版本导入
from src.core.container.refactored_container_components import (
    ContainerComponent,
    FactoryComponent,
    LocatorComponent,
    RegistryComponent,
    ResolverComponent,
    ComponentFactory,
    create_container_components
)

# 保持向后兼容的别名
__all__ = [
    'ContainerComponent',
    'FactoryComponent',
    'LocatorComponent',
    'RegistryComponent',
    'ResolverComponent',
    'ComponentFactory',
    'create_container_components'
]

