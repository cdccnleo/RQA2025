# config_merger

**文件路径**: `mergers\config_merger.py`

## 模块描述

基础设施层 - 配置管理组件

config_merger 模块

高级配置合并器，支持多种合并策略和冲突解决

## 导入语句

```python
import copy
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import Union
from enum import Enum
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import time
```

## 类

### MergeStrategy

合并策略枚举

**继承**: Enum

### ConflictResolution

冲突解决策略枚举

**继承**: Enum

### ConfigMerger

高级配置合并器

**方法**:

- `__init__`
- `merge`
- `_merge_overwrite`
- `_merge_shallow`
- `_merge_deep`
- ... 等9个方法

### HierarchicalConfigMerger

层次化配置合并器

**继承**: ConfigMerger

**方法**:

- `__init__`
- `merge_hierarchical`

### EnvironmentAwareConfigMerger

环境感知配置合并器

**继承**: ConfigMerger

**方法**:

- `__init__`
- `merge_with_environment`

### ProfileBasedConfigMerger

基于配置文件的配置合并器

**继承**: ConfigMerger

**方法**:

- `__init__`
- `merge_with_profiles`

## 函数

### merge_configs

便捷的配置合并函数

Args:
    target: 目标配置
    source: 源配置
    strategy: 合并策略

Returns:
    合并后的配置

**参数**:

- `target: <ast.Subscript object at 0x00000225901C5B20>`
- `source: <ast.Subscript object at 0x00000225901C5730>`
- `strategy: MergeStrategy`

**返回值**: `<ast.Subscript object at 0x00000225901F5280>`

### merge_hierarchical_configs

便捷的层次化配置合并函数

Args:
    configs: 配置源字典
    priority_order: 优先级顺序

Returns:
    合并后的配置

**参数**:

- `configs: <ast.Subscript object at 0x00000225901F5220>`
- `priority_order: <ast.Subscript object at 0x000002258FD934F0>`

**返回值**: `<ast.Subscript object at 0x00000225901E6D90>`

### merge_environment_configs

便捷的环境配置合并函数

Args:
    base_config: 基础配置
    env_configs: 环境配置字典
    environment: 当前环境

Returns:
    合并后的配置

**参数**:

- `base_config: <ast.Subscript object at 0x00000225901DA220>`
- `env_configs: <ast.Subscript object at 0x00000225901DA6D0>`
- `environment: str`

**返回值**: `<ast.Subscript object at 0x00000225901DA880>`

