# priority_manager

**文件路径**: `core\priority_manager.py`

## 模块描述

配置优先级管理器

管理不同配置源的优先级和合并策略

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from enum import Enum
from infrastructure.config.core.imports import logging
```

## 类

### ConfigPriority

配置优先级枚举

**继承**: Enum

### ConfigPriorityManager

配置优先级管理器

**方法**:

- `__init__`
- `add_config_layer`
- `_rebuild_merged_config`
- `_deep_merge`
- `get_merged_config`
- ... 等2个方法

