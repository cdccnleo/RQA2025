# sync_node_manager

**文件路径**: `services\sync_node_manager.py`

## 模块描述

同步节点管理器
管理分布式配置同步的节点

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from enum import Enum
from dataclasses import dataclass
from infrastructure.config.core.imports import time
```

## 类

### SyncStatus

同步状态枚举

**继承**: Enum

### SyncNode

同步节点

**方法**:

- `__post_init__`

### SyncNodeManager

同步节点管理器

**方法**:

- `__init__`
- `register_node`
- `unregister_node`
- `get_node`
- `update_node_status`
- ... 等5个方法

