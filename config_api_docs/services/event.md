# event

**文件路径**: `services\event.py`

## 模块描述

轻量级事件系统实现
提供配置变更通知的核心机制

## 导入语句

```python
from infrastructure.config.core.imports import os
from infrastructure.config.core.imports import threading
from typing import Callable
from typing import Dict
from typing import List
from typing import Any
from typing import Optional
import uuid
```

## 类

### ConfigEvents

配置管理相关事件类型

### Event

事件类

**方法**:

- `__init__`

### EventBus

事件总线类

**方法**:

- `__init__`
- `subscribe`
- `unsubscribe`
- `publish`

### EventSystem

线程安全的事件发布 - 订阅系统

**方法**:

- `__init__`
- `subscribe`
- `unsubscribe`
- `publish`
- `get_default`
- ... 等2个方法

