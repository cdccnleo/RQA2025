# config_monitor

**文件路径**: `config_monitor.py`

## 模块描述

配置监控器
监控配置变更和性能指标

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import datetime
from infrastructure.config.config_event import ConfigChangeEvent
```

## 类

### ConfigMonitor

配置监控器

**方法**:

- `__init__`
- `add_listener`
- `remove_listener`
- `record_config_change`
- `get_recent_changes`
- ... 等2个方法

