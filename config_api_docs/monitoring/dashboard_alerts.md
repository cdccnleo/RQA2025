# dashboard_alerts

**文件路径**: `monitoring\dashboard_alerts.py`

## 模块描述

监控面板告警管理

实现告警的创建、管理和处理

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Callable
from abc import ABC
from abc import abstractmethod
from infrastructure.config.monitoring.dashboard_models import Alert
# ... 等4个导入
```

## 类

### AlertManager

告警管理器基类

**继承**: ABC

**方法**:

- `__init__`
- `create_alert`
- `resolve_alert`
- `acknowledge_alert`
- `add_listener`
- ... 等7个方法

### InMemoryAlertManager

内存告警管理器实现

**继承**: AlertManager

**方法**:

- `__init__`
- `create_alert`
- `resolve_alert`
- `acknowledge_alert`
- `get_active_alerts_count`
- ... 等1个方法

