# dashboard_manager

**文件路径**: `monitoring\dashboard_manager.py`

## 模块描述

监控面板统一管理器

整合所有监控功能的核心管理器

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Union
from infrastructure.config.monitoring.dashboard_models import MonitoringConfig
from infrastructure.config.monitoring.dashboard_models import PerformanceMetrics
from infrastructure.config.monitoring.dashboard_models import SystemResources
# ... 等6个导入
```

## 类

### UnifiedMonitoringManager

统一监控管理器

**方法**:

- `__init__`
- `set_metrics_collector`
- `set_alert_manager`
- `start_monitoring`
- `stop_monitoring`
- ... 等8个方法

