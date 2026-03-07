# dashboard_collectors

**文件路径**: `monitoring\dashboard_collectors.py`

## 模块描述

监控面板数据收集器

实现各种指标数据的收集和管理

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from abc import ABC
from abc import abstractmethod
from infrastructure.config.monitoring.dashboard_models import MetricValue
from infrastructure.config.monitoring.dashboard_models import Metric
# ... 等6个导入
```

## 类

### MetricsCollector

指标收集器基类

**继承**: ABC

**方法**:

- `__init__`
- `collect_system_metrics`
- `collect_config_metrics`
- `start_collection`
- `stop_collection`
- ... 等5个方法

### InMemoryMetricsCollector

内存指标收集器实现

**继承**: MetricsCollector

**方法**:

- `__init__`
- `collect_system_metrics`
- `collect_config_metrics`
- `record_operation`
- `add_custom_metric`
- ... 等2个方法

