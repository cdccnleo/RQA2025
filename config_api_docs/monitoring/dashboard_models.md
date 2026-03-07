# dashboard_models

**文件路径**: `monitoring\dashboard_models.py`

## 模块描述

监控面板数据模型

定义监控系统的数据结构和枚举

## 导入语句

```python
from infrastructure.config.core.imports import datetime
from infrastructure.config.core.imports import timedelta
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Union
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
```

## 类

### MetricType

指标类型枚举

**继承**: Enum

### AlertSeverity

告警严重程度

**继承**: Enum

### AlertStatus

告警状态

**继承**: Enum

### MetricValue

指标值

### Metric

指标定义

### Alert

告警定义

### MonitoringConfig

监控配置

### PerformanceMetrics

性能指标 (兼容原有接口)

### SystemResources

系统资源使用情况

### ConfigOperationStats

配置操作统计

**方法**:

- `add_metric`
- `get_success_rate`
- `to_dict`

