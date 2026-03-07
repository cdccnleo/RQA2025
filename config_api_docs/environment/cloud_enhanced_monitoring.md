# cloud_enhanced_monitoring

**文件路径**: `environment\cloud_enhanced_monitoring.py`

## 模块描述

云原生增强监控管理器

提供高级监控功能，包括指标聚合、告警管理和性能分析

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import datetime
from infrastructure.config.environment.cloud_native_configs import CloudNativeMonitoringConfig
from infrastructure.config.monitoring.dashboard_models import MetricValue
# ... 等5个导入
```

## 类

### EnhancedMonitoringManager

增强监控管理器

**方法**:

- `__init__`
- `_initialize_components`
- `add_custom_metric`
- `get_metric_statistics`
- `define_alert_pattern`
- ... 等7个方法

### MetricsAggregator

指标聚合器

**方法**:

- `__init__`
- `aggregate_metric`
- `_percentile`

### AlertCorrelator

告警关联器

**方法**:

- `__init__`
- `correlate_alert`
- `get_alert_groups`

### PerformanceAnalyzer

性能分析器

**方法**:

- `__init__`
- `detect_anomaly`

