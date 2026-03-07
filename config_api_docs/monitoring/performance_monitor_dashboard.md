# performance_monitor_dashboard

**文件路径**: `monitoring\performance_monitor_dashboard.py`

## 模块描述

统一性能监控面板

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.monitoring.core import PerformanceMonitorDashboardCore
from infrastructure.config.monitoring.anomaly_detector import AnomalyDetector
from infrastructure.config.monitoring.trend_analyzer import TrendAnalyzer
from infrastructure.config.monitoring.performance_predictor import PerformancePredictor
```

## 类

### PerformanceMonitorDashboard

统一性能监控面板 - 整合所有监控功能

**方法**:

- `__init__`
- `start_monitoring`
- `stop_monitoring`
- `record_operation`
- `get_operation_stats`
- ... 等5个方法

