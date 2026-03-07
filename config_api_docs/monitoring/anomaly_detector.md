# anomaly_detector

**文件路径**: `monitoring\anomaly_detector.py`

## 模块描述

异常检测功能

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.common_mixins import ConfigComponentMixin
```

## 类

### AnomalyDetector

异常检测器

**继承**: ConfigComponentMixin

**方法**:

- `__init__`
- `update_baseline`
- `detect_anomaly`

