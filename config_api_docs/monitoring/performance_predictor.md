# performance_predictor

**文件路径**: `monitoring\performance_predictor.py`

## 模块描述

性能预测功能

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.common_mixins import ConfigComponentMixin
```

## 类

### PerformancePredictor

性能预测器

**继承**: ConfigComponentMixin

**方法**:

- `__init__`
- `add_historical_data`
- `predict_next_value`
- `predict_trend`

