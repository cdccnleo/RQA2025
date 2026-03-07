# cloud_auto_scaling

**文件路径**: `environment\cloud_auto_scaling.py`

## 模块描述

云原生自动伸缩管理器

实现基于指标的自动伸缩功能，支持CPU、内存和自定义指标

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
from infrastructure.config.environment.cloud_native_configs import AutoScalingConfig
from infrastructure.config.environment.cloud_native_configs import ScalingPolicy
```

## 类

### AutoScalingManager

自动伸缩管理器

**方法**:

- `__init__`
- `should_scale_up`
- `should_scale_down`
- `_check_cpu_scale_up`
- `_check_cpu_scale_down`
- ... 等20个方法

