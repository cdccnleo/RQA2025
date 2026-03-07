# cloud_native_configs

**文件路径**: `environment\cloud_native_configs.py`

## 模块描述

云原生环境配置类

定义云原生环境相关的配置数据结构

## 导入语句

```python
from infrastructure.config.core.imports import dataclass
from infrastructure.config.core.imports import field
from infrastructure.config.core.imports import Enum
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
```

## 类

### ServiceMeshType

服务网格类型

**继承**: Enum

### CloudProvider

云提供商

**继承**: Enum

### ScalingPolicy

自动伸缩策略

**继承**: Enum

### ServiceMeshConfig

服务网格配置

**方法**:

- `__post_init__`

### MultiCloudConfig

多云配置

**方法**:

- `__post_init__`

### AutoScalingConfig

自动伸缩配置

**方法**:

- `__post_init__`

### CloudNativeMonitoringConfig

云原生监控配置

**方法**:

- `__post_init__`

