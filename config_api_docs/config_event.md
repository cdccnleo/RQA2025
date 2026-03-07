# config_event

**文件路径**: `config_event.py`

## 模块描述

配置事件管理
处理配置相关的各种事件

## 导入语句

```python
from infrastructure.config.core.imports import time
import uuid
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import datetime
```

## 类

### ConfigEvent

配置事件基类

**方法**:

- `__init__`
- `to_dict`

### ConfigChangeEvent

配置变更事件

**继承**: ConfigEvent

**方法**:

- `__init__`
- `_determine_change_type`

### ConfigLoadEvent

配置加载事件

**继承**: ConfigEvent

**方法**:

- `__init__`

### ConfigValidationEvent

配置验证事件

**继承**: ConfigEvent

**方法**:

- `__init__`

### ConfigReloadEvent

配置重载事件

**继承**: ConfigEvent

**方法**:

- `__init__`

### ConfigBackupEvent

配置备份事件

**继承**: ConfigEvent

**方法**:

- `__init__`

### ConfigErrorEvent

配置错误事件

**继承**: ConfigEvent

**方法**:

- `__init__`

