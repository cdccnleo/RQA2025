# yaml_loader

**文件路径**: `loaders\yaml_loader.py`

## 模块描述

基础设施层 - 配置管理组件

yaml_loader 模块

YAML配置加载策略，支持YAML格式的配置文件加载

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import Path
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Tuple
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import logging
from infrastructure.config.interfaces.unified_interface import ConfigLoaderStrategy
from infrastructure.config.config_exceptions import ConfigLoadError
# ... 等7个导入
```

## 类

### YAMLLoader

YAML配置加载策略

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `_check_yaml_availability`
- `load`
- `get_last_metadata`
- `batch_load`
- ... 等9个方法

