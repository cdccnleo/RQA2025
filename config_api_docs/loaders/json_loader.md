# json_loader

**文件路径**: `loaders\json_loader.py`

## 模块描述

基础设施层 - 配置管理组件

json_loader 模块

配置管理相关的文件
提供配置管理相关的功能实现。

## 导入语句

```python
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import Path
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Tuple
from infrastructure.config.core.imports import List
from infrastructure.config.interfaces.unified_interface import ConfigLoaderStrategy
from infrastructure.config.config_exceptions import ConfigLoadError
```

## 类

### JSONLoader

JSON配置加载策略，支持单文件和批量加载

**继承**: ConfigLoaderStrategy

**方法**:

- `load`
- `batch_load`
- `can_load`
- `get_supported_extensions`

