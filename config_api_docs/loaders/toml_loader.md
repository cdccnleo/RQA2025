# toml_loader

**文件路径**: `loaders\toml_loader.py`

## 模块描述

基础设施层 - 配置管理组件

toml_loader 模块

TOML配置加载策略，支持TOML格式的配置文件加载

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import Path
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Tuple
from infrastructure.config.core.imports import Optional
from infrastructure.config.interfaces.unified_interface import ConfigLoaderStrategy
from infrastructure.config.config_exceptions import ConfigLoadError
from infrastructure.config.core.imports import logging
# ... 等5个导入
```

## 类

### TOMLLoader

TOML配置加载策略

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `_check_toml_availability`
- `load`
- `can_load`
- `get_supported_extensions`
- ... 等12个方法

