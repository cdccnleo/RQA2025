# database_loader

**文件路径**: `loaders\database_loader.py`

## 模块描述

基础设施层 - 配置管理组件

database_loader 模块

数据库配置加载策略，支持多种数据库的配置加载

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Tuple
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.interfaces.unified_interface import ConfigLoaderStrategy
from infrastructure.config.config_exceptions import ConfigLoadError
from infrastructure.config.config_exceptions import ConfigError
from infrastructure.config.core.imports import logging
# ... 等5个导入
```

## 类

### DatabaseLoader

数据库配置加载策略

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `load`
- `get_last_metadata`
- `can_load`
- `get_supported_extensions`
- ... 等5个方法

