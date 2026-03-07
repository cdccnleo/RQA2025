# cloud_loader

**文件路径**: `loaders\cloud_loader.py`

## 模块描述

基础设施层 - 配置管理组件

cloud_loader 模块

云配置加载策略，支持AWS、Azure、Google Cloud等云服务

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
from infrastructure.config.core.imports import logging
```

## 类

### CloudLoader

云配置加载策略

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `load`
- `can_load`
- `_parse_cloud_path`
- `get_last_metadata`
- ... 等11个方法

