# config_manager_storage

**文件路径**: `core\config_manager_storage.py`

## 模块描述

配置管理器存储功能 (拆分自unified_manager.py)

包含存储相关的所有方法：加载、保存、导出、导入等

## 导入语句

```python
import yaml
from infrastructure.config.core.imports import datetime
import shutil
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import os
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
# ... 等5个导入
```

## 类

### UnifiedConfigManagerWithStorage

带存储功能的配置管理器

**继承**: <ast.Call object at 0x00000225901A3F10>

**方法**:

- `get_section`
- `load_config`
- `save_config`
- `get_all_sections`
- `reload_config`
- ... 等5个方法

## 函数

### _get_unified_config_manager

延迟获取UnifiedConfigManager类

