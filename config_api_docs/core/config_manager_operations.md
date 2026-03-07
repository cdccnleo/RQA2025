# config_manager_operations

**文件路径**: `core\config_manager_operations.py`

## 模块描述

配置管理器操作功能 (拆分自unified_manager.py)

包含配置操作的所有方法：验证、监听、增强功能等

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.config_manager_storage import UnifiedConfigManagerWithStorage
from infrastructure.config.core.common_methods import ConfigCommonMethods
import re
```

## 类

### UnifiedConfigManagerWithOperations

带完整操作功能的配置管理器

**继承**: UnifiedConfigManagerWithStorage

**方法**:

- `validate_config`
- `_validate_with_rules`
- `get_with_fallback`
- `set_with_validation`
- `batch_update`
- ... 等5个方法

