# enhanced_config_validator

**文件路径**: `utils\enhanced_config_validator.py`

## 模块描述

增强版配置验证器
提供配置项的有效性验证和业务规则校验

## 导入语句

```python
import re
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import Union
# ... 等4个导入
```

## 类

### ValidationLevel

验证级别

**继承**: Enum

### ValidationError

验证错误

### EnhancedConfigValidator

增强版配置验证器

**方法**:

- `__init__`
- `validate_database_config`
- `validate_api_config`
- `validate_logging_config`
- `validate_security_config`
- ... 等5个方法

## 函数

### get_enhanced_config_validator

获取全局增强版配置验证器

**返回值**: `EnhancedConfigValidator`

### validate_config_file

验证配置文件

**参数**:

- `config_file: str`

**返回值**: `<ast.Subscript object at 0x000002259030F430>`

