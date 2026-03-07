# enhanced_validators

**文件路径**: `validators\enhanced_validators.py`

## 模块描述

增强的配置验证器

提供更全面的配置验证功能

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
import re
from infrastructure.config.core.imports import logging
from infrastructure.config.utils.enhanced_config_validator import EnhancedConfigValidator
```

## 类

### ConfigValidationResult

配置验证结果

**方法**:

- `__init__`
- `add_error`
- `add_warning`
- `add_recommendation`

