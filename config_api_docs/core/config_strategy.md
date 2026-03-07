# config_strategy

**文件路径**: `core\config_strategy.py`

## 导入语句

```python
from infrastructure.config.core.strategy_base import *
from infrastructure.config.core.strategy_loaders import *
from infrastructure.config.core.strategy_manager import *
from infrastructure.config.core.imports import logging
from infrastructure.config.config_exceptions import ConfigLoadError
from infrastructure.config.config_exceptions import ConfigValidationError
from infrastructure.config.config_exceptions import ConfigError
```

## 类

### ConfigValidatorStrategy

配置验证器策略（向后兼容性）

**继承**: IConfigStrategy, ABC

**方法**:

- `__init__`
- `strategy_type`
- `name`
- `is_enabled`
- `get_priority`
- ... 等3个方法

### ConfigProviderStrategy

配置提供者策略（向后兼容性）

**继承**: IConfigStrategy, ABC

**方法**:

- `__init__`
- `strategy_type`
- `name`
- `is_enabled`
- `get_priority`
- ... 等3个方法

