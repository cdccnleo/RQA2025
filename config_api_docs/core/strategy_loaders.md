# strategy_loaders

**文件路径**: `core\strategy_loaders.py`

## 模块描述

配置策略加载器实现

实现各种配置加载器的具体策略

## 导入语句

```python
from infrastructure.config.core.strategy_base import ConfigLoaderStrategy
from infrastructure.config.core.strategy_base import ConfigFormat
from infrastructure.config.core.strategy_base import ConfigSourceType
from infrastructure.config.core.strategy_base import LoadResult
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import Union
# ... 等6个导入
```

## 类

### JSONConfigLoader

JSON配置加载器

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `execute`
- `can_handle_source`
- `get_supported_formats`
- `validate_source`

### EnvironmentConfigLoaderStrategy

环境变量配置加载器策略

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `execute`
- `_set_nested_value`
- `_convert_env_value`
- `can_handle_source`
- ... 等2个方法

### YAMLConfigLoader

YAML配置加载器

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `execute`
- `can_handle_source`

### TOMLConfigLoader

TOML配置加载器

**继承**: ConfigLoaderStrategy

**方法**:

- `__init__`
- `execute`
- `can_handle_source`

