# configstorage

**文件路径**: `storage\types\configstorage.py`

## 模块描述

配置文件存储相关类

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Union
from infrastructure.config.core.imports import dataclass
from infrastructure.config.core.imports import field
from infrastructure.config.core.imports import Enum
from infrastructure.config.storage.types.storagetype import StorageType
```

## 类

### ConfigStorage

配置存储（向后兼容简单接口）

**继承**: FileConfigStorage

**方法**:

- `__init__`
- `set_config`
- `get_config`
- `list_configs`

