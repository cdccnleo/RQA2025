# iconfigstorage

**文件路径**: `storage\types\iconfigstorage.py`

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
from infrastructure.config.core.imports import ABC
from infrastructure.config.core.imports import abstractmethod
# ... 等3个导入
```

## 类

### IConfigStorage

配置存储接口

**继承**: ABC

**方法**:

- `get`
- `set`
- `delete`
- `exists`
- `list_keys`
- ... 等2个方法

