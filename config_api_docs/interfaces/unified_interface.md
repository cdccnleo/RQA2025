# unified_interface

**文件路径**: `interfaces\unified_interface.py`

## 模块描述

统一接口定义文件
提供配置管理相关的标准接口和枚举

## 导入语句

```python
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Callable
from typing import Type
from typing import Tuple
from abc import ABC
from abc import abstractmethod
# ... 等3个导入
```

## 类

### CachePolicy

缓存策略枚举

**继承**: Enum

### ServiceStatus

服务状态枚举

**继承**: Enum

### IConfigManagerComponent

配置管理器接口

**继承**: ABC

**方法**:

- `get`
- `set`
- `update`
- `watch`
- `reload`
- ... 等1个方法

### IConfigManagerFactoryComponent

配置管理器工厂接口

**继承**: ABC

**方法**:

- `create_manager`
- `register_manager`
- `get_available_managers`

### StrategyType

策略类型枚举

**继承**: Enum

### ConfigFormat

配置格式枚举

**继承**: Enum

### ConfigSourceType

配置源类型枚举

**继承**: Enum

### IConfigStrategy

配置策略接口

**继承**: ABC

**方法**:

- `name`
- `type`
- `execute`
- `can_handle`

### ConfigLoaderStrategy

配置加载策略基类

**继承**: IConfigStrategy, ABC

**方法**:

- `__init__`
- `name`
- `type`
- `load`
- `can_handle_source`
- ... 等1个方法

