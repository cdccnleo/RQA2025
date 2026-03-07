# typed_config

**文件路径**: `core\typed_config.py`

## 模块描述

类型安全的配置接口模块

提供了以下功能：
- 基于类型注解的配置访问
- 自动类型转换
- 配置值缓存
- 默认值支持

## 导入语句

```python
from typing import Any
from typing import Generic
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union
from typing import get_type_hints
import inspect
from dataclasses import dataclass
from dataclasses import field
# ... 等5个导入
```

## 类

### TypedConfigValue

类型安全的配置值

**继承**: <ast.Subscript object at 0x000002258FD63520>

**方法**:

- `get`
- `_convert_value`
- `_convert_to_type`

### TypedConfigBase

类型安全的配置基类

**方法**:

- `__init__`
- `_initialize_config_values`
- `__getattribute__`

### TypedConfigSimple

类型化配置（向后兼容简单接口）

**方法**:

- `__init__`
- `set_typed`
- `get_typed`
- `validate_type`

### TypedConfiguration

TypedConfiguration别名，保持向后兼容性

**继承**: TypedConfigBase

## 函数

### config_value

创建类型安全的配置值

用法:

class MyConfig(TypedConfigBase):

server_port: int = config_value("server.port", 8080)
debug_mode: bool = config_value("debug", False)

**参数**:

- `key: str`
- `default: Any`
- `description: str = `

**返回值**: `Any`

### get_typed_config

获取类型安全的配置实例（带缓存）

**参数**:

- `config_class: <ast.Subscript object at 0x000002258FD74070>`
- `config_manager`
- `env = default`

**返回值**: `TypedConfigBase`

