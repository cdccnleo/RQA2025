# simple_config_factory

**文件路径**: `simple_config_factory.py`

## 模块描述

简单配置工厂
提供简单的配置管理器创建功能

## 导入语句

```python
from typing import Type
from typing import Dict
from typing import Any
from infrastructure.config.core.config_manager_complete import UnifiedConfigManager
from infrastructure.config.core.config_factory_core import UnifiedConfigFactory
```

## 类

### SimpleConfigFactory

简单配置工厂

**方法**:

- `__init__`
- `create_manager`
- `get_manager`
- `remove_manager`
- `list_managers`
- ... 等1个方法

## 函数

### get_simple_factory

获取全局简单工厂实例

**返回值**: `SimpleConfigFactory`

### create_simple_manager

创建简单配置管理器（便捷函数）

**参数**:

- `name: str = default`
- `config: <ast.Subscript object at 0x000002258FD23AF0>`

**返回值**: `UnifiedConfigManager`

### get_simple_manager

获取简单配置管理器（便捷函数）

**参数**:

- `name: str = default`

**返回值**: `UnifiedConfigManager`

