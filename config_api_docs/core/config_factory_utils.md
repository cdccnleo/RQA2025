# config_factory_utils

**文件路径**: `core\config_factory_utils.py`

## 模块描述

配置工厂工具函数 (拆分自factory.py)

包含全局工厂实例和便捷函数

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Type
from infrastructure.config.core.imports import List
from infrastructure.config.core.config_factory_core import UnifiedConfigFactory
from infrastructure.config.interfaces.unified_interface import IConfigManager
from infrastructure.config.core.imports import logging
```

## 函数

### get_config_factory

获取全局配置工厂实例

Returns:
UnifiedConfigFactory: 配置工厂实例

**返回值**: `UnifiedConfigFactory`

### reset_global_factory

重置全局工厂实例

### create_config_manager

便捷的配置管理器创建函数

Args:
manager_type: 管理器类型
**kwargs: 创建参数

Returns:
IConfigManager: 配置管理器实例

**参数**:

- `manager_type: str = unified`
- `**kwargs`

**返回值**: `IConfigManager`

### get_available_config_types

获取所有可用的配置管理器类型

Returns:
List[str]: 类型列表

**返回值**: `<ast.Subscript object at 0x000002258FD803D0>`

### get_factory_stats

获取工厂统计信息

Returns:
Dict[str, Any]: 统计信息

**返回值**: `<ast.Subscript object at 0x000002258FD800D0>`

### get_config_manager

获取配置管理器实例（便捷函数）

**参数**:

- `manager_type: str = unified`
- `**kwargs`

**返回值**: `IConfigManager`

### register_config_manager

注册配置管理器（便捷函数）

**参数**:

- `name: str`
- `manager_class: <ast.Subscript object at 0x000002258FD4D1C0>`

**返回值**: `<ast.Constant object at 0x000002258FD4DCD0>`

