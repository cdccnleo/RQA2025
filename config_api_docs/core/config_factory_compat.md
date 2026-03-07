# config_factory_compat

**文件路径**: `core\config_factory_compat.py`

## 模块描述

配置工厂向后兼容层 (拆分自factory.py)

包含ConfigFactory类和向后兼容函数

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Type
from infrastructure.config.core.config_factory_core import UnifiedConfigFactory
from infrastructure.config.interfaces.unified_interface import IConfigManager
from infrastructure.config.core.imports import logging
from infrastructure.config.core.config_manager_complete import UnifiedConfigManager
```

## 类

### ConfigFactory

向后兼容的配置工厂类

保持与原有factory.py的接口兼容性

**方法**:

- `create_config_manager`
- `get_config_manager`
- `destroy_config_manager`
- `get_all_managers`
- `create_config_provider`
- ... 等2个方法

## 函数

### get_default_config_manager

获取默认配置管理器（向后兼容）

**返回值**: `IConfigManager`

### reset_default_config_manager

重置默认配置管理器（向后兼容）

