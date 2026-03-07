# config_factory_core

**文件路径**: `core\config_factory_core.py`

## 模块描述

统一配置工厂核心 (拆分自factory.py)

包含UnifiedConfigFactory核心工厂逻辑

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Type
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import logging
from abc import ABC
from abc import abstractmethod
from infrastructure.config.core.config_manager_complete import UnifiedConfigManager
from infrastructure.config.interfaces.unified_interface import IConfigManagerComponent
# ... 等1个导入
```

## 类

### UnifiedConfigFactory

统一配置工厂

整合了所有配置工厂的功能：
- 配置管理器注册和创建
- 实例缓存管理
- 生命周期管理
- 类型验证
- 性能监控

**继承**: IConfigManagerFactory

**方法**:

- `__init__`
- `_register_default_managers`
- `register_manager`
- `register_provider`
- `unregister_manager`
- ... 等10个方法

