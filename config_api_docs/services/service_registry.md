# service_registry

**文件路径**: `services\service_registry.py`

## 导入语句

```python
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.services.container import get_container
from infrastructure.config.services.container import register
from infrastructure.config.services.container import Lifecycle
from infrastructure.config.core.config_manager_complete import UnifiedConfigManager
from infrastructure.config.database.unified_database_manager import UnifiedDatabaseManager
from infrastructure.cache.memory_cache_manager import MemoryCacheManager
from infrastructure.cache.disk_cache_manager import DiskCacheManager
# ... 等6个导入
```

## 类

### InfrastructureServiceRegistry

service_registry - 配置管理

职责说明：
负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

核心职责：
- 配置文件的读取和解析
- 配置参数的验证
- 配置的热重载
- 配置的分发和同步
- 环境变量管理
- 配置加密和安全

相关接口：
- IConfigComponent
- IConfigManager
- IConfigValidator

**方法**:

- `__init__`
- `register_all_services`
- `_register_config_services`
- `_register_database_services`
- `_register_cache_services`
- ... 等8个方法

## 函数

### get_service_registry

获取全局服务注册器

**返回值**: `InfrastructureServiceRegistry`

### register_infrastructure_services

注册所有基础设施层服务

**返回值**: `<ast.Constant object at 0x000002258FD4D9A0>`

### get_service

获取服务实例

**参数**:

- `service_type: type`

**返回值**: `Any`

### has_service

检查服务是否已注册

**参数**:

- `service_type: type`

**返回值**: `bool`

