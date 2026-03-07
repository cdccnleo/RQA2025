# config_service

**文件路径**: `core\config_service.py`

## 模块描述

统一配置服务 (优化版)

整合所有配置服务功能，提供统一的配置服务框架
合并了config_service.py, config_service_components.py, unified_service.py的功能

支持:
- 配置加载和管理
- 缓存服务集成
- 验证器集成
- 热重载功能
- 服务组件化架构
- 性能监控
- 错误处理和恢复

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Tuple
from infrastructure.config.core.imports import Callable
# ... 等7个导入
```

## 类

### ServiceHealth

服务健康状态

**继承**: Enum

### IConfigServiceComponent

配置服务组件接口

**继承**: ABC

**方法**:

- `initialize`
- `start`
- `stop`
- `get_status`
- `get_health`
- ... 等1个方法

### IConfigService

配置服务接口

**继承**: ABC

**方法**:

- `load_config`
- `reload_config`
- `get_config`
- `set_config`
- `validate_config`
- ... 等1个方法

### UnifiedConfigService

统一配置服务

整合所有配置服务功能：
- 配置加载和管理
- 缓存集成
- 验证器集成
- 热重载
- 性能监控
- 健康检查

**继承**: IConfigService, IConfigServiceComponent

**方法**:

- `__init__`
- `initialize`
- `start`
- `stop`
- `load_config`
- ... 等11个方法

### ConfigServiceFactory

配置服务工厂

**方法**:

- `__init__`
- `create_service`
- `register_service_type`

### ConfigService

向后兼容的配置服务类

**方法**:

- `__init__`
- `register_loader`
- `register_validator`
- `load_config`
- `reload_config`
- ... 等4个方法

## 函数

### create_config_service

创建配置服务 (便捷函数)

**参数**:

- `service_type: str = unified`

**返回值**: `IConfigService`

