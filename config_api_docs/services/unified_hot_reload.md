# unified_hot_reload

**文件路径**: `services\unified_hot_reload.py`

## 模块描述

统一配置管理器热重载功能
提供配置文件热重载功能

## 导入语句

```python
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import Set
# ... 等2个导入
```

## 类

### UnifiedHotReload

统一配置管理器热重载功能

**方法**:

- `__init__`
- `start_hot_reload`
- `stop_hot_reload`
- `watch_file`
- `unwatch_file`
- ... 等8个方法

## 函数

### start_hot_reload

unified_hot_reload - 配置管理

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

    启动热重载(全局函数)
    

**返回值**: `bool`

### stop_hot_reload

停止热重载（全局函数）

**返回值**: `bool`

