# provider

**文件路径**: `tools\provider.py`

## 导入语句

```python
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import os
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import logging
from abc import abstractmethod
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.interfaces.unified_interface import IConfigProvider
```

## 类

### ConfigProvider

provider - 配置管理

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

**继承**: IConfigProvider

**方法**:

- `load`
- `save`
- `get_default`

### DefaultConfigProvider

默认配置提供者

**继承**: ConfigProvider

**方法**:

- `__init__`
- `load`
- `save`
- `get_default`
- `_load_from_file`
- ... 等5个方法

