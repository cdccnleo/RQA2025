# paths

**文件路径**: `tools\paths.py`

## 导入语句

```python
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import Path
from configparser import ConfigParser
from infrastructure.config.core.imports import os
from src.utils.logger import get_logger
```

## 类

### PathConfig

paths - 配置管理

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
管理项目路径配置，实现动态配置加载与目录创建

    属性：
        BASE_DIR (Path): 项目根目录
        DATA_DIR (Path): 数据存储目录
        MODEL_DIR (Path): 模型存储目录
        LOG_DIR (Path): 日志目录
        CACHE_DIR (Path): 缓存目录
    

**方法**:

- `__init__`
- `_load_config`
- `_create_directories`
- `get_model_path`
- `get_cache_file`

### ConfigPaths

配置路径管理器

**方法**:

- `__init__`
- `_ensure_directories`
- `get_config_file`
- `get_data_file`
- `get_log_file`
- ... 等2个方法

## 函数

### get_path_config

获取路径配置实例（延迟初始化）

**返回值**: `<ast.Constant object at 0x000002258FD74F10>`

### get_config_path

获取配置文件路径"

返回:
    Path: 配置文件的绝对路径

**返回值**: `Path`

