# schema

**文件路径**: `tools\schema.py`

## 导入语句

```python
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Union
from infrastructure.config.core.imports import json
from abc import ABC
from abc import abstractmethod
from enum import Enum
# ... 等2个导入
```

## 类

### SchemaConfigValidator

schema - 配置管理

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

**继承**: ABC

**方法**:

- `validate`
- `get_errors`

### SchemaValidator

基于模式的配置验证器

**继承**: SchemaConfigValidator

**方法**:

- `__init__`
- `validate`
- `get_errors`
- `_validate_object`
- `_validate_value`
- ... 等5个方法

### ConfigSchema

配置模式

**方法**:

- `__init__`
- `validate`
- `get_errors`
- `get_schema`

### ConfigSchemaRegistry

配置模式注册表

**方法**:

- `__init__`
- `register`
- `get`
- `validate`
- `list_schemas`

### ConfigType

配置类型枚举

**继承**: Enum

### ConfigConstraint

配置约束

**方法**:

- `__init__`
- `validate`

## 函数

### create_default_schema_registry

创建默认的模式注册表

**返回值**: `ConfigSchemaRegistry`

