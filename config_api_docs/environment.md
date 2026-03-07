# environment

**文件路径**: `environment.py`

## 模块描述

基础设施层 - 配置管理组件

environment 模块

配置管理相关的文件
提供配置管理相关的功能实现。

## 导入语句

```python
from infrastructure.config.core.imports import os
```

## 类

### ConfigEnvironment

配置环境管理器

**方法**:

- `__init__`
- `get_environment`
- `is_production`
- `is_development`
- `is_testing`
- ... 等4个方法

## 函数

### is_production

检查是否在生产环境

### is_development

检查是否在开发环境

### is_testing

检查是否在测试环境

