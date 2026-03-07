# secure_config

**文件路径**: `security\secure_config.py`

## 模块描述

安全的邮件配置管理模块
支持环境变量和加密存储，确保敏感信息安全

## 导入语句

```python
from infrastructure.config.core.imports import os
from infrastructure.config.core.imports import json
import base64
from infrastructure.config.core.imports import Path
from typing import Dict
from cryptography.fernet import Fernet
from infrastructure.config.core.imports import logging
```

## 类

### SecureEmailConfig

secure_config - 配置管理

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
安全的邮件配置管理器

**方法**:

- `__init__`
- `_get_encryption_key`
- `_get_cipher`
- `_encrypt_value`
- `_decrypt_value`
- ... 等4个方法

### SecureConfig

通用安全配置管理器

**方法**:

- `__init__`
- `initialize`
- `encrypt_value`
- `decrypt_value`
- `get_secure_value`
- ... 等3个方法

## 函数

### get_email_config

获取邮件配置的便捷函数

**返回值**: `Dict`

