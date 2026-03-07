# app

**文件路径**: `web\app.py`

## 导入语句

```python
from infrastructure.config.core.imports import os
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from fastapi import FastAPI
from fastapi import HTTPException
# ... 等12个导入
```

## 类

### WebManagementService

Web管理服务存根实现

**方法**:

- `authenticate_user`
- `create_session`
- `validate_session`
- `get_dashboard_data`
- `update_config_value`
- ... 等10个方法

### LoginRequest

**继承**: BaseModel

### ConfigUpdateRequest

**继承**: BaseModel

### SyncRequest

**继承**: BaseModel

### ConflictResolveRequest

**继承**: BaseModel

## 函数

### get_current_user

app - 配置管理

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
获取当前用户

    Args:
        credentials: HTTP认证凭据

    Returns:
        Dict[str, Any]: 用户信息
    

**参数**:

- `credentials: HTTPAuthorizationCredentials`

**返回值**: `<ast.Subscript object at 0x000002259024EB80>`

