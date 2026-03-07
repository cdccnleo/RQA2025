# cloud_service_mesh

**文件路径**: `environment\cloud_service_mesh.py`

## 模块描述

云原生服务网格管理器

实现服务网格的安装、配置和管理功能

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import os
from infrastructure.config.core.imports import threading
from infrastructure.config.environment.cloud_native_configs import ServiceMeshConfig
from infrastructure.config.environment.cloud_native_configs import ServiceMeshType
```

## 类

### ServiceMeshManager

服务网格管理器

**方法**:

- `__init__`
- `_setup_client`
- `install_service_mesh`
- `_install_istio`
- `_install_linkerd`
- ... 等8个方法

