# cloud_multi_cloud

**文件路径**: `environment\cloud_multi_cloud.py`

## 模块描述

云原生多云管理器

实现多云环境的配置、切换和故障转移功能

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import os
from infrastructure.config.core.imports import subprocess
from infrastructure.config.core.imports import threading
from infrastructure.config.environment.cloud_native_configs import MultiCloudConfig
# ... 等1个导入
```

## 类

### MultiCloudManager

多云管理器

**方法**:

- `__init__`
- `_setup_providers`
- `_setup_aws_provider`
- `_setup_azure_provider`
- `_setup_gcp_provider`
- ... 等14个方法

