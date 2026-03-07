# cloud_native_test_platform

**文件路径**: `tests\cloud_native_test_platform.py`

## 模块描述

云原生测试平台

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Union
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import threading
from infrastructure.config.tests.models.platformtype import PlatformType
from infrastructure.config.tests.models.testservicestatus import TestServiceStatus
from infrastructure.config.tests.models.testenvironment import TestEnvironment
# ... 等5个导入
```

## 类

### ContainerManager

容器管理器

**方法**:

- `__init__`
- `create_container`
- `start_container`
- `stop_container`
- `remove_container`
- ... 等2个方法

### KubernetesManager

Kubernetes管理器

**方法**:

- `__init__`
- `check_kubectl`
- `create_namespace`
- `deploy_service`
- `_create_deployment_yaml`
- ... 等4个方法

### MicroserviceTestRunner

微服务测试运行器

**方法**:

- `__init__`
- `run_health_check`
- `run_load_test`
- `run_integration_test`
- `get_test_results`

### CloudNativeTestPlatform

云原生测试平台主类

**方法**:

- `__init__`
- `deploy_service`
- `run_tests`
- `get_service_info`
- `list_services`
- ... 等2个方法

