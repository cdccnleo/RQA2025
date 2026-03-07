# common_mixins

**文件路径**: `core\common_mixins.py`

## 导入语句

```python
from infrastructure.config.core.imports import threading
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import time
```

## 类

### ConfigComponentMixin

配置组件基础Mixin类

提供通用的初始化和基础功能，避免重复代码。

**方法**:

- `_init_threading_support`
- `_init_config_storage`
- `_init_metrics_collection`
- `_init_alert_system`
- `_init_history_tracking`
- ... 等2个方法

### MonitoringMixin

监控组件Mixin类

**继承**: ConfigComponentMixin

**方法**:

- `__init__`
- `record_metric`
- `get_latest_metric`

### CRUDOperationsMixin

CRUD操作Mixin类

**继承**: ConfigComponentMixin

**方法**:

- `__init__`
- `create`
- `read`
- `update`
- `delete`
- ... 等1个方法

### ComponentLifecycleMixin

组件生命周期Mixin类

**继承**: ConfigComponentMixin

**方法**:

- `__init__`
- `initialize`
- `start`
- `stop`
- `restart`
- ... 等6个方法

