# hotreloadmanager

**文件路径**: `security\components\hotreloadmanager.py`

## 模块描述

安全配置相关类

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Union
from infrastructure.config.core.imports import dataclass
from infrastructure.config.core.imports import field
from infrastructure.config.core.imports import Enum
from infrastructure.config.core.imports import threading
from infrastructure.config.core.common_mixins import ConfigComponentMixin
```

## 类

### HotReloadManager

热重载管理器

**方法**:

- `__init__`
- `watch_file`
- `start_monitoring`
- `stop_monitoring`
- `_monitor_loop`
- ... 等2个方法

