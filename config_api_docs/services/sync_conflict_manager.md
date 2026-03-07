# sync_conflict_manager

**文件路径**: `services\sync_conflict_manager.py`

## 模块描述

同步冲突管理器
处理分布式配置同步中的冲突检测和解决

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
import hashlib
from infrastructure.config.core.imports import json
from infrastructure.config.core.imports import defaultdict
```

## 类

### SyncConflictManager

同步冲突管理器

**方法**:

- `__init__`
- `calculate_config_checksum`
- `detect_conflicts`
- `resolve_conflicts`
- `get_conflicts`
- ... 等3个方法

