# edge_computing_test_platform

**文件路径**: `tests\edge_computing_test_platform.py`

## 模块描述

边缘计算测试平台

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Union
from infrastructure.config.core.imports import time
from infrastructure.config.core.imports import threading
from infrastructure.config.tests.edge_models.edgenodetype import EdgeNodeType
from infrastructure.config.tests.edge_models.nodestatus import NodeStatus
from infrastructure.config.tests.edge_models.testtype import TestType
# ... 等3个导入
```

## 类

### EdgeNodeManager

边缘节点管理器

**方法**:

- `__init__`
- `add_node`
- `remove_node`
- `get_node`
- `get_all_nodes`
- ... 等7个方法

### EdgeComputingTestPlatform

边缘计算测试平台主类

**方法**:

- `__init__`
- `start_platform`
- `stop_platform`
- `add_edge_node`
- `remove_edge_node`
- ... 等5个方法

