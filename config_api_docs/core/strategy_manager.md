# strategy_manager

**文件路径**: `core\strategy_manager.py`

## 模块描述

配置策略管理器

管理各种配置策略的注册、执行和协调

## 导入语句

```python
from infrastructure.config.core.strategy_base import *
from infrastructure.config.core.strategy_loaders import *
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import Union
from infrastructure.config.core.imports import logging
from infrastructure.config.core.imports import threading
```

## 类

### StrategyManager

策略管理器

**方法**:

- `__init__`
- `register_strategy`
- `unregister_strategy`
- `get_strategy`
- `get_strategies_by_type`
- ... 等8个方法

## 函数

### get_strategy_manager

获取策略管理器全局实例

**返回值**: `StrategyManager`

