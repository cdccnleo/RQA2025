# distributedconfigstorage

**文件路径**: `storage\types\distributedconfigstorage.py`

## 模块描述

配置文件存储相关类

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
from infrastructure.config.storage.types.storagetype import StorageType
from infrastructure.config.storage.types.distributedstoragetype import DistributedStorageType
# ... 等4个导入
```

## 类

### DistributedConfigStorage

分布式配置存储实现 - P0级别完整实现

**继承**: IConfigStorage

**方法**:

- `__init__`
- `_initialize_client`
- `_init_redis_client`
- `_init_etcd_client`
- `_init_consul_client`
- ... 等10个方法

