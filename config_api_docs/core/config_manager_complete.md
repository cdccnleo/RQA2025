# config_manager_complete

**文件路径**: `core\config_manager_complete.py`

## 模块描述

统一配置管理器完整版 (组合所有拆分的功能)

将核心、存储和操作功能组合成完整的配置管理器

## 导入语句

```python
from infrastructure.config.core.config_manager_operations import UnifiedConfigManagerWithOperations
```

## 类

### UnifiedConfigManager

统一配置管理器完整版

组合了所有拆分的功能模块：
- 核心功能 (config_manager_core.py)
- 存储功能 (config_manager_storage.py)
- 操作功能 (config_manager_operations.py)

**继承**: UnifiedConfigManagerWithOperations

**方法**:

- `__init__`
- `_initialize_enhanced_features`

