# 配置管理代码组织优化迁移指南

## 概述
本次优化整合了多个重复的配置管理器实现，统一为 `UnifiedConfigManager`。

## 变更内容

### 1. 删除的重复实现
- `src/infrastructure/config/core/manager.py` - 旧的ConfigManager
- `src/infrastructure/config/managers/unified.py` - 旧的UnifiedConfigManager  
- `src/infrastructure/config/services/config_service.py` - DistributedConfigManager
- `src/features/config.py` - FeatureConfigManager

### 2. 新的统一实现
- `src/infrastructure/config/unified_manager.py` - 新的UnifiedConfigManager
- `src/infrastructure/config/interfaces/unified_interface.py` - 统一接口定义

## 迁移步骤

### 步骤1: 更新导入语句
```python
# 旧代码
from src.infrastructure.config.core.manager import ConfigManager
from src.infrastructure.config.managers.unified import UnifiedConfigManager
from src.infrastructure.config.services.config_service import DistributedConfigManager
from src.features.config import FeatureConfigManager

# 新代码
from src.infrastructure.config.unified_manager import (
    UnifiedConfigManager, 
    get_unified_config_manager,
    get_config, 
    set_config
)
```

### 步骤2: 更新配置管理器实例化
```python
# 旧代码
config_manager = ConfigManager()
unified_manager = UnifiedConfigManager()

# 新代码
config_manager = get_unified_config_manager()
# 或者直接使用函数
value = get_config('database.url')
set_config('logging.level', 'INFO')
```

### 步骤3: 更新配置作用域使用
```python
# 旧代码
config = unified_manager.get_scope_config(ConfigScope.FEATURES)

# 新代码
config = get_config('feature.enabled', ConfigScope.FEATURES)
```

## 兼容性说明

### 保持兼容的API
- `get(key, default=None)` - 获取配置值
- `set(key, value)` - 设置配置值
- `load(source)` - 加载配置
- `save(destination)` - 保存配置
- `validate()` - 验证配置

### 新增的API
- `get_config(key, scope, default)` - 全局函数获取配置
- `set_config(key, value, scope)` - 全局函数设置配置
- `get_scope_config(scope)` - 获取作用域配置
- `set_scope_config(scope, config)` - 设置作用域配置

## 测试验证

### 1. 功能测试
```python
from src.infrastructure.config.unified_manager import get_config, set_config, ConfigScope

# 测试基本功能
set_config('test.key', 'test_value')
assert get_config('test.key') == 'test_value'

# 测试作用域功能
set_config('feature.enabled', True, ConfigScope.FEATURES)
assert get_config('feature.enabled', ConfigScope.FEATURES) == True
```

### 2. 兼容性测试
```python
from src.infrastructure.config.unified_manager import get_unified_config_manager

config_manager = get_unified_config_manager()
config_manager.set('database.url', 'sqlite:///test.db')
assert config_manager.get('database.url') == 'sqlite:///test.db'
```

## 回滚方案

如果遇到问题，可以回滚到备份的配置管理代码：

1. 恢复备份文件
2. 更新导入语句
3. 重新运行测试

备份位置: `backup/config_optimization/`
