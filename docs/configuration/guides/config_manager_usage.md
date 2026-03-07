# ConfigManager 使用指南

## 功能概述
ConfigManager 提供强大的配置管理能力，包括：
- 多环境配置管理
- 配置验证(JSON Schema)
- 版本历史记录
- 配置差异比较
- 配置回滚
- 配置快照

## 基础用法

```python
from src.infrastructure.config.config_manager import ConfigManager

# 初始化配置管理器
manager = ConfigManager()

# 设置配置
manager.set("database.host", "localhost", env="production")

# 获取配置
db_host = manager.get("database.host", env="production")
```

## 高级功能

### 配置验证
```python
SCHEMA = {
    "type": "object",
    "properties": {
        "database": {"type": "object"},
        "api_key": {"type": "string"}
    },
    "required": ["database"]
}

manager = ConfigManager(schema=SCHEMA)

# 验证配置
if not manager.validate_config("production"):
    print("配置验证失败")
```

### 版本控制
```python
# 更新配置并记录版本
manager.set("api.timeout", 30, env="production", 
           user="admin", comment="增加API超时时间")

# 获取版本历史
history = manager.get_history("production")

# 回滚配置
manager.rollback("production", version="1234567890")
```

### 配置比较与导入导出
```python
# 比较两个环境的差异
diff = manager.diff_configs("staging", "production")

# 创建配置快照
snapshot = manager.create_snapshot("production")

# 导出配置到文件
manager.export_config("production", "prod_config_backup.json")

# 从文件导入配置
manager.import_config("production", "prod_config_backup.json", 
                     user="admin", comment="从备份恢复配置")
```

## 最佳实践
1. 为每个环境定义明确的schema
2. 所有配置变更都应添加注释
3. 定期备份重要配置
4. 生产环境配置变更前先在测试环境验证

## 接口化版本 (v2.1+)

### TypeScript 接口使用
```typescript
import { IConfigManager } from '@infra/config';

// 通过DI获取实例
const configManager = container.get<IConfigManager>('IConfigManager');

// 获取配置
const dbConfig = await configManager.getConfig('database');

// 监听配置变化
configManager.watchConfig('feature_flags', (newValue) => {
  console.log('Feature flags updated:', newValue);
});
```

### Python 适配层
```python
from src.infrastructure.config.interface import IConfigManager

class ConfigManagerAdapter(IConfigManager):
    def __init__(self, legacy_manager):
        self.manager = legacy_manager
        
    async def get_config(self, key: str):
        return self.manager.get(key)
        
    def watch_config(self, key: str, callback):
        self.manager.on_change(key, callback)
```

### 迁移指南
1. 接口变化：
   - 同步方法改为异步
   - 严格类型约束
   - 明确生命周期管理

2. 性能优化：
   - 批量操作接口
   - 内存缓存策略
   - 变更事件节流
```
