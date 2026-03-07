# 配置管理模块API使用指南

## 📖 概述

配置管理模块提供了统一的配置管理功能，支持多种配置格式、验证机制和扩展功能。

## 🚀 快速开始

### 基本使用

```python
from src.infrastructure.config import ConfigFactory, ConfigManager

# 创建配置管理器
config_manager = ConfigFactory.create_config_manager()

# 设置配置
result = config_manager.set("cache.enabled", True)
if result.success:
    print("配置设置成功")
else:
    print(f"配置设置失败: {result.error}")

# 获取配置
cache_enabled = config_manager.get("cache.enabled", default=False)
print(f"缓存是否启用: {cache_enabled}")
```

### 完整服务创建

```python
# 创建完整的配置服务
config_service = ConfigFactory.create_complete_config_service(
    env='production',
    config_provider=ConfigFactory.create_provider('file'),
    validator=ConfigFactory.create_validator('schema'),
    event_bus=ConfigFactory.create_event_bus('redis'),
    version_manager=ConfigFactory.create_version_manager('simple')
)
```

## 🔧 核心API

### ConfigManager

#### 基本操作

```python
# 获取配置
value = config_manager.get("key", default=None)

# 设置配置
result = config_manager.set("key", value)

# 验证配置
is_valid, errors = config_manager.validate(config_dict)

# 加载配置
result = config_manager.load("config.json")

# 保存配置
result = config_manager.save("config.json")
```

#### 高级功能

```python
# 监听配置变化
def on_config_change(key, old_value, new_value):
    print(f"配置 {key} 从 {old_value} 变更为 {new_value}")

subscription_id = config_manager.watch("cache.enabled", on_config_change)

# 取消监听
config_manager.unwatch("cache.enabled", subscription_id)

# 创建版本
version_id = config_manager.create_version()

# 切换版本
config_manager.switch_version(version_id)
```

### ConfigValidator

```python
from src.infrastructure.config import ConfigValidator

validator = ConfigValidator()

# 验证配置
is_valid, errors = validator.validate(config_dict)

# 验证配置更新
is_valid, errors = validator.validate_update("key", new_value, current_config)
```

### ConfigProvider

```python
from src.infrastructure.config import ConfigProvider

provider = ConfigProvider()

# 加载配置
config = provider.load("config.json")

# 保存配置
success = provider.save(config, "config.json")

# 获取默认配置
default_config = provider.get_default()
```

## 📁 配置格式支持

### JSON配置

```json
{
  "cache": {
    "enabled": true,
    "max_size": 1000,
    "ttl": 3600
  },
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "rqa_db"
  }
}
```

### 环境变量配置

```bash
export RQA_CACHE_ENABLED=true
export RQA_CACHE_MAX_SIZE=1000
export RQA_DATABASE_HOST=localhost
```

### 混合配置

```python
# 从文件加载基础配置
config_manager.load("base_config.json")

# 从环境变量覆盖
config_manager.load_from_env("RQA_")
```

## 🔒 安全配置

### 敏感信息加密

```python
from src.infrastructure.config.services.security import SecurityService

security_service = SecurityService()

# 加密敏感配置
encrypted_config = security_service.encrypt_config({
    "database": {
        "password": "secret_password"
    }
})

# 解密配置
decrypted_config = security_service.decrypt_config(encrypted_config)
```

## 📊 监控和日志

### 配置变更监控

```python
# 监听所有配置变更
def on_any_config_change(key, old_value, new_value):
    print(f"配置变更: {key} = {new_value}")

config_manager.watch("*", on_any_config_change)
```

### 审计日志

```python
# 启用审计日志
config_manager.enable_audit_logging()

# 查看审计日志
audit_logs = config_manager.get_audit_logs()
```

## 🛠️ 高级功能

### 配置模板

```python
# 创建配置模板
template = {
    "cache": {
        "enabled": "${CACHE_ENABLED:true}",
        "max_size": "${CACHE_MAX_SIZE:1000}"
    }
}

# 应用模板
config = config_manager.apply_template(template, {
    "CACHE_ENABLED": "false",
    "CACHE_MAX_SIZE": "2000"
})
```

### 配置同步

```python
# 同步到远程
config_manager.sync_to_remote("https://config-server.com/api/config")

# 从远程同步
config_manager.sync_from_remote("https://config-server.com/api/config")
```

## 🔧 故障排除

### 常见问题

1. **配置加载失败**
   ```python
   # 检查文件是否存在
   import os
   if os.path.exists("config.json"):
       config_manager.load("config.json")
   else:
       print("配置文件不存在")
   ```

2. **配置验证失败**
   ```python
   # 获取详细错误信息
   is_valid, errors = config_manager.validate(config)
   if not is_valid:
       for error in errors:
           print(f"验证错误: {error}")
   ```

3. **性能问题**
   ```python
   # 启用缓存
   config_manager.enable_caching()
   
   # 批量操作
   config_manager.batch_set({
       "key1": "value1",
       "key2": "value2"
   })
   ```

## 📚 最佳实践

1. **使用环境变量覆盖敏感配置**
2. **启用配置验证确保数据完整性**
3. **使用版本控制管理配置变更**
4. **定期备份重要配置**
5. **监控配置变更避免意外修改**

## 🔗 相关文档

- [架构设计文档](../architecture/config_final_architecture.md)
- [优化建议文档](../architecture/config_optimization_suggestions.md)
- [测试指南](../testing/config_testing_guide.md) 