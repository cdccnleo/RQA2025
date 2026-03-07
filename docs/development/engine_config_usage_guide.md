# 引擎层配置管理系统使用指南

## 概述

引擎层配置管理系统提供了统一的配置管理功能，包括配置管理器、验证器、加载器、模式定义和热重载等功能。本指南介绍如何使用这些组件来管理引擎层的配置。

## 核心组件

### 1. 配置管理器 (EngineConfigManager)

配置管理器是系统的核心组件，提供统一的配置接口。

```python
from src.engine.config import EngineConfigManager, ConfigScope

# 创建配置管理器
config_manager = EngineConfigManager(
    config_dir="config/engine",
    env="default",
    enable_hot_reload=True,
    enable_validation=True
)

# 获取配置
max_workers = config_manager.get("max_workers", ConfigScope.REALTIME, default=4)
queue_size = config_manager.get("queue_size", ConfigScope.REALTIME, default=10000)

# 设置配置
config_manager.set("max_workers", 8, ConfigScope.REALTIME)
config_manager.set("queue_size", 20000, ConfigScope.REALTIME)

# 获取作用域配置
realtime_config = config_manager.get_scope_config(ConfigScope.REALTIME)
buffer_config = config_manager.get_scope_config(ConfigScope.BUFFER)
```

### 2. 配置验证器 (EngineConfigValidator)

配置验证器提供配置验证和规则检查功能。

```python
from src.engine.config import EngineConfigValidator

# 创建验证器
validator = EngineConfigValidator()

# 验证作用域配置
config = {"max_workers": 4, "queue_size": 10000}
is_valid, errors = validator.validate_scope_config(config, "realtime")

if not is_valid:
    print("配置验证失败:", errors)
else:
    print("配置验证通过")

# 添加自定义验证规则
def custom_rule(value):
    return isinstance(value, str) and len(value) > 0

validator.add_rule("custom_rule", custom_rule, "值必须是非空字符串")
```

### 3. 配置加载器 (EngineConfigLoader)

配置加载器支持多种格式的配置文件加载和保存。

```python
from src.engine.config import EngineConfigLoader
from pathlib import Path

# 创建加载器
loader = EngineConfigLoader(Path("config/engine"))

# 加载配置文件
config_data = loader.load_config("config/engine/default.json")

# 保存配置
config_data = {"test": {"key": "value"}}
success = loader.save_config(config_data, "config/engine/test.json")

# 加载所有配置文件
all_configs = loader.load_all_configs("*.json")

# 合并配置
configs = [
    {"section1": {"key1": "value1"}},
    {"section1": {"key2": "value2"}}
]
merged = loader.merge_configs(configs)
```

### 4. 配置模式 (EngineConfigSchema)

配置模式定义配置结构和验证规则。

```python
from src.engine.config import EngineConfigSchema

# 创建模式
schema = EngineConfigSchema()

# 验证配置
config = {"max_workers": 4, "queue_size": 10000}
is_valid, errors = schema.validate(config, "realtime")

# 获取模式定义
realtime_schema = schema.get_schema("realtime")

# 获取模式摘要
summary = schema.get_schema_summary()
```

### 5. 热重载 (EngineConfigHotReload)

热重载支持配置文件的实时监控和自动重载。

```python
from src.engine.config import EngineConfigHotReload

# 创建热重载管理器
hot_reload = EngineConfigHotReload(config_manager)

# 启动热重载
hot_reload.start()

# 监控特定文件
hot_reload.watch_file("config/engine/default.json")

# 添加重载回调
def reload_callback(reload_info):
    print(f"配置重载: {reload_info.file_path}")

hot_reload.add_reload_callback(reload_callback)

# 停止热重载
hot_reload.stop()
```

## 配置作用域

系统定义了以下配置作用域：

- `REALTIME`: 实时引擎配置
- `BUFFER`: 缓冲区配置
- `DISPATCHER`: 事件分发器配置
- `LEVEL2`: Level2数据处理配置
- `OPTIMIZATION`: 优化配置
- `PRODUCTION`: 生产环境配置
- `MONITORING`: 监控配置
- `GLOBAL`: 全局配置

## 配置验证规则

### 实时引擎配置
- `max_workers`: 1-32之间的整数
- `queue_size`: 正整数
- `timeout`: 正数

### 缓冲区配置
- `pool_size`: 正整数
- `chunk_size`: 2的幂次方
- `memory_limit`: 正整数

### 分发器配置
- `max_queues`: 1-100之间的整数
- `priority_levels`: 1-10之间的整数

### Level2配置
- `max_symbols`: 正整数
- `order_book_depth`: 1-50之间的整数

### 优化配置
- `optimization_interval`: 大于等于60秒
- `performance_threshold`: 0-1之间的数值

### 生产配置
- `api_timeout`: 正整数
- `max_concurrent_requests`: 正整数

### 监控配置
- `data_retention_days`: 1-365之间的整数

## 使用示例

### 基本使用

```python
from src.engine.config import (
    EngineConfigManager, ConfigScope, 
    get_engine_config_manager, get_engine_config
)

# 获取全局配置管理器
config_manager = get_engine_config_manager()

# 获取配置
max_workers = get_engine_config("max_workers", ConfigScope.REALTIME, 4)
queue_size = get_engine_config("queue_size", ConfigScope.REALTIME, 10000)

# 设置配置
config_manager.set("max_workers", 8, ConfigScope.REALTIME)

# 验证配置
is_valid, errors = config_manager.validate_config(ConfigScope.REALTIME)
if not is_valid:
    print("配置验证失败:", errors)
```

### 组件注册

```python
# 注册组件
config_manager.register_component("realtime_engine", engine_instance, ["max_workers", "queue_size"])

# 获取组件配置
component_config = config_manager.get_component_config("realtime_engine")

# 更新组件配置
config_manager.update_component_config("realtime_engine", {"max_workers": 8})
```

### 配置观察者

```python
def config_changed(key, old_value, new_value, scope):
    print(f"配置变更: {key} = {old_value} -> {new_value}")

# 添加观察者
watcher_id = config_manager.add_watcher("max_workers", config_changed)

# 移除观察者
config_manager.remove_watcher("max_workers", watcher_id)
```

### 配置文件管理

```python
# 加载配置文件
success = config_manager.load_config("config/engine/default.json")

# 保存配置
success = config_manager.save_config("config/engine/backup.json")

# 获取配置摘要
summary = config_manager.get_config_summary()
print(f"总作用域数: {summary['total_scopes']}")
print(f"总配置数: {summary['total_configs']}")
```

### 性能监控

```python
# 获取性能指标
metrics = config_manager.get_performance_metrics()
print(f"配置加载次数: {metrics['config_loads']}")
print(f"配置保存次数: {metrics['config_saves']}")
print(f"验证检查次数: {metrics['validation_checks']}")

# 健康检查
health = config_manager.health_check()
print(f"状态: {health['status']}")
print(f"配置有效: {health['config_valid']}")
```

## 最佳实践

### 1. 配置管理
- 使用作用域隔离不同模块的配置
- 为每个环境创建独立的配置文件
- 定期备份重要配置

### 2. 配置验证
- 在应用启动时验证所有配置
- 添加自定义验证规则确保配置正确性
- 使用模式定义明确配置结构

### 3. 热重载
- 在生产环境中谨慎使用热重载
- 添加重载回调进行日志记录
- 监控重载历史避免频繁变更

### 4. 性能优化
- 使用缓存减少配置加载开销
- 合理设置观察者数量
- 定期清理历史记录

### 5. 错误处理
- 捕获配置加载和验证异常
- 提供默认配置作为备选方案
- 记录详细的错误信息

## 故障排除

### 常见问题

1. **配置验证失败**
   - 检查配置值是否在允许范围内
   - 确认所有必需字段都已提供
   - 验证字段类型是否正确

2. **热重载不工作**
   - 确认文件监控权限
   - 检查文件路径是否正确
   - 验证观察者是否正常启动

3. **配置加载失败**
   - 检查文件格式是否正确
   - 确认文件路径存在
   - 验证文件权限

4. **性能问题**
   - 减少观察者数量
   - 清理历史记录
   - 优化验证规则

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查配置状态
health = config_manager.health_check()
print(health)

# 查看验证摘要
summary = validator.get_validation_summary()
print(summary)

# 检查热重载状态
status = hot_reload.get_reload_status()
print(status)
```

## 总结

引擎层配置管理系统提供了完整的配置管理解决方案，包括：

- **统一配置接口**: 通过EngineConfigManager提供一致的配置访问
- **配置验证**: 通过EngineConfigValidator确保配置正确性
- **多格式支持**: 通过EngineConfigLoader支持JSON、YAML、INI等格式
- **模式定义**: 通过EngineConfigSchema定义配置结构
- **热重载**: 通过EngineConfigHotReload支持实时配置更新
- **性能监控**: 提供详细的性能指标和健康检查

通过合理使用这些组件，可以构建稳定、高效的配置管理系统，为引擎层的运行提供可靠保障。

---

**文档维护**: 开发团队  
**最后更新**: 2025-01-27  
**版本**: 1.0 