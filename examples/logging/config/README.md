# Logger配置系统使用指南

## 📖 概述

RQA2025的Logger配置系统提供了灵活的配置选项，支持多种配置方式，包括文件配置、环境变量配置和热重载配置。系统设计的目标是让Logger配置既强大又易用，同时支持生产环境的动态配置需求。

## 🎯 配置方式

### 1. 文件配置 (推荐)

支持YAML和JSON格式的配置文件：

#### YAML配置示例
```yaml
version: "1.0"
manager:
  default_level: INFO
  log_directory: "logs"
pool:
  enabled: true
  max_size: 100
handlers:
  file:
    handler_type: "file"
    level: INFO
    output_path: "logs/app.log"
loggers:
  business:
    level: INFO
    category: BUSINESS
    handlers: ["business_file"]
```

#### JSON配置示例
```json
{
  "version": "1.0",
  "manager": {
    "default_level": "INFO",
    "log_directory": "logs"
  },
  "pool": {
    "enabled": true,
    "max_size": 100
  }
}
```

### 2. 环境变量配置

通过环境变量动态覆盖配置：

```bash
# 基本配置
export LOGGER_LEVEL=DEBUG
export LOGGER_LOG_DIR=/var/log/app
export LOGGER_POOL_MAX_SIZE=200

# 专用Logger配置
export LOGGER_BUSINESS_LEVEL=INFO
export LOGGER_AUDIT_LEVEL=WARNING
```

### 3. 编程式配置

通过代码直接配置：

```python
from infrastructure.logging.config import LoggerConfigManager, LoggerConfig

# 创建配置
config = LoggerConfig()
config.manager.default_level = "DEBUG"

# 创建管理器
manager = LoggerConfigManager()
manager.update_config(config)
```

## 🚀 快速开始

### 1. 创建配置文件

```python
from infrastructure.logging.config import create_default_config_file

# 创建YAML配置文件
create_default_config_file("logger_config.yaml", "yaml")

# 创建JSON配置文件
create_default_config_file("logger_config.json", "json")
```

### 2. 使用配置管理器

```python
from infrastructure.logging.config import create_config_manager

# 创建配置管理器（自动重载）
manager = create_config_manager("logger_config.yaml")

# 获取配置
config = manager.get_config()

# 修改配置
manager.set_config_value("manager.default_level", "DEBUG")
```

### 3. 环境变量覆盖

```python
from infrastructure.logging.config import EnvironmentLoggerConfig

# 加载环境变量配置
env_config = EnvironmentLoggerConfig()
overrides = env_config.load_from_env()

# 应用到基础配置
final_config = env_config.apply_env_overrides(base_config)
```

### 4. 热重载配置

```python
from infrastructure.logging.config import HotReloadLoggerConfig

# 创建热重载处理器
hot_reload = HotReloadLoggerConfig(manager, check_interval=5.0)

# 添加回调
def on_reload(new_config):
    print(f"配置已重载: {new_config.version}")

hot_reload.add_reload_callback(on_reload)

# 启动监控
with hot_reload:
    # 配置文件变化时自动重载
    pass
```

## 📋 配置选项详解

### 管理器配置 (manager)

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| default_level | LogLevel | INFO | 默认日志级别 |
| default_format | LogFormat | STRUCTURED | 默认日志格式 |
| default_category | LogCategory | GENERAL | 默认日志分类 |
| log_directory | str/Path | "logs" | 日志目录 |
| enable_async | bool | true | 启用异步日志 |
| async_queue_size | int | 1000 | 异步队列大小 |
| flush_interval | float | 1.0 | 刷新间隔(秒) |
| error_handling | str | "ignore" | 错误处理策略 |

### 对象池配置 (pool)

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| enabled | bool | true | 启用对象池 |
| max_size | int | 100 | 池最大大小 |
| initial_size | int | 10 | 池初始大小 |
| idle_timeout | int | 300 | 空闲超时(秒) |
| cleanup_interval | int | 60 | 清理间隔(秒) |
| enable_lru | bool | true | 启用LRU淘汰 |
| preload_loggers | list | [] | 预加载Logger列表 |

### 处理器配置 (handlers)

#### 文件处理器
```yaml
file_handler:
  handler_type: "file"
  level: INFO
  output_path: "logs/app.log"
  max_bytes: 10485760  # 10MB
  backup_count: 5
  encoding: "utf-8"
  formatter:
    format_type: STRUCTURED
```

#### 控制台处理器
```yaml
console_handler:
  handler_type: "console"
  level: INFO
  formatter:
    format_type: SIMPLE
    format_template: "%(asctime)s - %(levelname)s - %(message)s"
```

### 专用Logger配置 (loggers)

```yaml
business_logger:
  level: INFO
  category: BUSINESS
  handlers: ["business_file", "json_file"]
  log_directory: "logs/business"
  # 其他专用配置...
```

## 🔧 高级功能

### 配置验证

```python
from infrastructure.logging.config import LoggerConfigValidator

validator = LoggerConfigValidator()
config = LoggerConfig()

if validator.validate(config):
    print("配置验证通过")
else:
    print("配置验证失败:")
    for error in validator.get_errors():
        print(f"  - {error}")
```

### 配置合并

```python
from infrastructure.logging.config import LoggerConfigLoader

loader = LoggerConfigLoader()

# 多源配置合并
sources = [
    {"source": "base_config.yaml", "type": "yaml"},
    {"source": "env_overrides", "type": "env"},
    {"source": "runtime_config.json", "type": "json"}
]

merged_config = loader.load_multiple_sources(sources, merge_strategy="override")
```

### 配置备份和恢复

```python
# 配置备份
hot_reload.export_config("config_backup.json", include_backups=True)

# 配置恢复
hot_reload.import_config("config_backup.json")
```

## 🌐 环境变量映射

| 环境变量 | 配置路径 | 描述 |
|----------|----------|------|
| LOGGER_LEVEL | manager.default_level | 默认日志级别 |
| LOGGER_LOG_DIR | manager.log_directory | 日志目录 |
| LOGGER_POOL_MAX_SIZE | pool.max_size | 对象池大小 |
| LOGGER_BUSINESS_LEVEL | loggers.business.level | 业务Logger级别 |
| LOGGER_ASYNC_ENABLED | manager.enable_async | 异步日志开关 |

完整的环境变量列表请参考 `env_config_example.txt`。

## 📊 监控和统计

### 重载统计

```python
stats = hot_reload.get_reload_stats()
print(f"重载次数: {stats['reload_count']}")
print(f"失败次数: {stats['failed_reload_count']}")
print(f"监控状态: {stats['is_running']}")
```

### 配置变更历史

```python
backups = hot_reload.list_backups()
for backup in backups:
    print(f"版本: {backup['version']}, 时间: {backup['timestamp']}")
```

## 🚨 故障排除

### 常见问题

1. **配置文件未找到**
   ```python
   # 检查文件是否存在
   import os
   if not os.path.exists("logger_config.yaml"):
       create_default_config_file("logger_config.yaml")
   ```

2. **配置验证失败**
   ```python
   # 查看详细错误信息
   errors = validator.get_errors()
   warnings = validator.get_warnings()
   ```

3. **热重载不工作**
   ```python
   # 检查文件权限和监控状态
   stats = hot_reload.get_reload_stats()
   print(f"监控运行: {stats['is_running']}")
   ```

## 📚 相关文档

- [Logger API文档](../../docs/api/logger_api.md) - Logger使用指南
- [架构设计文档](../../docs/architecture/infrastructure_architecture_design.md) - 系统架构说明
- [重构完成报告](../../INFRASTRUCTURE_REFACTORING_COMPLETION_REPORT.md) - 重构详情

---

**配置系统设计理念**: 简单、灵活、可扩展，支持从开发环境到生产环境的无缝配置管理。
