# 从 FileWatcher 迁移到 ConfigWatcher 指南

## 背景

`FileWatcher` 类已被废弃，推荐使用更强大、更健壮的 `ConfigWatcher` 类。本指南将帮助您将代码从 `FileWatcher` 迁移到 `ConfigWatcher`。

## 为什么要迁移？

`ConfigWatcher` 相比 `FileWatcher` 有以下优势：

1. **更健壮的实现**：
   - 防抖机制，避免频繁触发回调
   - 文件哈希检查，只在内容真正变化时触发回调
   - 完整的错误处理和日志记录

2. **更高的性能**：
   - 线程池处理回调
   - 批处理事件
   - 平台特定优化（Linux上使用inotify，Windows上优化轮询）

3. **更多功能**：
   - 支持多环境配置
   - 支持配置重载
   - 支持嵌套配置访问

4. **更好的可维护性**：
   - 完整的日志记录
   - 更清晰的代码结构
   - 更好的错误处理

## API 差异

| FileWatcher | ConfigWatcher | 说明 |
|-------------|---------------|------|
| `register(file_path, callback)` | `watch(key, callback, env="default", immediate=True)` | ConfigWatcher 支持环境和立即触发选项 |
| `start()` | `start()` 或 `start_watching()` | ConfigWatcher 提供兼容方法 |
| `stop()` | `stop()` 或 `stop_watching()` | ConfigWatcher 提供兼容方法 |
| 无 | `ensure_watcher(env)` | 确保指定环境的监控已设置 |
| 无 | `set_config_change_callback(callback)` | 设置全局配置变更回调 |
| 无 | `set_manager(manager)` | 设置 ConfigManager 引用 |
| 无 | `is_alive()` | 检查监控线程是否运行中 |

## 迁移步骤

### 1. 导入更改

```python
# 旧代码
from src.infrastructure.config.watcher.file_watcher import FileWatcher

# 新代码
from src.infrastructure.config.config_watcher import ConfigWatcher
```

### 2. 初始化更改

```python
# 旧代码
watcher = FileWatcher()

# 新代码
watcher = ConfigWatcher(config_dir="/path/to/config")
```

### 3. 注册回调更改

```python
# 旧代码
watcher.register("/path/to/config.json", my_callback)

# 新代码
watcher.watch("config_key", my_callback, env="default")
```

### 4. 回调函数签名更改

```python
# 旧代码
def my_callback(file_path):
    # 处理文件路径
    pass

# 新代码
def my_callback(value):
    # 处理配置值
    pass
```

## 完整迁移示例

### 旧代码

```python
from src.infrastructure.config.watcher.file_watcher import FileWatcher

def handle_config_change(file_path):
    print(f"配置文件变更: {file_path}")
    # 手动重新加载配置
    with open(file_path, 'r') as f:
        config = json.load(f)
    # 处理配置...

watcher = FileWatcher()
watcher.register("/path/to/config.json", handle_config_change)
watcher.start()

# 应用逻辑...

watcher.stop()
```

### 新代码

```python
from src.infrastructure.config.config_watcher import ConfigWatcher

def handle_config_change(value):
    print(f"配置值变更: {value}")
    # 直接使用新的配置值
    # 处理配置...

watcher = ConfigWatcher(config_dir="/path/to")
watcher.watch("database.connection", handle_config_change, env="default")
watcher.start()

# 应用逻辑...

watcher.stop()
```

## 使用上下文管理器

ConfigWatcher 支持上下文管理器语法，可以自动管理资源：

```python
with ConfigWatcher(config_dir="/path/to") as watcher:
    watcher.watch("database.connection", handle_config_change)
    # 应用逻辑...
    # 退出上下文时自动调用 stop()
```

## 高级用法

### 1. 多环境配置

```python
watcher = ConfigWatcher(config_dir="/path/to/configs")
watcher.watch("database.url", handle_db_change, env="prod")
watcher.watch("database.url", handle_db_change, env="dev")
```

### 2. 嵌套配置访问

```python
def handle_port_change(port):
    print(f"数据库端口变更为: {port}")

watcher.watch("database.connection.port", handle_port_change)
```

### 3. 全局配置变更回调

```python
def global_config_change(file_path):
    print(f"配置文件变更: {file_path}")

watcher.set_config_change_callback(global_config_change)
```

## 常见问题

### 1. 回调函数接收的参数不同

FileWatcher 的回调函数接收文件路径作为参数，而 ConfigWatcher 的回调函数接收配置值作为参数。如果您需要知道哪个文件发生了变化，可以使用 `set_config_change_callback` 方法设置全局回调。

### 2. 配置目录结构

ConfigWatcher 支持两种配置目录结构：

1. 扁平结构：`config/default.json`
2. 嵌套结构：`config/default/default.json`

确保您的配置目录结构与 ConfigWatcher 的期望一致。

### 3. 性能考虑

ConfigWatcher 使用线程池处理回调，如果您有大量回调或回调处理耗时较长，可以调整线程池大小：

```python
watcher = ConfigWatcher(config_dir="/path/to", max_workers=8)
```

## 结论

迁移到 ConfigWatcher 将为您的应用程序提供更健壮、更高性能的配置监控功能。虽然需要进行一些代码更改，但长期收益远大于迁移成本。
