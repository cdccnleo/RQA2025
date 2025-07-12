# RQA2025 基础设施层升级指南

## 1. 概述
本文档记录基础设施层的重大升级内容，包括配置管理、错误处理等核心模块的变更说明和使用指南。

## 2. 新特性

### 2.1 配置管理系统
- **统一配置管理**：通过`ConfigManager`集中管理所有配置项
- **热更新支持**：配置文件修改后自动重新加载
- **环境隔离**：支持不同环境(dev/test/prod)的配置隔离
- **类型验证**：通过`ConfigValidator`实现配置项强类型验证

### 2.2 错误处理系统
- **统一异常处理**：通过`ErrorHandler`集中处理各类异常
- **自动重试**：通过`RetryHandler`实现带退避策略的自动重试
- **错误追踪**：记录完整的错误上下文和堆栈信息
- **告警集成**：支持自定义告警钩子

## 3. 使用指南

### 3.1 配置管理

#### 基本用法
```python
from src.infrastructure.config.config_manager import ConfigManager

# 初始化配置管理器
with ConfigManager(config_dir="./config") as manager:
    # 获取配置项
    db_host = manager.get("database.host")
```

#### 配置验证
```python
from pydantic import BaseModel
from src.infrastructure.config.config_validator import ConfigValidator

# 定义配置模式
class AppConfig(BaseModel):
    thread_count: int = Field(gt=0)
    debug: bool = False

# 注册并验证配置
validator = ConfigValidator()
validator.register_schema("app", AppConfig)
validated = validator.validate("app", raw_config)
```

### 3.2 错误处理

#### 异常捕获和处理
```python
from src.infrastructure.error.error_handler import ErrorHandler

handler = ErrorHandler()

# 注册处理器
def handle_db_error(e):
    logger.error("Database error occurred")
    return None

handler.register_handler(DatabaseError, handle_db_error)

# 使用处理器
try:
    db_operation()
except Exception as e:
    handler.handle(e)
```

#### 自动重试
```python
from src.infrastructure.error.retry_handler import RetryHandler

retry = RetryHandler(max_attempts=3)

@retry
def unstable_api_call():
    # 可能失败的操作
    response = requests.get(url)
    response.raise_for_status()
```

## 4. 最佳实践

### 4.1 配置管理
- 将配置按功能模块分组管理
- 敏感配置应加密存储
- 生产环境配置应与代码分离
- 使用`ConfigValidator`确保配置正确性

### 4.2 错误处理
- 区分可重试和不可重试错误
- 为关键操作添加适当的重试策略
- 记录足够的错误上下文信息
- 实现分级告警机制

## 5. 迁移说明

### 5.1 从旧版本迁移
1. 将分散的配置项迁移到统一的配置文件中
2. 替换原有的错误处理代码为新的`ErrorHandler`
3. 为可能失败的操作添加`RetryHandler`装饰器

### 5.2 兼容性说明
- 新模块与旧代码可以共存
- 建议逐步迁移而非一次性替换
- 提供了兼容性适配层（见`legacy_adapter.py`）

## 6. API参考

### ConfigManager
| 方法 | 描述 |
|------|------|
| `get(key, default=None)` | 获取配置项 |
| `start_watcher()` | 启动配置热更新监听 |
| `stop_watcher()` | 停止配置监听 |

### ErrorHandler
| 方法 | 描述 |
|------|------|
| `register_handler(exc_type, handler)` | 注册异常处理器 |
| `handle(exception)` | 处理异常 |
| `get_records()` | 获取错误记录 |

[更多API详情参见完整文档](api_reference.md)
