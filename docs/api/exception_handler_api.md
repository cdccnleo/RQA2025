# 异常处理框架API文档

## 概述

通用异常处理框架提供标准化的异常处理机制，支持多种处理策略、重试机制和详细的上下文跟踪。该框架减少代码重复，提高错误处理的一致性和可维护性。

## 架构概览

```
common_exception_handler.py
├── ExceptionHandlingStrategy    # 异常处理策略枚举
├── LogLevel                    # 日志级别枚举
├── ExceptionContext           # 异常上下文信息类
├── handle_exceptions()        # 通用异常处理装饰器
├── ExceptionCollector         # 异常收集器类
├── global_exception_collector # 全局异常收集器实例

便捷装饰器:
├── handle_config_exceptions()    # 配置异常处理
├── handle_cache_exceptions()     # 缓存异常处理
├── handle_monitoring_exceptions() # 监控异常处理
├── handle_validation_exceptions() # 验证异常处理
```

## 核心组件API

### ExceptionHandlingStrategy 枚举

```python
from infrastructure.config.core.common_exception_handler import ExceptionHandlingStrategy

# 异常处理策略
ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT    # 记录日志并返回默认值
ExceptionHandlingStrategy.LOG_AND_RERAISE         # 记录日志并重新抛出异常
ExceptionHandlingStrategy.SILENT_RETURN_DEFAULT    # 静默返回默认值
ExceptionHandlingStrategy.COLLECT_AND_RETURN       # 收集异常并返回结果
```

### LogLevel 枚举

```python
from infrastructure.config.core.common_exception_handler import LogLevel

# 日志级别
LogLevel.DEBUG    # 调试级
LogLevel.INFO     # 信息级
LogLevel.WARNING  # 警告级
LogLevel.ERROR    # 错误级
LogLevel.CRITICAL # 严重级
```

### ExceptionContext 类

```python
from infrastructure.config.core.common_exception_handler import ExceptionContext

# 创建异常上下文
context = ExceptionContext(
    operation="database_connect",
    component="DatabaseManager",
    parameters={"host": "localhost", "port": 3306},
    start_time=time.time()
)

# 完成操作
context.complete(success=False, error="Connection timeout")

# 转换为字典
context_dict = context.to_dict()
```

## 装饰器API

### handle_exceptions() 装饰器

```python
from infrastructure.config.core.common_exception_handler import handle_exceptions

@handle_exceptions(
    strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
    default_return=None,
    log_level=LogLevel.ERROR,
    include_context=True,
    max_retries=3,
    retry_delay=0.5
)
def risky_operation(param1, param2):
    """可能抛出异常的操作"""
    # 业务逻辑
    return some_risky_call(param1, param2)

# 使用示例
result = risky_operation("value1", "value2")
# 如果出现异常，会记录错误日志并返回None
```

**参数说明：**
- `strategy`: 异常处理策略
- `default_return`: 异常时的默认返回值
- `log_level`: 日志记录级别
- `include_context`: 是否包含操作上下文信息
- `max_retries`: 最大重试次数 (0表示不重试)
- `retry_delay`: 重试间隔时间(秒)

### 便捷装饰器

#### handle_config_exceptions()

```python
from infrastructure.config.core.common_exception_handler import handle_config_exceptions

@handle_config_exceptions(default_return={}, log_level=LogLevel.WARNING)
def load_config(file_path):
    """加载配置文件"""
    with open(file_path, 'r') as f:
        return json.load(f)
```

#### handle_cache_exceptions()

```python
from infrastructure.config.core.common_exception_handler import handle_cache_exceptions

@handle_cache_exceptions(default_return=None, log_level=LogLevel.ERROR)
def cache_get(self, key):
    """从缓存获取数据"""
    return self.cache.get(key)
```

#### handle_monitoring_exceptions()

```python
from infrastructure.config.core.common_exception_handler import handle_monitoring_exceptions

@handle_monitoring_exceptions(default_return={}, log_level=LogLevel.WARNING)
def collect_metrics():
    """收集监控指标"""
    # 监控逻辑，如果失败返回空字典
    return get_system_metrics()
```

#### handle_validation_exceptions()

```python
from infrastructure.config.core.common_exception_handler import handle_validation_exceptions

@handle_validation_exceptions(default_return=[], log_level=LogLevel.ERROR)
def validate_config(config):
    """验证配置"""
    # 返回验证结果列表
    return perform_validation(config)
```

## ExceptionCollector 类

```python
from infrastructure.config.core.common_exception_handler import ExceptionCollector

# 创建异常收集器
collector = ExceptionCollector(max_exceptions=100)

# 添加异常
try:
    risky_operation()
except Exception as e:
    collector.add_exception(e, context=ExceptionContext(...))

# 检查状态
if collector.has_exceptions():
    print(f"收集到 {len(collector.exceptions)} 个异常")

# 获取异常摘要
summary = collector.get_summary()
print(f"异常类型统计: {summary['exception_types']}")

# 清空收集器
collector.clear()
```

**方法：**
- `add_exception(exception, context)`: 添加异常
- `get_exceptions(severity_filter)`: 获取异常列表
- `has_exceptions()`: 检查是否有异常
- `get_summary()`: 获取异常摘要统计
- `clear()`: 清空异常收集器

## 全局异常收集器

```python
from infrastructure.config.core.common_exception_handler import global_exception_collector

# 使用全局异常收集器
try:
    operation_that_might_fail()
except Exception as e:
    global_exception_collector.add_exception(e)

# 获取全局异常统计
summary = global_exception_collector.get_summary()
print(f"系统总异常数: {summary['total_exceptions']}")
```

## 使用模式

### 1. 基础异常处理

```python
from infrastructure.config.core.common_exception_handler import handle_exceptions

@handle_exceptions(
    strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
    default_return="default_value"
)
def simple_operation():
    # 可能失败的操作
    return risky_call()
```

### 2. 重试机制

```python
@handle_exceptions(
    strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
    default_return=None,
    max_retries=3,
    retry_delay=1.0
)
def network_operation():
    """网络操作，支持重试"""
    return requests.get("https://api.example.com/data")
```

### 3. 详细上下文跟踪

```python
from infrastructure.config.core.common_logger import create_operation_context

@handle_exceptions(
    strategy=ExceptionHandlingStrategy.COLLECT_AND_RETURN,
    include_context=True
)
def complex_operation(user_id, config):
    """复杂操作，记录详细上下文"""
    context = create_operation_context(
        component="ComplexOperation",
        operation="process_data",
        parameters={"user_id": user_id, "config_keys": list(config.keys())}
    )

    # 业务逻辑
    result = process_complex_data(config)

    context.complete(success=True, result=f"processed {len(result)} items")
    return result
```

### 4. 异常收集和分析

```python
from infrastructure.config.core.common_exception_handler import ExceptionCollector

class BatchProcessor:
    def __init__(self):
        self.exception_collector = ExceptionCollector()

    def process_batch(self, items):
        results = []

        for item in items:
            try:
                result = self.process_item(item)
                results.append(result)
            except Exception as e:
                context = ExceptionContext(
                    operation="process_item",
                    component="BatchProcessor",
                    parameters={"item_id": item.get('id')}
                )
                self.exception_collector.add_exception(e, context)
                results.append(None)  # 或其他默认值

        return results

    def get_processing_summary(self):
        """获取处理摘要"""
        summary = self.exception_collector.get_summary()
        return {
            "total_exceptions": summary['total_exceptions'],
            "exception_types": summary['exception_types'],
            "success_rate": 1.0 - (summary['total_exceptions'] / len(self.items))
        }
```

## 高级用法

### 自定义异常处理策略

```python
from infrastructure.config.core.common_exception_handler import handle_exceptions
from functools import wraps

def custom_exception_handler(strategy="email_alert"):
    """自定义异常处理策略"""
    def decorator(func):
        @wraps(func)
        @handle_exceptions(
            strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
            default_return=None,
            include_context=True
        )
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 自定义处理逻辑
                if strategy == "email_alert":
                    send_email_alert(f"异常告警: {func.__name__}", str(e))
                elif strategy == "metrics":
                    increment_error_metric(func.__name__)
                # 重新抛出或返回默认值由装饰器处理
                raise
        return wrapper
    return decorator

@custom_exception_handler("email_alert")
def critical_operation():
    """关键操作，异常时发送邮件告警"""
    return risky_operation()
```

### 异常链追踪

```python
class ExceptionChainTracker:
    """异常链追踪器"""

    def __init__(self):
        self.exception_chain = []

    @handle_exceptions(
        strategy=ExceptionHandlingStrategy.COLLECT_AND_RETURN,
        include_context=True
    )
    def execute_with_tracking(self, operation, *args, **kwargs):
        """执行操作并追踪异常链"""
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            # 记录异常链
            chain_entry = {
                "operation": operation.__name__,
                "exception": str(e),
                "args": args,
                "kwargs": list(kwargs.keys()),  # 不记录敏感信息
                "timestamp": time.time()
            }
            self.exception_chain.append(chain_entry)

            # 继续抛出，让装饰器处理
            raise

    def get_exception_chain_report(self):
        """获取异常链报告"""
        return {
            "total_exceptions": len(self.exception_chain),
            "chain": self.exception_chain,
            "most_common_operation": self._get_most_common_operation()
        }

    def _get_most_common_operation(self):
        """获取最常出现异常的操作"""
        from collections import Counter
        operations = [entry['operation'] for entry in self.exception_chain]
        return Counter(operations).most_common(1)[0][0] if operations else None
```

## 性能监控

异常处理框架包含性能监控功能：

```python
from infrastructure.config.core.common_exception_handler import handle_exceptions
import time

@handle_exceptions(include_context=True)
def monitored_operation():
    """带性能监控的操作"""
    start_time = time.time()
    result = perform_operation()
    duration = time.time() - start_time

    # 上下文会自动记录操作时间
    return result
```

## 最佳实践

### 1. 选择合适的处理策略

```python
# 对于配置加载：记录日志并返回默认值
@handle_config_exceptions(default_return={})
def load_config(): pass

# 对于缓存操作：记录错误并返回None
@handle_cache_exceptions(default_return=None)
def cache_get(): pass

# 对于验证操作：收集异常并返回结果
@handle_validation_exceptions(default_return=[])
def validate(): pass
```

### 2. 使用上下文信息

```python
@handle_exceptions(include_context=True)
def business_operation(user_id, order_id):
    """业务操作，记录详细上下文"""
    # 上下文会自动包含操作名、参数等信息
    return process_order(user_id, order_id)
```

### 3. 合理设置重试策略

```python
# 网络操作：重试3次，每次间隔1秒
@handle_exceptions(max_retries=3, retry_delay=1.0)
def api_call(): pass

# 数据库操作：重试2次，每次间隔0.5秒
@handle_exceptions(max_retries=2, retry_delay=0.5)
def db_query(): pass
```

### 4. 异常收集和监控

```python
class ServiceMonitor:
    def __init__(self):
        self.collector = ExceptionCollector()

    def monitored_method(self, *args, **kwargs):
        @handle_exceptions(
            strategy=ExceptionHandlingStrategy.COLLECT_AND_RETURN,
            include_context=True
        )
        def inner():
            return self.actual_method(*args, **kwargs)

        result = inner()

        if isinstance(result, dict) and 'success' in result and not result['success']:
            # 处理异常情况
            self.handle_service_exception(result)

        return result
```

## 向后兼容性

异常处理框架保持向后兼容：

```python
# 兼容旧的导入方式
from infrastructure.config.core.common_exception_handler import (
    handle_exceptions,  # 推荐的新方式
    ExceptionCollector,
    global_exception_collector
)
```

## 线程安全

异常处理框架的所有组件都是线程安全的：

- `ExceptionCollector` 使用内部锁保护状态
- 装饰器生成的上下文信息是线程独立的
- 全局异常收集器支持并发访问

## 性能影响

异常处理框架对正常执行路径的影响最小：

- 正常执行：几乎无性能开销（仅上下文创建）
- 异常处理：包含日志记录和可能的清理操作
- 重试机制：按需启用，避免不必要的性能开销

---

**版本**: v1.0
**更新日期**: 2025-09-23
**兼容性**: Python 3.8+
