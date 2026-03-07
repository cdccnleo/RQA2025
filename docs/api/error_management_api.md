# RQA2025 基础设施层错误管理 API 文档

## 📋 文档概述

本文档详细描述了RQA2025基础设施层错误管理模块的API接口、核心组件和使用方法。该模块提供统一的企业级错误处理、异常管理、重试机制和熔断器功能。

**版本信息**: v1.0.0
**更新日期**: 2025年9月23日
**模块路径**: `src/infrastructure/error/`

## 🏗️ 架构概述

错误管理模块采用分层架构设计：

```
错误管理模块
├── core/           # 核心接口和基础组件
├── handlers/       # 错误处理器实现
├── exceptions/     # 统一异常定义
├── policies/       # 错误处理策略
└── tests/         # 单元测试
```

## 🔧 核心组件

### 1. ErrorHandler - 通用错误处理器

#### 类定义
```python
class ErrorHandler(IErrorHandler):
    def __init__(self, max_history: int = 1000)
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
    def register_handler(self, error_type: Type[Exception], handler: Callable) -> None
    def register_strategy(self, strategy_name: str, strategy: Callable) -> None
    def get_error_history(self) -> List[Dict[str, Any]]
    def clear_history(self) -> None
    def get_stats(self) -> Dict[str, Any]
```

#### 主要方法

##### `handle_error(error, context=None)`
处理错误并返回处理结果。

**参数**:
- `error` (Exception): 要处理的异常对象
- `context` (Optional[Dict[str, Any]]): 错误上下文信息

**返回值**:
```python
{
    'handled': bool,           # 是否成功处理
    'error_type': str,         # 错误类型
    'message': str,           # 错误消息
    'severity': str,          # 严重程度 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    'category': str,          # 错误类别 (SYSTEM/BUSINESS/NETWORK/DATABASE/...)
    'context': dict,          # 原始上下文
    'error_context': dict     # 错误上下文详情
}
```

**示例**:
```python
from infrastructure.error import ErrorHandler

handler = ErrorHandler()
try:
    # 一些可能出错的操作
    result = risky_operation()
except Exception as e:
    result = handler.handle_error(e, {'operation': 'risky_operation'})
    if not result['handled']:
        print(f"未处理的错误: {result['message']}")
```

##### `register_handler(error_type, handler)`
注册特定类型错误的处理器。

**参数**:
- `error_type` (Type[Exception]): 异常类型
- `handler` (Callable): 处理函数，签名 `(error, context) -> Dict[str, Any]`

**示例**:
```python
def handle_network_error(error, context):
    return {
        'handled': True,
        'recovery_action': 'retry',
        'message': f"网络错误已处理: {error}"
    }

handler.register_handler(ConnectionError, handle_network_error)
```

##### `get_stats()`
获取错误处理统计信息。

**返回值**:
```python
{
    'total_errors': int,           # 总错误数
    'errors_by_type': dict,        # 按类型统计
    'registered_handlers': int,    # 已注册处理器数
    'registered_strategies': int,  # 已注册策略数
    'max_history': int,           # 最大历史记录数
    'current_history_size': int   # 当前历史记录数
}
```

### 2. CircuitBreaker - 熔断器

#### 类定义
```python
class CircuitBreaker(ICircuitBreaker):
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, name: str = "default")
    def call(self, func: Callable, *args, **kwargs) -> Any
    def get_status(self) -> Dict[str, Any]
    def reset(self) -> None
    def trip(self) -> None
```

#### 主要方法

##### `call(func, *args, **kwargs)`
通过熔断器执行函数调用。

**参数**:
- `func` (Callable): 要执行的函数
- `*args`, `**kwargs`: 函数参数

**返回值**: 函数执行结果

**异常**: 当熔断器开启时抛出异常

**示例**:
```python
from infrastructure.error import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

def unreliable_api_call():
    # 可能失败的API调用
    return call_external_api()

try:
    result = breaker.call(unreliable_api_call)
    print(f"API调用成功: {result}")
except Exception as e:
    print(f"熔断器已开启或调用失败: {e}")
```

##### `get_status()`
获取熔断器状态信息。

**返回值**:
```python
{
    'name': str,                    # 熔断器名称
    'state': str,                   # 状态 (closed/open/half_open)
    'failure_count': int,           # 失败次数
    'success_count': int,           # 成功次数
    'failure_threshold': int,       # 失败阈值
    'recovery_timeout': float,      # 恢复超时时间
    'last_failure_time': float,     # 最后失败时间
    'time_since_last_failure': float # 自最后失败以来时间
}
```

### 3. RetryPolicy - 重试策略

#### 类定义
```python
class RetryPolicy(IRetryPolicy):
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, strategy: str = "exponential", jitter: bool = True)
    def execute(self, func: Callable, *args, **kwargs) -> Any
    def get_retry_stats(self) -> Dict[str, Any]
    def reset_stats(self) -> None
```

#### 主要方法

##### `execute(func, *args, **kwargs)`
使用重试策略执行函数。

**参数**:
- `func` (Callable): 要执行的函数
- `*args`, `**kwargs`: 函数参数

**返回值**: 函数执行结果

**异常**: 当所有重试都失败时抛出最后一次异常

**示例**:
```python
from infrastructure.error import RetryPolicy

policy = RetryPolicy(max_retries=3, strategy='exponential')

def flaky_operation():
    # 可能偶尔失败的操作
    if random.random() < 0.7:  # 70%概率失败
        raise ConnectionError("Temporary network issue")
    return "success"

try:
    result = policy.execute(flaky_operation)
    print(f"操作成功: {result}")
except Exception as e:
    print(f"所有重试都失败: {e}")
```

##### 支持的重试策略

| 策略 | 描述 | 示例延迟序列 |
|-----|------|-------------|
| `fixed` | 固定间隔 | 1.0s, 1.0s, 1.0s |
| `linear` | 线性递增 | 1.0s, 2.0s, 3.0s |
| `exponential` | 指数递增 | 1.0s, 2.0s, 4.0s |
| `random` | 随机间隔 | 0.5-1.5s, 0.5-1.5s |

## 🚨 异常体系

### 统一异常类层次

```
InfrastructureError (基类)
├── DataLoaderError         # 数据加载错误
├── ConfigurationError      # 配置错误
├── NetworkError           # 网络错误
├── DatabaseError          # 数据库错误
├── CacheError             # 缓存错误
├── SecurityError          # 安全错误
├── SystemError            # 系统错误
├── CriticalError          # 严重错误
├── WarningError           # 警告错误
├── InfoLevelError         # 信息级别错误
├── RetryableError         # 可重试错误
│   ├── RetryError        # 重试错误
│   └── CircuitBreakerOpenError  # 熔断器开启错误
└── TradingError           # 交易相关错误
    ├── OrderRejectedError
    └── InvalidPriceError
```

### 错误代码体系

| 错误代码范围 | 分类 | 示例 |
|-------------|------|------|
| 1000-1999 | 数据相关 | `DATA_NOT_FOUND = 1001` |
| 2000-2999 | 配置相关 | `CONFIG_INVALID = 2002` |
| 3000-3999 | 网络相关 | `NETWORK_ERROR = 3001` |
| 4000-4999 | 数据库相关 | `DATABASE_ERROR = 4001` |
| 5000-5999 | 缓存相关 | `CACHE_ERROR = 5001` |
| 6000-6999 | 安全相关 | `AUTHENTICATION_ERROR = 6002` |
| 9000-9999 | 系统相关 | `SYSTEM_ERROR = 9000` |

### 异常工具函数

#### `is_retryable_error(error)`
检查错误是否可重试。

```python
from infrastructure.error import is_retryable_error, RetryableError

error = RetryableError("可重试错误")
print(is_retryable_error(error))  # True
```

#### `get_error_code(error)`
获取错误的错误代码。

```python
from infrastructure.error import get_error_code, InfrastructureError, ErrorCode

error = InfrastructureError("测试", ErrorCode.DATA_NOT_FOUND)
print(get_error_code(error))  # ErrorCode.DATA_NOT_FOUND
```

#### `get_error_severity(error)`
获取错误严重程度。

```python
from infrastructure.error import get_error_severity, CriticalError

error = CriticalError("严重错误")
print(get_error_severity(error))  # "CRITICAL"
```

## 📊 使用模式

### 1. 基础错误处理

```python
from infrastructure.error import ErrorHandler

# 创建错误处理器
handler = ErrorHandler()

# 处理错误
try:
    risky_operation()
except Exception as e:
    result = handler.handle_error(e, {'operation': 'risky_operation'})
    if not result['handled']:
        # 处理未处理的错误
        log_error(result)
```

### 2. 熔断器保护

```python
from infrastructure.error import CircuitBreaker

# 创建熔断器
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

# 保护外部调用
def call_external_service():
    return breaker.call(external_api_call)

try:
    result = call_external_service()
except Exception as e:
    print(f"服务调用失败: {e}")
```

### 3. 重试机制

```python
from infrastructure.error import RetryPolicy

# 创建重试策略
policy = RetryPolicy(max_retries=3, strategy='exponential')

# 执行可能失败的操作
result = policy.execute(unreliable_operation)
```

### 4. 自定义错误处理器

```python
from infrastructure.error import ErrorHandler, DatabaseError

handler = ErrorHandler()

# 注册数据库错误处理器
def handle_db_error(error, context):
    # 记录到监控系统
    monitor.record_error(error, context)

    # 尝试恢复连接
    if isinstance(error, ConnectionError):
        return {'handled': True, 'action': 'reconnect'}

    return {'handled': False, 'message': '数据库错误'}

handler.register_handler(DatabaseError, handle_db_error)
```

### 5. 综合使用

```python
from infrastructure.error import (
    ErrorHandler, CircuitBreaker, RetryPolicy,
    InfrastructureError, ErrorCode
)

# 创建组件
handler = ErrorHandler()
breaker = CircuitBreaker(failure_threshold=3)
policy = RetryPolicy(max_retries=2)

# 综合保护的API调用
def robust_api_call():
    def api_call():
        try:
            return breaker.call(make_http_request)
        except Exception as e:
            # 转换为基础设施异常
            raise InfrastructureError(
                f"API调用失败: {e}",
                ErrorCode.NETWORK_ERROR,
                {'original_error': str(e)}
            ) from e

    # 使用重试策略
    return policy.execute(api_call)

# 使用
try:
    result = robust_api_call()
    print(f"调用成功: {result}")
except Exception as e:
    # 使用错误处理器处理最终失败
    error_result = handler.handle_error(e, {'operation': 'api_call'})
    print(f"最终失败: {error_result['message']}")
```

## 🔧 配置选项

### ErrorHandler配置

```python
handler = ErrorHandler(
    max_history=1000  # 最大错误历史记录数
)
```

### CircuitBreaker配置

```python
breaker = CircuitBreaker(
    failure_threshold=5,    # 失败阈值
    recovery_timeout=60.0,  # 恢复超时时间(秒)
    name="api_service"      # 熔断器名称
)
```

### RetryPolicy配置

```python
policy = RetryPolicy(
    max_retries=3,           # 最大重试次数
    base_delay=1.0,          # 基础延迟时间
    max_delay=60.0,          # 最大延迟时间
    strategy="exponential",  # 重试策略
    jitter=True              # 启用抖动
)
```

## 📈 性能考虑

### 内存使用
- `ErrorHandler` 的 `max_history` 参数控制内存使用
- 默认历史记录限制为1000条，可根据需要调整

### 线程安全
- 所有组件都是线程安全的
- 使用适当的锁机制保护共享状态

### 异步支持
- 支持在异步环境中使用
- 但需要注意熔断器状态的并发访问

## 🔍 监控和调试

### 统计信息获取

```python
# ErrorHandler统计
stats = handler.get_stats()
print(f"总错误数: {stats['total_errors']}")
print(f"已注册处理器: {stats['registered_handlers']}")

# CircuitBreaker状态
status = breaker.get_status()
print(f"熔断器状态: {status['state']}")

# RetryPolicy统计
retry_stats = policy.get_retry_stats()
print(f"重试策略: {retry_stats['strategy']}")
```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用调试日志查看内部处理流程
handler = ErrorHandler()
```

## 🚨 最佳实践

### 1. 异常处理层次
- 在业务层使用具体的异常类型
- 在基础设施层转换为统一异常
- 在顶层使用错误处理器进行最终处理

### 2. 熔断器使用
- 为每个外部依赖配置独立的熔断器
- 设置合理的失败阈值和恢复时间
- 监控熔断器状态和切换事件

### 3. 重试策略选择
- 网络调用使用指数退避
- 数据库操作使用固定间隔
- 对于idempotent操作才使用重试

### 4. 错误分类
- 使用统一的错误代码体系
- 根据严重程度进行不同处理
- 为不同错误类型配置合适的恢复策略

### 5. 监控告警
- 监控错误率和处理成功率
- 设置熔断器状态变更告警
- 跟踪重试成功率和恢复时间

## 📚 相关文档

- [基础设施层架构设计](../architecture/infrastructure_architecture_design.md)
- [错误处理测试文档](error_management_tests.md)
- [监控告警集成](monitoring_integration.md)

---

*本文档基于RQA2025基础设施层错误管理模块的实际实现编写，提供了完整的企业级错误处理解决方案。*
