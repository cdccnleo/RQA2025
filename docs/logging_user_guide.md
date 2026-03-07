# 日志系统使用指南

## 快速开始

### 1. 基本使用

```python
from src.infrastructure.logging import BaseLogger, LogLevel

# 创建Logger实例
logger = BaseLogger(name="my_app", level=LogLevel.INFO)

# 记录日志
logger.info("应用程序启动")
logger.warning("发现配置问题")
logger.error("数据库连接失败")
```

### 2. 使用便捷构造函数

```python
from src.infrastructure.logging import (
    create_base_logger,
    create_business_logger,
    create_audit_logger
)

# 基础Logger
app_logger = create_base_logger("my_app")

# 业务Logger
business_logger = create_business_logger("ecommerce")

# 审计Logger
audit_logger = create_audit_logger("security")
```

## 配置选项

### 日志级别设置

```python
from src.infrastructure.logging import LogLevel

logger = BaseLogger(name="app")

# 设置为DEBUG级别（显示所有日志）
logger.set_level(LogLevel.DEBUG)

# 设置为ERROR级别（只显示错误和严重错误）
logger.set_level(LogLevel.ERROR)

# 获取当前级别
current_level = logger.get_level()
```

### 日志格式配置

```python
from src.infrastructure.logging import LogFormat

logger = BaseLogger(name="app")

# 使用JSON格式（便于日志分析系统）
logger.set_formatter(LogFormat.JSON)

# 使用结构化格式（包含更多上下文信息）
logger.set_formatter(LogFormat.STRUCTURED)
```

### 自定义日志目录

```python
# 指定自定义日志目录
logger = BaseLogger(
    name="app",
    log_dir="/var/log/myapp"
)

# 业务日志分离存储
business_logger = create_business_logger(
    "business",
    "/var/log/business"
)
```

## 业务日志记录

### 记录业务事件

```python
business_logger = create_business_logger("order_service")

# 记录订单创建事件
business_logger.log_business_event(
    event_type="order_created",
    event_id="order_12345",
    user_id="user_67890",
    data={
        "amount": 299.99,
        "currency": "CNY",
        "items": [
            {"id": "item_1", "name": "商品A", "quantity": 2},
            {"id": "item_2", "name": "商品B", "quantity": 1}
        ]
    },
    status="success"
)

# 记录支付事件
business_logger.log_business_event(
    event_type="payment_completed",
    event_id="payment_12345",
    user_id="user_67890",
    data={
        "order_id": "order_12345",
        "amount": 299.99,
        "method": "alipay",
        "transaction_id": "txn_abcdef123456"
    },
    status="success"
)
```

### 记录失败事件

```python
# 记录失败的业务操作
business_logger.log_business_event(
    event_type="order_failed",
    event_id="order_12346",
    user_id="user_67890",
    data={
        "amount": 499.99,
        "error": "insufficient_funds",
        "reason": "账户余额不足"
    },
    status="failed"
)
```

## 审计日志记录

### 记录用户操作

```python
audit_logger = create_audit_logger("user_audit")

# 记录用户登录
audit_logger.log_operation(
    operation="user_login",
    user_id="user_67890",
    resource="session",
    action="create",
    success=True
)

# 记录数据访问
audit_logger.log_operation(
    operation="data_access",
    user_id="user_67890",
    resource="customer_records",
    action="read",
    success=True
)

# 记录权限变更
audit_logger.log_operation(
    operation="permission_change",
    user_id="admin_123",
    resource="user_67890/permissions",
    action="update",
    success=True
)
```

### 记录安全事件

```python
# 记录失败的登录尝试
audit_logger.log_security_event(
    event_type="failed_login",
    user_id="user_67890",
    details={
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0...",
        "attempt_count": 3,
        "lockout_duration": 300  # 5分钟
    },
    severity="medium"
)

# 记录可疑活动
audit_logger.log_security_event(
    event_type="suspicious_activity",
    user_id="user_67890",
    details={
        "activity": "bulk_data_export",
        "record_count": 10000,
        "source_ip": "10.0.0.5",
        "unusual_timing": True
    },
    severity="high"
)
```

## 基础设施日志

### 记录系统组件操作

```python
from src.infrastructure.logging import (
    get_infrastructure_logger,
    log_infrastructure_operation,
    log_infrastructure_error
)

# 获取基础设施Logger
cache_logger = get_infrastructure_logger("cache_manager")
db_logger = get_infrastructure_logger("database")

# 记录缓存操作
log_infrastructure_operation(
    "cache_warmup_started",
    {
        "cache_type": "redis",
        "expected_keys": 10000,
        "estimated_duration": 30
    }
)

# 记录数据库操作
db_logger.info("Database connection pool initialized", pool_size=10)

# 记录错误
log_infrastructure_error(
    "cache_connection_failed",
    {
        "service": "redis",
        "endpoint": "redis-cluster:6379",
        "error": "connection_timeout",
        "retry_count": 3
    }
)
```

## 高级用法

### 自定义处理器

```python
import logging
from src.infrastructure.logging import BaseLogger

logger = BaseLogger(name="advanced_app")

# 添加自定义处理器
custom_handler = logging.FileHandler("custom.log")
custom_handler.setLevel(logging.WARNING)
logger.add_handler(custom_handler)

# 现在WARNING及以上的日志会同时写入默认日志文件和custom.log
logger.warning("This goes to both files")
```

### 带额外上下文的日志

```python
logger = BaseLogger(name="context_app")

# 添加请求上下文
logger.info("Processing request",
    request_id="req_123",
    user_id="user_456",
    endpoint="/api/orders",
    method="POST",
    response_time=150,
    status_code=200
)

# 添加业务上下文
logger.info("Order processed",
    order_id="order_789",
    customer_id="cust_101",
    total_amount=299.99,
    payment_method="credit_card",
    fulfillment_center="FC001"
)
```

### 条件日志记录

```python
logger = BaseLogger(name="conditional_app")

# 只有在DEBUG级别时才记录详细信息
if logger.get_level() == LogLevel.DEBUG:
    logger.debug("Detailed debug information", {
        "stack_trace": "...",
        "memory_usage": "150MB",
        "active_connections": 25
    })

# 使用不同的日志级别处理不同严重程度的问题
def handle_error(error, context):
    if error.is_critical():
        logger.critical("Critical system error", error.details)
    elif error.is_recoverable():
        logger.warning("Recoverable error occurred", error.details)
    else:
        logger.error("Unexpected error", error.details)
```

## 最佳实践

### 1. 日志级别使用指南

- **DEBUG**: 开发调试信息，生产环境通常关闭
- **INFO**: 重要业务逻辑和系统状态变化
- **WARNING**: 潜在问题，不影响正常功能
- **ERROR**: 功能错误，需要关注
- **CRITICAL**: 系统级严重错误，可能导致服务中断

### 2. 日志内容规范

```python
# ✅ 推荐的日志格式
logger.info("User login successful", {
    "user_id": user.id,
    "login_method": "password",
    "ip_address": request.ip,
    "session_id": session.id
})

# ❌ 避免的日志格式
logger.info(f"User {user.name} logged in from {request.ip}")
# 问题：信息分散，不便于搜索和分析
```

### 3. 结构化日志记录

```python
# ✅ 使用结构化格式
logger.info("Payment processed", {
    "payment_id": "pay_123",
    "order_id": "ord_456",
    "amount": 99.99,
    "currency": "USD",
    "status": "completed",
    "processing_time_ms": 150,
    "payment_method": "stripe"
})

# 便于后续分析和监控
```

### 4. 错误日志记录

```python
try:
    # 业务逻辑
    process_payment(order)
except PaymentError as e:
    logger.error("Payment processing failed", {
        "order_id": order.id,
        "user_id": order.user_id,
        "amount": order.total,
        "error_code": e.code,
        "error_message": str(e),
        "retry_count": e.retry_count,
        "next_retry_at": e.next_retry_at.isoformat()
    })
    raise
```

### 5. 性能敏感日志

```python
import time

start_time = time.time()
result = expensive_operation()
duration = time.time() - start_time

# 只在性能异常时记录详细信息
if duration > 1.0:  # 超过1秒
    logger.warning("Slow operation detected", {
        "operation": "expensive_operation",
        "duration_seconds": duration,
        "threshold_seconds": 1.0,
        "input_size": len(input_data),
        "result_size": len(result)
    })
```

### 6. 日志分类使用

```python
# 业务日志 - 用于业务分析
business_logger.log_business_event("user_registration", ...)

# 审计日志 - 用于安全审计
audit_logger.log_operation("data_export", ...)

# 系统日志 - 用于系统监控
system_logger = BaseLogger("system", category=LogCategory.SYSTEM)
system_logger.info("System health check passed")

# 性能日志 - 用于性能监控
perf_logger = BaseLogger("performance", category=LogCategory.PERFORMANCE)
perf_logger.info("API response time", {"endpoint": "/api/orders", "duration_ms": 150})
```

## 故障排除

### 常见问题

#### 1. 日志文件没有生成

```python
# 检查日志目录权限
logger = BaseLogger(name="test", log_dir="/var/log/app")
stats = logger.get_stats()
print(f"Log directory: {stats['log_dir']}")

# 确保目录存在且可写
import os
log_dir = "/var/log/app"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
```

#### 2. 日志级别设置不生效

```python
logger = BaseLogger(name="test")

# 正确的设置方式
logger.set_level(LogLevel.DEBUG)
assert logger.get_level() == LogLevel.DEBUG

# 检查处理器级别
for handler in logger._handlers:
    print(f"Handler level: {handler.level}")
```

#### 3. 内存泄漏问题

```python
# 避免在循环中重复添加处理器
logger = BaseLogger(name="test")

# ❌ 错误：每次都添加新处理器
for i in range(100):
    logger.add_handler(logging.StreamHandler())  # 内存泄漏

# ✅ 正确：重用处理器
handler = logging.StreamHandler()
logger.add_handler(handler)
```

### 调试技巧

```python
# 查看Logger状态
logger = BaseLogger(name="debug_test")
stats = logger.get_stats()
print("Logger stats:", stats)

# 检查处理器配置
for i, handler in enumerate(logger._handlers):
    print(f"Handler {i}: {type(handler).__name__}, level={handler.level}")

# 测试日志输出
logger.set_level(LogLevel.DEBUG)
logger.debug("Debug test message")
logger.info("Info test message")
```

## 配置示例

### 生产环境配置

```python
from src.infrastructure.logging import create_business_logger, create_audit_logger, LogLevel, LogFormat

# 生产环境配置
business_logger = create_business_logger(
    "production_business",
    "/var/log/production/business"
)
business_logger.set_level(LogLevel.INFO)
business_logger.set_formatter(LogFormat.JSON)

audit_logger = create_audit_logger(
    "production_audit",
    "/var/log/production/audit"
)
audit_logger.set_level(LogLevel.INFO)
audit_logger.set_formatter(LogFormat.JSON)
```

### 开发环境配置

```python
from src.infrastructure.logging import create_base_logger, LogLevel, LogFormat

# 开发环境配置 - 更详细的日志
dev_logger = create_base_logger(
    "development",
    LogLevel.DEBUG,
    "logs/dev"
)
dev_logger.set_formatter(LogFormat.STRUCTURED)

# 同时输出到控制台和文件
console_handler = logging.StreamHandler()
dev_logger.add_handler(console_handler)
```

