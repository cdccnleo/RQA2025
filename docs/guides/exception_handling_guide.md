# 统一异常处理框架使用指南

## 📋 概述

RQA2025系统统一异常处理框架提供了一套完整的异常处理解决方案，包括异常类型定义、处理装饰器、监控告警和日志记录等功能。

## 🎯 核心特性

- **统一异常体系**: 标准化的异常类型和错误码
- **智能异常处理**: 基于策略的异常映射和重试机制
- **监控告警集成**: 实时异常监控和智能告警
- **多格式日志**: 支持结构化、简单、详细三种日志格式
- **易于使用**: 简单的装饰器接口，开箱即用

## 📚 异常类型体系

### 基础异常类

```python
from src.core.unified_exceptions import RQA2025Exception

# 所有系统异常的基类
class RQA2025Exception(Exception):
    def __init__(self, message: str, error_code: int = -1,
                 error_type: str = "UNKNOWN", context: Dict[str, Any] = None,
                 severity: str = "ERROR"):
        pass
```

### 业务层异常

```python
from src.core.unified_exceptions import (
    BusinessException, ValidationError, BusinessLogicError,
    WorkflowError, TradingError, RiskError, StrategyError
)

# 数据验证异常
ValidationError("用户名不能为空", field="username", value="")

# 业务逻辑异常
BusinessLogicError("账户余额不足", operation="withdraw", entity_id="user_123")

# 交易异常
TradingError("订单执行失败", order_id="order_456", symbol="AAPL")
```

### 基础设施层异常

```python
from src.core.unified_exceptions import (
    InfrastructureException, ConfigurationError, CacheError,
    LoggingError, DatabaseError, NetworkError, FileSystemError
)

# 配置异常
ConfigurationError("数据库连接配置缺失", config_key="database.host")

# 缓存异常
CacheError("缓存键不存在", cache_key="user_session")

# 数据库异常
DatabaseError("查询执行失败", operation="select", query="SELECT * FROM users")

# 网络异常
NetworkError("API调用超时", endpoint="https://api.example.com", status_code=504)
```

### 系统层异常

```python
from src.core.unified_exceptions import (
    SystemException, SecurityError, PerformanceError
)

# 安全异常
SecurityError("权限验证失败", user_id="user_123", permission="admin")

# 性能异常
PerformanceError("响应时间过长", metric_name="api_response_time", threshold=5000)
```

## 🛠️ 异常处理装饰器

### 通用异常处理装饰器

```python
from src.core.unified_exceptions import handle_exceptions

@handle_exceptions(
    service_name="user_service",
    log_level="warning",
    re_raise=True,
    enable_retry=True,
    enable_alerts=False
)
def create_user(username: str, email: str):
    """创建用户"""
    if not username:
        raise ValidationError("用户名不能为空", field="username")

    # 业务逻辑...
    return {"id": 123, "username": username, "email": email}
```

### 专用异常处理装饰器

```python
from src.core.unified_exceptions import (
    handle_business_exceptions,           # 业务逻辑异常处理
    handle_infrastructure_exceptions,     # 基础设施异常处理
    handle_system_exceptions,             # 系统异常处理
    handle_external_service_exceptions,   # 外部服务异常处理
    handle_database_exceptions,           # 数据库异常处理
    handle_network_exceptions             # 网络异常处理
)

# 业务服务
@handle_business_exceptions
def process_order(order_data: dict):
    """处理订单"""
    # 业务逻辑...
    pass

# 基础设施服务
@handle_infrastructure_exceptions
def load_config(config_path: str):
    """加载配置"""
    # 配置加载逻辑...
    pass

# 数据库操作
@handle_database_exceptions
def save_user(user: dict):
    """保存用户"""
    # 数据库操作...
    pass

# 外部API调用
@handle_external_service_exceptions
def call_external_api(endpoint: str, data: dict):
    """调用外部API"""
    # API调用逻辑...
    pass
```

## 🔄 重试机制

### 基本重试

```python
from src.core.unified_exceptions import handle_exceptions, RetryMechanism

# 使用装饰器启用重试
@handle_exceptions("api_client", enable_retry=True)
def call_api(endpoint: str):
    # API调用逻辑，如果失败会自动重试
    pass

# 手动使用重试机制
retry = RetryMechanism(max_attempts=5, delay_seconds=1.0, backoff_factor=2.0)
result = retry.execute_with_retry(call_api, "https://api.example.com")
```

### 自定义重试策略

```python
from src.core.unified_exceptions import ExceptionHandlingStrategy

# 自定义策略
strategy = ExceptionHandlingStrategy(
    service_name="critical_service",
    log_level="error",
    re_raise=True,
    enable_alerts=True
)

# 获取重试策略
retry_policy = strategy.get_retry_policy(NetworkError("连接超时"))
if retry_policy:
    print(f"重试配置: {retry_policy}")
    # {'max_attempts': 3, 'delay_seconds': 1, 'backoff_factor': 2}
```

## 📊 监控和告警

### 初始化监控系统

```python
from src.core.unified_exceptions import (
    init_exception_monitoring, add_exception_alert_callback
)

# 初始化监控
init_exception_monitoring("my_application")

# 添加告警回调
def email_alert(alert_data: dict):
    """发送邮件告警"""
    print(f"发送告警邮件: {alert_data['message']}")

def slack_alert(alert_data: dict):
    """发送Slack告警"""
    print(f"发送Slack通知: {alert_data['message']}")

add_exception_alert_callback(email_alert)
add_exception_alert_callback(slack_alert)
```

### 配置告警阈值

```python
from src.core.unified_exceptions import configure_exceptions

# 配置告警阈值
configure_exceptions({
    'alert_thresholds': {
        'error_rate_per_minute': 10,      # 每分钟错误率阈值
        'critical_errors_per_hour': 5,    # 每小时严重错误阈值
        'repeated_errors_threshold': 5    # 重复错误告警阈值
    }
})
```

### 获取监控报告

```python
from src.core.unified_exceptions import get_exception_health_report

# 获取健康报告
report = get_exception_health_report()
print(f"监控状态: {report['monitoring_active']}")
print(f"异常总数: {report['statistics']['total_exceptions']}")
print(f"告警回调数量: {report['alert_callbacks_count']}")
```

## 📝 日志记录

### 设置日志格式

```python
from src.core.unified_exceptions import set_exception_log_format

# 设置为结构化日志（JSON格式，便于日志分析系统）
set_exception_log_format('structured')

# 设置为简单日志（便于人工阅读）
set_exception_log_format('simple')

# 设置为详细日志（包含完整堆栈信息）
set_exception_log_format('detailed')
```

### 手动日志记录

```python
from src.core.unified_exceptions import global_exception_logger, ValidationError

try:
    # 一些可能失败的操作
    raise ValidationError("数据验证失败", field="email")
except ValidationError as e:
    # 手动记录异常
    global_exception_logger.log_exception(e, {
        "operation": "user_registration",
        "user_id": "user_123",
        "additional_info": "注册失败原因分析"
    })
```

## ✅ 参数验证

### 内置验证函数

```python
from src.core.unified_exceptions import (
    validate_not_none, validate_range, validate_string_length
)

def create_post(title: str, content: str, rating: int):
    """创建帖子"""
    # 验证必填参数
    validate_not_none(title, "title")
    validate_not_none(content, "content")

    # 验证数值范围
    validate_range(rating, 1, 5, "rating")

    # 验证字符串长度
    validate_string_length(title, 1, 200, "title")
    validate_string_length(content, 10, 10000, "content")

    # 业务逻辑...
    return {"title": title, "content": content, "rating": rating}
```

### 自定义验证

```python
from src.core.unified_exceptions import ValidationError

def validate_email(email: str):
    """验证邮箱格式"""
    import re
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        raise ValidationError("邮箱格式不正确", field="email", value=email)

def validate_password_strength(password: str):
    """验证密码强度"""
    if len(password) < 8:
        raise ValidationError("密码长度不能少于8位", field="password")

    if not any(c.isupper() for c in password):
        raise ValidationError("密码必须包含大写字母", field="password")

    if not any(c.islower() for c in password):
        raise ValidationError("密码必须包含小写字母", field="password")

    if not any(c.isdigit() for c in password):
        raise ValidationError("密码必须包含数字", field="password")
```

## 🎨 配置管理

### 更新配置

```python
from src.core.unified_exceptions import configure_exceptions, get_exception_config

# 更新配置
configure_exceptions({
    'monitoring_enabled': True,
    'alerts_enabled': True,
    'log_format': 'structured',
    'retry_enabled': True
})

# 获取配置
config = get_exception_config()
log_format = get_exception_config('log_format')
thresholds = get_exception_config('alert_thresholds')
```

### 重置配置

```python
from src.core.unified_exceptions import ExceptionConfiguration

config_manager = ExceptionConfiguration()
config_manager.reset_to_defaults()
```

## 📈 统计和分析

### 获取异常统计

```python
from src.core.unified_exceptions import get_exception_stats

# 获取全局异常统计
stats = get_exception_stats()
print(f"总异常数: {stats['total_exceptions']}")
print(f"异常类型分布: {stats['exception_types']}")

# 最近异常
for exc in stats['recent_exceptions']:
    print(f"{exc['timestamp']}: {exc['error_type']} - {exc['message']}")
```

### 自定义统计分析

```python
from src.core.unified_exceptions import global_exception_stats

# 分析特定时间段的异常
import datetime
one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)

recent_errors = [
    exc for exc in global_exception_stats.recent_exceptions
    if datetime.datetime.fromisoformat(exc['timestamp']) > one_hour_ago
]

print(f"过去1小时异常数: {len(recent_errors)}")

# 按严重程度统计
severity_count = {}
for exc in recent_errors:
    severity = exc['severity']
    severity_count[severity] = severity_count.get(severity, 0) + 1

print(f"严重程度分布: {severity_count}")
```

## 🚀 最佳实践

### 1. 选择合适的异常类型

```python
# ✅ 推荐：使用具体的异常类型
def process_payment(amount: float, card_number: str):
    if amount <= 0:
        raise ValidationError("支付金额必须大于0", field="amount", value=amount)

    if not card_number:
        raise ValidationError("卡号不能为空", field="card_number")

    # 模拟支付失败
    raise BusinessLogicError("支付处理失败，请稍后重试", operation="payment")

# ❌ 避免：使用通用异常
def process_payment_bad(amount: float, card_number: str):
    if amount <= 0:
        raise Exception("金额无效")  # 不够具体

    raise RQA2025Exception("支付失败")  # 缺少上下文
```

### 2. 提供有意义的上下文信息

```python
# ✅ 推荐：提供详细上下文
def transfer_money(from_account: str, to_account: str, amount: float):
    try:
        # 转账逻辑...
        pass
    except Exception as e:
        raise BusinessLogicError(
            f"转账失败: {from_account} -> {to_account}",
            operation="transfer",
            context={
                "from_account": from_account,
                "to_account": to_account,
                "amount": amount,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

# ❌ 避免：缺少上下文
def transfer_money_bad(from_account: str, to_account: str, amount: float):
    raise BusinessLogicError("转账失败")  # 缺少详细信息
```

### 3. 合理使用重试机制

```python
# ✅ 推荐：对瞬时故障使用重试
@handle_external_service_exceptions
def call_payment_gateway(payment_data: dict):
    # API调用可能由于网络问题失败，应该重试
    response = requests.post("https://payment.api/gateway", json=payment_data)
    response.raise_for_status()
    return response.json()

# ❌ 避免：对确定性错误使用重试
@handle_business_exceptions
def validate_user_age(age: int):
    if age < 18:
        raise ValidationError("年龄必须大于等于18岁", field="age", value=age)
    # 这个验证失败不是暂时的，不应该重试
```

### 4. 正确处理异常层次

```python
# ✅ 推荐：让异常自然传播，除非需要转换
@handle_business_exceptions
def complex_business_operation(data: dict):
    try:
        # 第一步
        step1_result = validate_and_process_step1(data)

        # 第二步
        step2_result = process_step2(step1_result)

        # 第三步
        return finalize_processing(step2_result)

    except ValidationError:
        # 验证错误直接重新抛出，不需要转换
        raise
    except BusinessLogicError:
        # 业务逻辑错误也直接重新抛出
        raise
    except Exception as e:
        # 其他异常转换为业务异常
        raise BusinessLogicError(
            f"业务操作失败: {str(e)}",
            operation="complex_business_operation"
        ) from e
```

### 5. 使用监控和告警

```python
# ✅ 推荐：为关键服务启用告警
@handle_infrastructure_exceptions
def connect_to_database():
    """连接数据库 - 关键基础设施服务"""
    # 这个函数失败应该触发告警
    pass

@handle_business_exceptions
def process_user_registration(user_data: dict):
    """用户注册 - 重要业务功能"""
    # 这个函数的异常应该被监控但不一定需要告警
    pass
```

## 🔧 故障排除

### 常见问题

#### 1. 异常没有被正确记录

```python
# 检查监控是否启用
from src.core.unified_exceptions import get_exception_config
config = get_exception_config()
print(f"监控启用: {config['monitoring_enabled']}")

# 手动初始化监控
from src.core.unified_exceptions import init_exception_monitoring
init_exception_monitoring("my_service")
```

#### 2. 装饰器没有生效

```python
# 检查导入是否正确
try:
    from src.core.unified_exceptions import handle_exceptions
    print("导入成功")
except ImportError as e:
    print(f"导入失败: {e}")

# 检查装饰器语法
@handle_exceptions("test_service")  # 注意不要忘记调用装饰器
def my_function():
    pass
```

#### 3. 告警没有触发

```python
# 检查告警配置
from src.core.unified_exceptions import get_exception_config
config = get_exception_config()
print(f"告警启用: {config['alerts_enabled']}")

# 检查是否有告警回调
from src.core.unified_exceptions import global_exception_monitor
print(f"告警回调数量: {len(global_exception_monitor.alert_callbacks)}")

# 检查告警阈值
thresholds = get_exception_config('alert_thresholds')
print(f"告警阈值: {thresholds}")
```

## 📖 相关文档

- [异常处理最佳实践](./best_practices/exception_handling_best_practices.md)
- [架构设计指南](./architecture/architecture_design_guide.md)
- [监控和告警配置](./configuration/monitoring_alerting_config.md)

---

*最后更新: 2025年10月3日*

