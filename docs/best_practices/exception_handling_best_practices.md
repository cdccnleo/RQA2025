# 异常处理最佳实践

## 📋 概述

本文档描述了RQA2025系统中异常处理的最佳实践，包括设计原则、代码规范、性能考虑和监控策略。

## 🎯 设计原则

### 1. 异常层次化设计

#### 原则
- 使用分层的异常体系，从具体到通用
- 保持异常类型的单一职责
- 提供足够的上下文信息

#### 最佳实践
```python
# ✅ 推荐：具体异常类型
class InsufficientFundsError(BusinessLogicError):
    """账户余额不足异常"""

    def __init__(self, account_id: str, required: float, available: float):
        super().__init__(
            f"账户 {account_id} 余额不足，需要 {required}，可用 {available}",
            operation="withdraw",
            entity_id=account_id,
            context={
                "required_amount": required,
                "available_amount": available
            }
        )

# ❌ 避免：过于通用的异常
raise Exception("操作失败")  # 缺少具体信息和类型
```

### 2. 异常信息完整性

#### 原则
- 提供问题描述、原因和解决建议
- 包含相关的上下文数据
- 使用标准化的错误码

#### 最佳实践
```python
# ✅ 推荐：完整的异常信息
def validate_trade_request(symbol: str, quantity: int, price: float):
    if not symbol:
        raise ValidationError(
            "交易代码不能为空",
            field="symbol",
            context={
                "operation": "validate_trade_request",
                "provided_value": symbol,
                "suggestion": "请提供有效的交易代码，如 'AAPL' 或 '000001.SZ'"
            }
        )

    if quantity <= 0:
        raise ValidationError(
            f"交易数量必须大于0，当前值: {quantity}",
            field="quantity",
            value=quantity,
            context={
                "min_value": 1,
                "suggestion": "请提供正整数交易数量"
            }
        )
```

## 🛠️ 代码规范

### 1. 异常处理位置

#### 在正确的位置捕获异常
```python
# ✅ 推荐：在服务边界捕获异常
class UserService:
    @handle_business_exceptions
    def create_user(self, user_data: dict) -> User:
        # 验证输入
        self._validate_user_data(user_data)

        # 执行业务逻辑
        user = self._create_user_entity(user_data)

        # 保存到数据库
        return self._save_user(user)

    def _validate_user_data(self, data: dict):
        """内部验证方法，不使用装饰器"""
        if not data.get('email'):
            raise ValidationError("邮箱不能为空", field="email")

    def _create_user_entity(self, data: dict) -> User:
        """内部业务逻辑方法"""
        return User(**data)

    def _save_user(self, user: User) -> User:
        """内部数据访问方法"""
        return self.repository.save(user)
```

#### 避免在不该捕获的地方捕获
```python
# ❌ 避免：在过早的位置捕获异常
def process_data(data):
    try:
        # 数据处理逻辑
        result = validate_and_transform(data)
        return save_result(result)
    except Exception as e:
        # 捕获了所有异常，隐藏了具体问题
        logger.error(f"数据处理失败: {e}")
        return None
```

### 2. 异常转换

#### 保持异常链
```python
# ✅ 推荐：使用 'from e' 保持异常链
@handle_business_exceptions
def business_operation(data):
    try:
        return infrastructure_operation(data)
    except InfrastructureException as e:
        # 转换为业务异常，但保持原始异常信息
        raise BusinessLogicError(
            f"业务操作失败: {e.message}",
            operation="business_operation"
        ) from e
```

#### 避免过度包装
```python
# ❌ 避免：过度包装异常
try:
    risky_operation()
except ValueError as e:
    raise BusinessLogicError("操作失败") from e
    # 丢失了原始的 ValueError 信息

# ✅ 推荐：保留原始异常信息
try:
    risky_operation()
except ValueError as e:
    raise BusinessLogicError(f"无效参数: {str(e)}") from e
```

### 3. 资源清理

#### 使用上下文管理器
```python
# ✅ 推荐：使用 try/finally 或上下文管理器
@handle_infrastructure_exceptions
def process_file(file_path: str):
    file_handle = None
    try:
        file_handle = open(file_path, 'r')
        data = file_handle.read()
        return process_data(data)
    except FileNotFoundError:
        raise FileSystemError(f"文件不存在: {file_path}", file_path=file_path)
    finally:
        if file_handle:
            file_handle.close()

# 更好的方式：使用上下文管理器
@handle_infrastructure_exceptions
def process_file_better(file_path: str):
    try:
        with open(file_path, 'r') as f:
            data = f.read()
            return process_data(data)
    except FileNotFoundError:
        raise FileSystemError(f"文件不存在: {file_path}", file_path=file_path)
```

## ⚡ 性能考虑

### 1. 异常开销

#### 了解异常的性能成本
```python
# ✅ 推荐：对预期情况使用条件检查
def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ValidationError("除数不能为0", field="divisor", value=b)
    return a / b

# ❌ 避免：对预期情况使用异常
def find_user(user_id: str) -> User:
    user = self.repository.find_by_id(user_id)
    if user is None:
        raise BusinessLogicError(f"用户不存在: {user_id}")
    return user
```

#### 批量操作的异常处理
```python
# ✅ 推荐：批量操作时收集错误
def batch_process_orders(orders: List[dict]) -> dict:
    results = []
    errors = []

    for order in orders:
        try:
            result = self.process_single_order(order)
            results.append(result)
        except Exception as e:
            errors.append({
                'order': order,
                'error': str(e),
                'error_type': type(e).__name__
            })

    return {
        'processed': results,
        'errors': errors,
        'success_rate': len(results) / len(orders) if orders else 0
    }
```

### 2. 日志和监控

#### 选择合适的日志级别
```python
# ✅ 推荐：根据严重程度选择日志级别
@handle_business_exceptions
def process_payment(payment_data: dict):
    try:
        # 正常业务逻辑
        return self.payment_gateway.charge(payment_data)
    except NetworkError as e:
        # 网络问题，警告级别
        logger.warning(f"支付网关连接失败，将重试: {e}")
        raise
    except ValidationError as e:
        # 输入验证失败，信息级别
        logger.info(f"支付数据验证失败: {e}")
        raise
    except BusinessLogicError as e:
        # 业务逻辑错误，错误级别
        logger.error(f"支付业务逻辑错误: {e}")
        raise
```

#### 避免日志泛滥
```python
# ❌ 避免：重复记录已处理的异常
@handle_exceptions("service_name")  # 装饰器已记录异常
def operation_with_logging():
    try:
        risky_operation()
    except Exception as e:
        logger.error(f"操作失败: {e}")  # 重复记录
        raise

# ✅ 推荐：让装饰器处理日志记录
@handle_exceptions("service_name")
def operation_clean():
    risky_operation()  # 异常由装饰器处理和记录
```

## 📊 监控和告警

### 1. 告警阈值设置

#### 基于业务场景设置阈值
```python
# 为不同服务设置不同的告警阈值
configure_exceptions({
    'alert_thresholds': {
        # 高频API服务
        'api_service': {
            'error_rate_per_minute': 5,    # 5%错误率
            'critical_errors_per_hour': 50  # 50个严重错误/小时
        },
        # 批处理服务
        'batch_service': {
            'error_rate_per_minute': 20,   # 20%错误率（批处理更宽松）
            'critical_errors_per_hour': 10  # 10个严重错误/小时
        },
        # 核心交易服务
        'trading_service': {
            'error_rate_per_minute': 1,    # 1%错误率（非常严格）
            'critical_errors_per_hour': 5   # 5个严重错误/小时
        }
    }
})
```

### 2. 告警升级策略

#### 实现告警升级
```python
from src.core.unified_exceptions import add_exception_alert_callback

def escalating_alert_handler(alert_data: dict):
    """告警升级处理"""
    severity = alert_data['severity']
    error_type = alert_data['alert_type']

    if severity == 'critical':
        # 严重告警：立即通知所有相关人员
        send_immediate_alert(alert_data)
        escalate_to_management(alert_data)

    elif severity == 'warning':
        if error_type == 'repeated_errors':
            # 重复错误警告：如果持续发生则升级
            if is_persistent_error(alert_data):
                send_warning_alert(alert_data)
        else:
            # 普通警告：只记录和监控
            log_warning(alert_data)

def is_persistent_error(alert_data: dict) -> bool:
    """检查是否为持续性错误"""
    # 检查最近1小时内的相同错误数量
    from src.core.unified_exceptions import global_exception_stats
    recent_similar = sum(1 for exc in global_exception_stats.recent_exceptions[-60:]
                        if exc['error_type'] == alert_data['exception']['error_type'])
    return recent_similar > 10

add_exception_alert_callback(escalating_alert_handler)
```

### 3. 异常趋势分析

#### 监控异常趋势
```python
def analyze_exception_trends():
    """分析异常趋势"""
    from src.core.unified_exceptions import get_exception_stats

    stats = get_exception_stats()

    # 计算错误率趋势
    error_rate_trend = calculate_error_rate_trend(stats)

    # 识别最常见的异常类型
    top_errors = get_top_error_types(stats, top_n=5)

    # 检查是否有新的异常类型出现
    new_error_types = detect_new_error_types(stats)

    return {
        'error_rate_trend': error_rate_trend,
        'top_errors': top_errors,
        'new_error_types': new_error_types,
        'recommendations': generate_recommendations(error_rate_trend, top_errors)
    }

def generate_recommendations(error_rate_trend, top_errors):
    """生成改进建议"""
    recommendations = []

    if error_rate_trend == 'increasing':
        recommendations.append("错误率呈上升趋势，建议检查最近的代码变更")

    for error_type, count in top_errors.items():
        if error_type == 'VALIDATION_ERROR':
            recommendations.append(f"验证错误频繁 ({count}次)，建议改进输入验证逻辑")
        elif error_type == 'NETWORK_ERROR':
            recommendations.append(f"网络错误频繁 ({count}次)，建议检查网络连接和重试策略")

    return recommendations
```

## 🔍 调试和故障排除

### 1. 异常链追踪

#### 使用正确的异常链
```python
# ✅ 推荐：保持完整的异常链
def complex_operation():
    try:
        step1()
    except Exception as e:
        try:
            step2()
        except Exception as e2:
            # 保留原始异常信息
            raise BusinessLogicError("多步操作失败") from e2
        else:
            # 如果step2成功，但step1失败了，需要重新抛出step1的异常
            raise
```

### 2. 异常上下文增强

#### 添加调试信息
```python
# ✅ 推荐：在异常中包含调试上下文
def process_transaction(transaction: dict):
    """处理交易"""
    context = {
        'transaction_id': transaction.get('id'),
        'user_id': transaction.get('user_id'),
        'amount': transaction.get('amount'),
        'timestamp': datetime.now().isoformat(),
        'processing_step': 'initial_validation'
    }

    try:
        # 验证交易
        validate_transaction(transaction, context)

        context['processing_step'] = 'risk_check'
        # 风险检查
        check_risk(transaction, context)

        context['processing_step'] = 'execution'
        # 执行交易
        return execute_transaction(transaction, context)

    except Exception as e:
        # 增强异常上下文
        if hasattr(e, 'context'):
            e.context.update(context)
        else:
            e.context = context
        raise
```

### 3. 异常测试

#### 编写异常处理测试
```python
import pytest
from unittest.mock import patch

def test_exception_handling():
    """测试异常处理"""
    service = UserService()

    # 测试正常情况
    user_data = {'email': 'test@example.com', 'name': 'Test User'}
    user = service.create_user(user_data)
    assert user.email == 'test@example.com'

    # 测试验证异常
    with pytest.raises(ValidationError) as exc_info:
        service.create_user({'name': 'Test User'})  # 缺少email

    assert exc_info.value.field == 'email'
    assert '不能为空' in str(exc_info.value)

    # 测试业务异常
    with patch.object(service.repository, 'save', side_effect=Exception("DB error")):
        with pytest.raises(BusinessLogicError) as exc_info:
            service.create_user(user_data)

        assert '业务操作失败' in str(exc_info.value)

def test_retry_mechanism():
    """测试重试机制"""
    from src.core.unified_exceptions import RetryMechanism

    call_count = 0
    def failing_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("临时连接失败")
        return "success"

    retry = RetryMechanism(max_attempts=5, delay_seconds=0.1)  # 快速重试用于测试

    result = retry.execute_with_retry(failing_operation)
    assert result == "success"
    assert call_count == 3  # 前两次失败，第三次成功
```

## 📈 度量和改进

### 1. 异常处理指标

#### 跟踪关键指标
```python
def get_exception_metrics():
    """获取异常处理指标"""
    from src.core.unified_exceptions import (
        get_exception_stats, get_exception_health_report
    )

    stats = get_exception_stats()
    health = get_exception_health_report()

    return {
        'total_exceptions': stats['total_exceptions'],
        'error_types_count': len(stats['exception_types']),
        'most_common_error': max(stats['exception_types'].items(), key=lambda x: x[1])[0],
        'monitoring_active': health['monitoring_active'],
        'alert_callbacks': health['alert_callbacks_count'],
        'configuration_completeness': calculate_config_completeness(health['config'])
    }

def calculate_config_completeness(config: dict) -> float:
    """计算配置完整性"""
    required_keys = ['monitoring_enabled', 'alerts_enabled', 'log_format', 'retry_enabled']
    present_keys = sum(1 for key in required_keys if key in config)
    return present_keys / len(required_keys)
```

### 2. 持续改进

#### 定期审查异常处理
```python
def review_exception_handling():
    """定期审查异常处理效果"""
    metrics = get_exception_metrics()

    issues = []

    # 检查监控覆盖率
    if not metrics['monitoring_active']:
        issues.append("异常监控未启用")

    # 检查告警配置
    if metrics['alert_callbacks'] == 0:
        issues.append("未配置告警回调")

    # 检查异常多样性
    if metrics['error_types_count'] > 20:
        issues.append(f"异常类型过多 ({metrics['error_types_count']})，可能需要重构")

    # 检查配置完整性
    if metrics['configuration_completeness'] < 0.8:
        issues.append(f"配置不完整 ({metrics['configuration_completeness']:.1%})")

    return {
        'metrics': metrics,
        'issues': issues,
        'recommendations': generate_improvement_recommendations(issues)
    }

def generate_improvement_recommendations(issues):
    """生成改进建议"""
    recommendations = []

    for issue in issues:
        if "监控未启用" in issue:
            recommendations.append("启用异常监控以提高系统可观测性")
        elif "未配置告警" in issue:
            recommendations.append("配置告警回调以便及时响应异常")
        elif "异常类型过多" in issue:
            recommendations.append("重构异常层次结构，合并相似异常类型")
        elif "配置不完整" in issue:
            recommendations.append("完善异常处理配置，确保所有必需设置都已配置")

    return recommendations
```

## 🎯 总结

遵循这些最佳实践可以：

1. **提高代码质量**: 统一的异常处理模式使代码更易理解和维护
2. **改善错误诊断**: 丰富的上下文信息帮助快速定位问题
3. **增强系统稳定性**: 适当的重试和恢复机制提高系统容错能力
4. **优化运维效率**: 自动监控和告警减少人工干预需求
5. **支持持续改进**: 完善的度量体系指导系统优化

记住，好的异常处理不是事后补救，而是在设计时就要考虑的系统特性。

---

*最后更新: 2025年10月3日*

