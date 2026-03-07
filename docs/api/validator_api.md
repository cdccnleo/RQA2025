# 配置验证器API文档

## 概述

配置验证器模块提供全面的配置验证功能，支持多种验证规则和策略。模块采用模块化设计，包含基础组件、专用验证器和组合工厂。

## 架构概览

```
validator_base.py          # 基础组件：枚举、接口、基类
├── ValidationSeverity     # 验证严重程度枚举
├── ValidationType        # 验证类型枚举
├── ValidationResult      # 验证结果类
├── ValidationRule        # 验证规则类
├── IConfigValidator      # 验证器接口
└── BaseConfigValidator   # 验证器基类

specialized_validators.py # 专用验证器实现
├── TradingHoursValidator # 交易时间验证器
├── DatabaseConfigValidator # 数据库配置验证器
├── LoggingConfigValidator  # 日志配置验证器
└── NetworkConfigValidator  # 网络配置验证器

validator_composition.py  # 组合和工厂
├── ConfigValidators      # 验证器组合类
├── UnifiedValidatorFactory # 验证器工厂
└── ConfigValidator       # 通用配置验证器
```

## 基础组件API

### ValidationSeverity 枚举

```python
from infrastructure.config.validators.validator_base import ValidationSeverity

# 验证严重程度等级
ValidationSeverity.INFO     # 信息级
ValidationSeverity.WARNING  # 警告级
ValidationSeverity.ERROR    # 错误级
ValidationSeverity.CRITICAL # 严重级

# 使用示例
severity = ValidationSeverity.ERROR
print(severity.value)  # 输出: "error"

# 支持比较运算
assert ValidationSeverity.ERROR > ValidationSeverity.WARNING
```

### ValidationType 枚举

```python
from infrastructure.config.validators.validator_base import ValidationType

# 验证类型
ValidationType.REQUIRED   # 必需字段验证
ValidationType.TYPE      # 类型验证
ValidationType.RANGE     # 范围验证
ValidationType.PATTERN   # 模式验证
ValidationType.CUSTOM    # 自定义验证
ValidationType.DEPENDENCY # 依赖验证
```

### ValidationResult 类

```python
from infrastructure.config.validators.validator_base import ValidationResult

# 创建验证结果
result = ValidationResult(
    success=True,
    message="验证通过",
    severity=ValidationSeverity.INFO,
    field="database.host",
    value="localhost"
)

# 检查验证状态
if result.success:
    print("验证通过")
else:
    print(f"验证失败: {result.message}")

# 转换为字典
result_dict = result.to_dict()
```

### ValidationRule 类

```python
from infrastructure.config.validators.validator_base import ValidationRule, ValidationType

# 创建验证规则
rule = ValidationRule(
    rule_type=ValidationType.REQUIRED,
    field="database.host",
    required=True
)

# 执行验证
result = rule.validate("localhost")
```

## 验证器接口和基类

### IConfigValidator 接口

```python
from infrastructure.config.validators.validator_base import IConfigValidator

class IConfigValidator(ABC):
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """验证配置"""
        pass

    @abstractmethod
    def validate_field(self, field: str, value: Any) -> ValidationResult:
        """验证单个字段"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """验证器名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """验证器描述"""
        pass
```

### BaseConfigValidator 基类

```python
from infrastructure.config.validators.validator_base import BaseConfigValidator

class MyValidator(BaseConfigValidator):
    def __init__(self):
        super().__init__(
            name="MyValidator",
            description="我的自定义验证器"
        )

        # 添加验证规则
        self.add_rule(ValidationRule(...))

    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """自定义验证逻辑"""
        results = []
        # 实现自定义验证
        return results
```

## 专用验证器

### TradingHoursValidator

```python
from infrastructure.config.validators.specialized_validators import TradingHoursValidator

validator = TradingHoursValidator()

config = {
    "trading_hours": {
        "start": "09:30",
        "end": "16:00",
        "timezone": "America/New_York"
    }
}

results = validator.validate(config)
```

**支持验证规则：**
- `trading_hours.start`: 开始时间格式验证 (HH:MM)
- `trading_hours.end`: 结束时间格式验证 (HH:MM)
- `trading_hours.timezone`: 时区名称验证
- 时间范围逻辑验证 (结束时间晚于开始时间)

### DatabaseConfigValidator

```python
from infrastructure.config.validators.specialized_validators import DatabaseConfigValidator

validator = DatabaseConfigValidator()

config = {
    "database": {
        "host": "localhost",
        "port": 3306,
        "name": "mydb",
        "username": "admin",
        "pool": {
            "min_size": 5,
            "max_size": 20
        }
    }
}

results = validator.validate(config)
```

**支持验证规则：**
- `database.host`: 主机地址格式验证
- `database.port`: 端口范围验证 (1-65535)
- `database.name`: 数据库名称验证
- `database.username`: 用户名验证
- `database.type`: 数据库类型验证 (mysql, postgresql, oracle, sqlite, mongodb)
- 连接池配置验证

### LoggingConfigValidator

```python
from infrastructure.config.validators.specialized_validators import LoggingConfigValidator

validator = LoggingConfigValidator()

config = {
    "logging": {
        "level": "INFO",
        "file": "/var/log/app.log",
        "format": "%(asctime)s - %(levelname)s - %(message)s"
    }
}

results = validator.validate(config)
```

**支持验证规则：**
- `logging.level`: 日志级别验证 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: 日志文件路径验证
- `logging.format`: 日志格式验证
- `logging.rotation`: 日志轮转配置验证

### NetworkConfigValidator

```python
from infrastructure.config.validators.specialized_validators import NetworkConfigValidator

validator = NetworkConfigValidator()

config = {
    "network": {
        "host": "0.0.0.0",
        "port": 8080,
        "ssl": {
            "cert_file": "/path/to/cert.pem",
            "key_file": "/path/to/key.pem"
        }
    }
}

results = validator.validate(config)
```

**支持验证规则：**
- `network.host`: 网络主机地址验证
- `network.port`: 端口范围验证 (1-65535)
- SSL证书文件路径验证
- 代理配置验证

## 组合验证器

### ConfigValidators 类

```python
from infrastructure.config.validators.validator_composition import ConfigValidators

# 创建组合验证器
composite_validator = ConfigValidators("MyCompositeValidator")

# 添加多个验证器
composite_validator.add_validator(DatabaseConfigValidator())
composite_validator.add_validator(LoggingConfigValidator())

# 执行验证
config = {
    "database": {...},
    "logging": {...}
}

results = composite_validator.validate(config)
```

**方法：**
- `add_validator(validator)`: 添加验证器
- `remove_validator(validator)`: 移除验证器
- `clear_validators()`: 清空验证器
- `set_fail_fast(fail_fast)`: 设置快速失败模式

### UnifiedValidatorFactory 类

```python
from infrastructure.config.validators.validator_composition import UnifiedValidatorFactory

# 创建工厂
factory = UnifiedValidatorFactory()

# 注册自定义验证器类型
factory.register_validator_type('custom', MyCustomValidator)

# 创建验证器实例
validator = factory.create_validator('database')

# 创建组合验证器
specs = [
    {'type': 'database'},
    {'type': 'logging'},
    {'type': 'network'}
]
composite = factory.create_composite_validator(specs)
```

**方法：**
- `register_validator_type(type_name, validator_class)`: 注册验证器类型
- `create_validator(type_name, *args, **kwargs)`: 创建验证器实例
- `create_composite_validator(specs)`: 创建组合验证器
- `get_available_types()`: 获取可用验证器类型

### ConfigValidator 类

```python
from infrastructure.config.validators.validator_composition import ConfigValidator

# 创建通用验证器
validator = ConfigValidator(rules=[
    {
        'type': 'REQUIRED',
        'field': 'app.name',
        'required': True
    },
    {
        'type': 'TYPE',
        'field': 'app.port',
        'params': {'type': int}
    }
])

# 添加自定义验证器
def validate_port_range(value):
    if not isinstance(value, int) or not (1024 <= value <= 65535):
        return ValidationResult(False, "端口范围无效")
    return ValidationResult(True, "端口验证通过")

validator.add_custom_validator('app.port', validate_port_range)

# 执行验证
config = {'app': {'name': 'MyApp', 'port': 8080}}
results = validator.validate(config)
```

## 便捷函数

### create_validator_factory()

```python
from infrastructure.config.validators.validators import create_validator_factory

factory = create_validator_factory()
```

### create_composite_validator()

```python
from infrastructure.config.validators.validators import create_composite_validator

specs = [
    {'type': 'database'},
    {'type': 'logging'}
]
validator = create_composite_validator(specs)
```

### validate_config()

```python
from infrastructure.config.validators.validators import validate_config

config = {
    'database': {...},
    'logging': {...}
}

# 使用默认验证器集合
results = validate_config(config)

# 使用自定义验证器集合
results = validate_config(config, validator_specs=[
    {'type': 'database'},
    {'type': 'network'}
])
```

## 验证结果处理

### 批量验证结果分析

```python
from infrastructure.config.validators.validators import validate_config

config = {...}
results = validate_config(config)

# 统计验证结果
total = len(results)
passed = sum(1 for r in results if r.success)
failed = total - passed

print(f"验证完成: {passed}/{total} 通过")

# 获取失败的验证
failed_results = [r for r in results if not r.success]
for result in failed_results:
    print(f"❌ {result.field}: {result.message}")
    if result.suggestions:
        print(f"   💡 建议: {', '.join(result.suggestions)}")
```

### 验证结果格式化

```python
# 转换为字典格式
result_dicts = [result.to_dict() for result in results]

# 按字段分组
from collections import defaultdict
field_results = defaultdict(list)
for result in results:
    field_results[result.field].append(result)

# 按严重程度分组
severity_results = defaultdict(list)
for result in results:
    severity_results[result.severity.value].append(result)
```

## 最佳实践

### 1. 验证器组合使用

```python
from infrastructure.config.validators.validators import create_composite_validator

# 为不同环境创建不同的验证器组合
def create_production_validator():
    return create_composite_validator([
        {'type': 'database'},
        {'type': 'logging'},
        {'type': 'network'}
    ])

def create_development_validator():
    return create_composite_validator([
        {'type': 'database'}
    ])
```

### 2. 自定义验证规则

```python
from infrastructure.config.validators.validator_base import ValidationRule, ValidationType

# 创建自定义验证规则
custom_rules = [
    ValidationRule(
        ValidationType.PATTERN,
        'email',
        required=True,
        params={'pattern': r'^[^@]+@[^@]+\.[^@]+$'}
    ),
    ValidationRule(
        ValidationType.RANGE,
        'age',
        required=False,
        params={'min': 0, 'max': 150}
    )
]
```

### 3. 错误处理和日志

```python
from infrastructure.config.core.common_exception_handler import handle_config_exceptions

class SafeConfigValidator(BaseConfigValidator):
    @handle_config_exceptions(default_return=[])
    def validate(self, config):
        return super().validate(config)
```

## 异常处理

验证器模块使用统一的异常处理机制：

```python
from infrastructure.config.core.common_exception_handler import (
    handle_config_exceptions,
    ConfigValidatorException
)

@handle_config_exceptions(default_return=ValidationResult(False, "验证异常"))
def safe_validate_field(self, field, value):
    return self.validate_field(field, value)
```

## 向后兼容性

验证器模块保持向后兼容性：

```python
# 旧的导入方式仍然有效
from infrastructure.config.validators import (
    ValidationSeverity,  # 兼容别名
    ConfigValidationResult  # ValidationResult的别名
)

# 工厂方法
from infrastructure.config.validators import ValidatorFactory  # UnifiedValidatorFactory的别名
```

## 性能考虑

- 验证器实例可以复用，避免重复创建
- 组合验证器支持快速失败模式，提高性能
- 大型配置建议分批验证，避免内存压力
- 自定义验证函数应避免阻塞操作

---

**版本**: v1.0
**更新日期**: 2025-09-23
**兼容性**: Python 3.8+
