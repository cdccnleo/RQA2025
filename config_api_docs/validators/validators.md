# validators

**文件路径**: `validators\validators.py`

## 模块描述

统一配置验证器 (优化版)

整合所有验证器功能，提供统一的配置验证框架
合并了validators.py, validator_factory.py, validator_components.py, unified_validator.py的功能

支持:
- 多策略验证组合
- 工厂模式管理
- 组件化架构
- 标准化的验证结果
- 丰富的验证规则

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import List
from infrastructure.config.core.imports import Optional
from infrastructure.config.core.imports import Tuple
from infrastructure.config.core.imports import Callable
from infrastructure.config.core.imports import Union
from infrastructure.config.core.imports import datetime
from infrastructure.config.core.imports import ABC
from infrastructure.config.core.imports import abstractmethod
# ... 等5个导入
```

## 类

### ValidationSeverity

验证严重程度

**继承**: Enum

**方法**:

- `__lt__`

### ValidationType

验证类型

**继承**: Enum

### ValidationResult

验证结果

统一验证结果格式

**方法**:

- `__init__`
- `add_error`
- `add_warning`
- `to_dict`
- `merge`

### IConfigValidator

配置验证器接口

**继承**: ABC

**方法**:

- `validate`
- `name`
- `description`

### BaseConfigValidator

基础配置验证器

**继承**: IConfigValidator

**方法**:

- `__init__`
- `name`
- `description`

### TradingHoursValidator

交易时段验证器

**继承**: BaseConfigValidator

**方法**:

- `__init__`
- `validate`

### DatabaseConfigValidator

数据库配置验证器

**继承**: BaseConfigValidator

**方法**:

- `__init__`
- `validate`
- `validate_database_config`

### LoggingConfigValidator

日志配置验证器

**继承**: BaseConfigValidator

**方法**:

- `__init__`
- `validate`

### NetworkConfigValidator

网络配置验证器

**继承**: BaseConfigValidator

**方法**:

- `__init__`
- `validate`
- `_is_valid_ip`

### ConfigValidators

多策略组合校验器 (兼容原有接口)

**方法**:

- `__init__`
- `validate`

### UnifiedValidatorFactory

统一验证器工厂

**方法**:

- `__init__`
- `_register_default_validators`
- `register_validator`
- `create_validator`
- `get_available_validators`
- ... 等1个方法

### ConfigValidator

配置验证器（向后兼容简单接口）

**继承**: BaseConfigValidator

**方法**:

- `__init__`
- `validate`
- `add_validation_rule`
- `_get_nested_value`

## 函数

### validate_trading_hours

验证交易时段配置 (兼容原有接口)

**参数**:

- `config: <ast.Subscript object at 0x000002259027D9A0>`

**返回值**: `bool`

### validate_database_config

验证数据库配置 (兼容原有接口)

**参数**:

- `config: <ast.Subscript object at 0x000002259027D5E0>`

**返回值**: `bool`

### validate_logging_config

验证日志配置 (兼容原有接口)

**参数**:

- `config: <ast.Subscript object at 0x000002259024AB20>`

**返回值**: `bool`

### validate_network_config

验证网络配置 (兼容原有接口)

**参数**:

- `config: <ast.Subscript object at 0x000002259024A160>`

**返回值**: `bool`

### get_validator_factory

获取全局验证器工厂实例

**返回值**: `UnifiedValidatorFactory`

### reset_validator_factory

重置全局验证器工厂实例

### create_validator

创建验证器 (便捷函数)

**参数**:

- `name: str`
- `**kwargs`

**返回值**: `IConfigValidator`

### create_validator_suite

创建验证器套件 (便捷函数)

**参数**:

- `validator_names: <ast.Subscript object at 0x00000225902BF370>`

**返回值**: `ConfigValidators`

### validate_config_with_suite

使用验证器套件验证配置 (便捷函数)

**参数**:

- `config: <ast.Subscript object at 0x000002259026ACD0>`
- `validator_names: <ast.Subscript object at 0x000002259026A430>`

**返回值**: `<ast.Subscript object at 0x000002259026AFA0>`

