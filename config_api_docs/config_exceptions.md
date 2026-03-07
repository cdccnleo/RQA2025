# config_exceptions

**文件路径**: `config_exceptions.py`

## 模块描述

配置系统异常定义
定义配置管理过程中可能出现的各种异常

## 导入语句

```python
from infrastructure.config.core.imports import Dict
from infrastructure.config.core.imports import Any
from infrastructure.config.core.imports import Optional
```

## 类

### ConfigError

配置系统基础异常

**继承**: Exception

**方法**:

- `__init__`

### ConfigValidationError

配置验证错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigLoadError

配置加载错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigNotFoundError

配置未找到错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigTypeError

配置类型错误

**继承**: ConfigError

### ConfigValueError

配置值错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigAccessError

配置访问错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigSecurityError

配置安全错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigFormatError

配置格式错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigMergeError

配置合并错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigBackupError

配置备份错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigRestoreError

配置恢复错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigVersionError

配置版本错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigEncryptionError

配置加密错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigDecryptionError

配置解密错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigNetworkError

配置网络错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigTimeoutError

配置超时错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigQuotaExceededError

配置配额超限错误

**继承**: ConfigError

**方法**:

- `__init__`

### ConfigTypeErrorOld

配置类型错误（向后兼容简单接口）

**继承**: ConfigError

**方法**:

- `__init__`

## 函数

### raise_config_error

根据错误类型抛出相应的异常

**参数**:

- `error_type: str`
- `message: str`
- `**kwargs`

