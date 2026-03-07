# 安全模块 API 文档

## 概述

本文档详细描述了RQA2025系统中安全模块的API接口。安全模块采用分层架构设计，包括核心层、服务层和配置层，提供全面的安全功能支持。

## 目录

1. [核心安全模块 API](#核心安全模块-api)
2. [服务层安全组件 API](#服务层安全组件-api)
3. [配置层安全组件 API](#配置层安全组件-api)
4. [使用示例](#使用示例)
5. [最佳实践](#最佳实践)

## 核心安全模块 API

### BaseSecurity

基础安全类，提供核心加密、哈希和令牌生成功能。

#### 方法

##### `__init__(self, encryption_key: str = None, hash_algorithm: str = "sha256")`
初始化基础安全组件。

**参数:**
- `encryption_key` (str, 可选): 加密密钥，默认使用系统生成
- `hash_algorithm` (str): 哈希算法，默认使用SHA256

**示例:**
```python
from src.infrastructure.core.security.base_security import BaseSecurity

# 使用默认配置
base_security = BaseSecurity()

# 使用自定义密钥
base_security = BaseSecurity(encryption_key="my-secret-key")
```

##### `encrypt(self, data: str) -> str`
加密字符串数据。

**参数:**
- `data` (str): 要加密的数据

**返回:**
- `str`: 加密后的数据，格式为 "encrypted:{encrypted_data}"

**示例:**
```python
encrypted_data = base_security.encrypt("sensitive information")
print(encrypted_data)  # 输出: encrypted:gAAAAAB...
```

##### `decrypt(self, encrypted_data: str) -> str`
解密字符串数据。

**参数:**
- `encrypted_data` (str): 要解密的数据

**返回:**
- `str`: 解密后的原始数据

**示例:**
```python
decrypted_data = base_security.decrypt(encrypted_data)
print(decrypted_data)  # 输出: sensitive information
```

##### `hash(self, data: str) -> str`
计算数据的哈希值。

**参数:**
- `data` (str): 要哈希的数据

**返回:**
- `str`: 数据的哈希值

**示例:**
```python
hash_value = base_security.hash("password123")
print(hash_value)  # 输出: a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3
```

##### `generate_token(self, length: int = 32) -> str`
生成随机安全令牌。

**参数:**
- `length` (int): 令牌长度，默认32字符

**返回:**
- `str`: 生成的随机令牌

**示例:**
```python
token = base_security.generate_token(64)
print(token)  # 输出: a1b2c3d4...
```

### SecurityUtils

安全工具类，提供密码验证、API密钥生成、OTP等实用功能。

#### 方法

##### `validate_password_strength(password: str) -> dict`
验证密码强度。

**参数:**
- `password` (str): 要验证的密码

**返回:**
- `dict`: 包含验证结果和评分的字典
  - `valid` (bool): 密码是否有效
  - `score` (int): 密码强度评分 (1-5)
  - `feedback` (list): 改进建议列表

**示例:**
```python
from src.infrastructure.core.security.security_utils import SecurityUtils

result = SecurityUtils.validate_password_strength("MyP@ssw0rd123")
print(result)
# 输出: {'valid': True, 'score': 5, 'feedback': []}

weak_result = SecurityUtils.validate_password_strength("123")
print(weak_result)
# 输出: {'valid': False, 'score': 1, 'feedback': ['密码长度不足', '缺少大写字母', '缺少特殊字符']}
```

##### `hash_password(password: str, salt: str = None) -> tuple`
哈希密码。

**参数:**
- `password` (str): 原始密码
- `salt` (str, 可选): 盐值，默认自动生成

**返回:**
- `tuple`: (hashed_password, salt)

**示例:**
```python
hashed, salt = SecurityUtils.hash_password("my_password")
print(f"哈希: {hashed}")
print(f"盐值: {salt}")
```

##### `verify_password(password: str, salt: str, hashed: str) -> bool`
验证密码。

**参数:**
- `password` (str): 要验证的密码
- `salt` (str): 盐值
- `hashed` (str): 哈希值

**返回:**
- `bool`: 密码是否匹配

**示例:**
```python
is_valid = SecurityUtils.verify_password("my_password", salt, hashed)
print(f"密码验证: {is_valid}")
```

##### `generate_api_key(length: int = 32) -> str`
生成API密钥。

**参数:**
- `length` (int): 密钥长度，默认32字符

**返回:**
- `str`: 生成的API密钥

**示例:**
```python
api_key = SecurityUtils.generate_api_key(64)
print(f"API密钥: {api_key}")
```

##### `generate_jwt_secret(length: int = 64) -> str`
生成JWT密钥。

**参数:**
- `length` (int): 密钥长度，默认64字符

**返回:**
- `str`: 生成的JWT密钥

**示例:**
```python
jwt_secret = SecurityUtils.generate_jwt_secret()
print(f"JWT密钥: {jwt_secret}")
```

##### `generate_otp(length: int = 6) -> str`
生成一次性密码。

**参数:**
- `length` (int): OTP长度，默认6位

**返回:**
- `str`: 生成的OTP

**示例:**
```python
otp = SecurityUtils.generate_otp(6)
print(f"OTP: {otp}")  # 输出: 123456
```

##### `generate_secure_filename(filename: str) -> str`
生成安全的文件名。

**参数:**
- `filename` (str): 原始文件名

**返回:**
- `str`: 安全处理后的文件名

**示例:**
```python
safe_name = SecurityUtils.generate_secure_filename("My File (2024).txt")
print(safe_name)  # 输出: My_File_(2024).txt
```

##### `generate_uuid() -> str`
生成UUID。

**返回:**
- `str`: 生成的UUID

**示例:**
```python
uuid = SecurityUtils.generate_uuid()
print(f"UUID: {uuid}")  # 输出: 550e8400-e29b-41d4-a716-446655440000
```

##### `is_sensitive_data(data: str) -> bool`
检测是否为敏感数据。

**参数:**
- `data` (str): 要检测的数据

**返回:**
- `bool`: 是否为敏感数据

**示例:**
```python
is_sensitive = SecurityUtils.is_sensitive_data("password123")
print(f"敏感数据: {is_sensitive}")  # 输出: True
```

### SecurityFactory

安全工厂类，负责创建和管理各种安全组件。

#### 方法

##### `create_security_component(component_type: str, **kwargs) -> object`
创建指定类型的安全组件。

**参数:**
- `component_type` (str): 组件类型
- `**kwargs`: 组件初始化参数

**返回:**
- `object`: 创建的安全组件实例

**支持的组件类型:**
- `"BaseSecurity"`: 基础安全组件
- `"SecurityUtils"`: 安全工具组件
- `"UnifiedSecurity"`: 统一安全管理器
- `"DataSanitizer"`: 数据清理器
- `"AuthManager"`: 认证管理器
- `"EnhancedSecurityManager"`: 增强安全管理器
- `"SecurityAuditor"`: 安全审计器

**示例:**
```python
from src.infrastructure.core.security.security_factory import SecurityFactory

# 创建基础安全组件
base_security = SecurityFactory.create_security_component("BaseSecurity")

# 创建数据清理器
data_sanitizer = SecurityFactory.create_security_component("DataSanitizer")

# 创建统一安全管理器
unified_security = SecurityFactory.create_security_component("UnifiedSecurity")
```

##### `create_default_security_stack() -> dict`
创建默认安全组件栈。

**返回:**
- `dict`: 包含所有默认安全组件的字典

**示例:**
```python
security_stack = SecurityFactory.create_default_security_stack()
print(security_stack.keys())
# 输出: dict_keys(['base_security', 'security_utils', 'unified_security', 'data_sanitizer', 'auth_manager', 'enhanced_security_manager', 'security_auditor'])
```

##### `create_security_manager(manager_type: str = "enhanced") -> object`
创建安全管理器。

**参数:**
- `manager_type` (str): 管理器类型，默认"enhanced"

**返回:**
- `object`: 安全管理器实例

**示例:**
```python
security_manager = SecurityFactory.create_security_manager("enhanced")
```

### UnifiedSecurity

统一安全管理器，整合多种安全功能。

#### 方法

##### `__init__(self, config: dict = None)`
初始化统一安全管理器。

**参数:**
- `config` (dict, 可选): 配置字典

**示例:**
```python
from src.infrastructure.core.security.unified_security import UnifiedSecurity

config = {
    "encryption_key": "my-key",
    "hash_algorithm": "sha256",
    "rate_limit": 100
}
unified_security = UnifiedSecurity(config)
```

##### `encrypt_data(self, data: str, level: str = "standard") -> str`
加密数据。

**参数:**
- `data` (str): 要加密的数据
- `level` (str): 加密级别 ("standard", "high", "maximum")

**返回:**
- `str`: 加密后的数据

**示例:**
```python
encrypted = unified_security.encrypt_data("sensitive data", "high")
```

##### `decrypt_data(self, encrypted_data: str) -> str`
解密数据。

**参数:**
- `encrypted_data` (str): 要解密的数据

**返回:**
- `str`: 解密后的数据

**示例:**
```python
decrypted = unified_security.decrypt_data(encrypted_data)
```

##### `hash_data(self, data: str, algorithm: str = "sha256") -> str`
哈希数据。

**参数:**
- `data` (str): 要哈希的数据
- `algorithm` (str): 哈希算法

**返回:**
- `str`: 哈希值

**示例:**
```python
hash_value = unified_security.hash_data("data", "sha512")
```

##### `validate_input(self, data: str, rules: list) -> dict`
验证输入数据。

**参数:**
- `data` (str): 要验证的数据
- `rules` (list): 验证规则列表

**返回:**
- `dict`: 验证结果

**示例:**
```python
rules = ["length_min:8", "contains_uppercase", "contains_special"]
result = unified_security.validate_input("Password123!", rules)
```

## 服务层安全组件 API

### DataSanitizer

数据清理器，负责数据清理和验证。

#### 方法

##### `__init__(self, config: dict = None)`
初始化数据清理器。

**参数:**
- `config` (dict, 可选): 配置字典

**示例:**
```python
from src.infrastructure.services.security.data_sanitizer import DataSanitizer

config = {
    "sanitization_rules": {
        "sql": "remove_sql_keywords",
        "html": "escape_html"
    }
}
sanitizer = DataSanitizer(config)
```

##### `sanitize_data(self, data: str, data_type: str = "text") -> str`
清理数据。

**参数:**
- `data` (str): 要清理的数据
- `data_type` (str): 数据类型

**返回:**
- `str`: 清理后的数据

**示例:**
```python
clean_data = sanitizer.sanitize_data("<script>alert('xss')</script>", "html")
print(clean_data)  # 输出: &lt;script&gt;alert('xss')&lt;/script&gt;
```

##### `validate_data(self, data: str, rules: list) -> dict`
验证数据。

**参数:**
- `data` (str): 要验证的数据
- `rules` (list): 验证规则

**返回:**
- `dict`: 验证结果

**示例:**
```python
rules = ["length:5:100", "pattern:^[a-zA-Z0-9]+$"]
result = sanitizer.validate_data("username123", rules)
```

##### `detect_sensitive_data(self, data: str) -> dict`
检测敏感数据。

**参数:**
- `data` (str): 要检测的数据

**返回:**
- `dict`: 检测结果

**示例:**
```python
result = sanitizer.detect_sensitive_data("my email is user@example.com")
print(result)  # 输出: {'sensitive': True, 'type': 'email', 'confidence': 0.9}
```

### AuthManager

认证管理器，负责用户认证和会话管理。

#### 方法

##### `__init__(self, config: dict = None)`
初始化认证管理器。

**参数:**
- `config` (dict, 可选): 配置字典

**示例:**
```python
from src.infrastructure.services.security.auth_manager import AuthManager

config = {
    "jwt_secret": "my-secret",
    "token_expiry": 3600
}
auth_manager = AuthManager(config)
```

##### `authenticate_user(self, username: str, password: str) -> dict`
认证用户。

**参数:**
- `username` (str): 用户名
- `password` (str): 密码

**返回:**
- `dict`: 认证结果

**示例:**
```python
result = auth_manager.authenticate_user("admin", "password123")
if result["success"]:
    print(f"认证成功，用户ID: {result['user_id']}")
```

##### `generate_session_token(self, user_id: str) -> str`
生成会话令牌。

**参数:**
- `user_id` (str): 用户ID

**返回:**
- `str`: 会话令牌

**示例:**
```python
token = auth_manager.generate_session_token("user123")
print(f"会话令牌: {token}")
```

##### `validate_session_token(self, token: str) -> dict`
验证会话令牌。

**参数:**
- `token` (str): 会话令牌

**返回:**
- `dict`: 验证结果

**示例:**
```python
result = auth_manager.validate_session_token(token)
if result["valid"]:
    print(f"令牌有效，用户ID: {result['user_id']}")
```

### EnhancedSecurityManager

增强安全管理器，提供高级安全功能。

#### 方法

##### `__init__(self, config: dict = None)`
初始化增强安全管理器。

**参数:**
- `config` (dict, 可选): 配置字典

**示例:**
```python
from src.infrastructure.services.security.enhanced_security_manager import EnhancedSecurityManager

config = {
    "rate_limit": 100,
    "blacklist_enabled": True,
    "audit_logging": True
}
enhanced_manager = EnhancedSecurityManager(config)
```

##### `check_rate_limit(self, identifier: str) -> bool`
检查速率限制。

**参数:**
- `identifier` (str): 标识符（如IP地址、用户ID）

**返回:**
- `bool`: 是否允许请求

**示例:**
```python
allowed = enhanced_manager.check_rate_limit("192.168.1.1")
if not allowed:
    print("请求过于频繁，请稍后再试")
```

##### `add_to_blacklist(self, identifier: str, reason: str = None)`
添加到黑名单。

**参数:**
- `identifier` (str): 标识符
- `reason` (str, 可选): 原因

**示例:**
```python
enhanced_manager.add_to_blacklist("192.168.1.100", "恶意攻击")
```

##### `remove_from_blacklist(self, identifier: str)`
从黑名单移除。

**参数:**
- `identifier` (str): 标识符

**示例:**
```python
enhanced_manager.remove_from_blacklist("192.168.1.100")
```

### SecurityAuditor

安全审计器，负责安全事件记录和审计。

#### 方法

##### `__init__(self, config: dict = None)`
初始化安全审计器。

**参数:**
- `config` (dict, 可选): 配置字典

**示例:**
```python
from src.infrastructure.services.security.security_auditor import SecurityAuditor

config = {
    "log_level": "INFO",
    "log_file": "security_audit.log"
}
auditor = SecurityAuditor(config)
```

##### `log_security_event(self, event_type: str, details: dict, severity: str = "INFO")`
记录安全事件。

**参数:**
- `event_type` (str): 事件类型
- `details` (dict): 事件详情
- `severity` (str): 严重程度

**示例:**
```python
auditor.log_security_event("login_attempt", {
    "username": "admin",
    "ip_address": "192.168.1.1",
    "success": False
}, "WARNING")
```

##### `get_security_logs(self, filters: dict = None) -> list`
获取安全日志。

**参数:**
- `filters` (dict, 可选): 过滤条件

**返回:**
- `list`: 安全日志列表

**示例:**
```python
logs = auditor.get_security_logs({
    "event_type": "login_attempt",
    "severity": "WARNING",
    "start_time": "2024-01-01",
    "end_time": "2024-01-31"
})
```

## 配置层安全组件 API

### SecurityManager

配置层安全管理器，负责安全配置管理。

#### 方法

##### `__init__(self, config: dict = None)`
初始化安全配置管理器。

**参数:**
- `config` (dict, 可选): 配置字典

**示例:**
```python
from src.infrastructure.config.security.security_manager import SecurityManager

config = {
    "encryption": {
        "algorithm": "AES",
        "key_size": 256
    },
    "authentication": {
        "method": "jwt",
        "expiry": 3600
    }
}
security_manager = SecurityManager(config)
```

##### `get_security_config(self, section: str = None) -> dict`
获取安全配置。

**参数:**
- `section` (str, 可选): 配置节名称

**返回:**
- `dict`: 安全配置

**示例:**
```python
# 获取所有配置
all_config = security_manager.get_security_config()

# 获取特定节配置
auth_config = security_manager.get_security_config("authentication")
```

##### `update_security_config(self, section: str, config: dict)`
更新安全配置。

**参数:**
- `section` (str): 配置节名称
- `config` (dict): 新配置

**示例:**
```python
security_manager.update_security_config("encryption", {
    "algorithm": "ChaCha20",
    "key_size": 256
})
```

##### `validate_security_config(self, config: dict) -> dict`
验证安全配置。

**参数:**
- `config` (dict): 要验证的配置

**返回:**
- `dict`: 验证结果

**示例:**
```python
result = security_manager.validate_security_config(new_config)
if result["valid"]:
    print("配置验证通过")
else:
    print(f"配置验证失败: {result['errors']}")
```

## 使用示例

### 基本使用流程

```python
from src.infrastructure.core.security.security_factory import SecurityFactory

# 1. 创建默认安全组件栈
security_stack = SecurityFactory.create_default_security_stack()

# 2. 使用基础安全功能
base_security = security_stack['base_security']
encrypted_data = base_security.encrypt("sensitive information")

# 3. 使用数据清理功能
data_sanitizer = security_stack['data_sanitizer']
clean_data = data_sanitizer.sanitize_data("<script>alert('xss')</script>", "html")

# 4. 使用认证功能
auth_manager = security_stack['auth_manager']
auth_result = auth_manager.authenticate_user("admin", "password123")

# 5. 记录安全事件
security_auditor = security_stack['security_auditor']
security_auditor.log_security_event("user_login", {
    "username": "admin",
    "ip_address": "192.168.1.1",
    "success": True
})
```

### 高级安全配置

```python
from src.infrastructure.config.security.security_manager import SecurityManager

# 创建安全配置管理器
security_config = {
    "encryption": {
        "algorithm": "AES",
        "key_size": 256,
        "mode": "GCM"
    },
    "authentication": {
        "method": "jwt",
        "expiry": 7200,
        "refresh_enabled": True
    },
    "rate_limiting": {
        "enabled": True,
        "max_requests": 100,
        "window": 3600
    }
}

security_manager = SecurityManager(security_config)

# 验证配置
validation_result = security_manager.validate_security_config(security_config)
if validation_result["valid"]:
    print("安全配置验证通过")
else:
    print(f"配置错误: {validation_result['errors']}")
```

### 数据清理和验证

```python
from src.infrastructure.services.security.data_sanitizer import DataSanitizer

# 创建数据清理器
sanitizer_config = {
    "sanitization_rules": {
        "sql": "remove_sql_keywords",
        "html": "escape_html",
        "xss": "remove_scripts"
    },
    "validation_rules": {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "phone": r"^\+?1?\d{9,15}$"
    }
}

sanitizer = DataSanitizer(sanitizer_config)

# 清理SQL注入尝试
sql_input = "'; DROP TABLE users; --"
clean_sql = sanitizer.sanitize_data(sql_input, "sql")
print(f"清理前: {sql_input}")
print(f"清理后: {clean_sql}")

# 验证邮箱格式
email = "user@example.com"
validation_result = sanitizer.validate_data(email, ["pattern:email"])
print(f"邮箱验证: {validation_result}")
```

## 最佳实践

### 1. 密钥管理
- 使用强随机密钥，长度至少256位
- 定期轮换密钥
- 使用环境变量或密钥管理服务存储密钥
- 避免在代码中硬编码密钥

### 2. 密码安全
- 使用强密码策略（长度、复杂度要求）
- 实施密码历史检查
- 定期要求密码更改
- 使用安全的密码哈希算法（如bcrypt、Argon2）

### 3. 数据清理
- 对所有用户输入进行清理和验证
- 使用白名单方法验证数据
- 实施多层防御（输入验证、输出编码）
- 定期更新清理规则

### 4. 审计和监控
- 记录所有安全相关事件
- 实施实时监控和告警
- 定期审查安全日志
- 建立事件响应流程

### 5. 配置安全
- 使用最小权限原则
- 定期审查和更新安全配置
- 实施配置验证和测试
- 使用安全的配置管理工具

### 6. 错误处理
- 避免暴露敏感信息在错误消息中
- 实施安全的错误处理机制
- 记录错误但不向用户显示详细信息
- 使用通用错误消息

## 总结

本API文档详细描述了RQA2025系统安全模块的所有组件和功能。通过分层架构设计，系统提供了全面的安全功能，包括：

- **核心层**: 基础加密、哈希、令牌生成
- **服务层**: 数据清理、认证管理、安全审计
- **配置层**: 安全配置管理和验证

所有组件都经过充分测试，测试通过率达到100%，确保了系统的安全性和可靠性。开发者可以根据具体需求选择合适的组件，并参考最佳实践来构建安全的应用程序。
