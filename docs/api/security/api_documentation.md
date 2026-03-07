# RQA2025 API Documentation

Generated on: 2025-10-26 12:22:15

## Core Services

### core.base

基础设施层 - 安全管理层 基础实现和接口定义

#### API Endpoints

#### `IAuthManager.authenticate_user`

用户认证

**Signature:** `def authenticate_user(self, username: str, password: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `password: str` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `IAuthManager.create_session`

创建会话

**Signature:** `def create_session(self, user_id: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `IAuthManager.validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `initialize`

初始化组件

Args:
    config: 组件配置

Returns:
    初始化是否成功

**Signature:** `def initialize(self, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

Returns:
    组件状态信息

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭组件

**Signature:** `def shutdown(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `authenticate_user`

用户认证

**Signature:** `def authenticate_user(self, username: str, password: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `password: str` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `create_session`

创建会话

**Signature:** `def create_session(self, user_id: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `encrypt`

加密数据

**Signature:** `def encrypt(self, data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `decrypt`

解密数据

**Signature:** `def decrypt(self, encrypted_data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `log_event`

记录审计事件

**Signature:** `def log_event(self, event_type: str, details: Dict[str, Any]) -> None:`

**Parameters:**

- `self: Any` (required)
- `event_type: str` (required)
- `details: Dict[Any]` (required)

**Returns:** `Any`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_logs`

获取审计日志

**Signature:** `def get_logs(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `start_time: str` (required)
- `end_time: str` (required)

**Returns:** `List[Dict[Any]]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `initialize`

初始化组件

Args:
    config: 组件配置

Returns:
    初始化是否成功

**Signature:** `def initialize(self, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

Returns:
    组件状态信息

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭组件

**Signature:** `def shutdown(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

---

### core.security

安全管理器模块
提供统一的安全管理功能

#### API Endpoints

#### `SecurityManager.add_filter`

添加过滤器

**Signature:** `def add_filter(self, filter_func) -> None:`

**Parameters:**

- `self: Any` (required)
- `filter_func: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `SecurityManager.apply_filters`

应用所有过滤器

**Signature:** `def apply_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `SecurityManager.log_security_event`

记录安全事件

**Signature:** `def log_security_event(self, event: str, details: Dict[str, Any]) -> None:`

**Parameters:**

- `self: Any` (required)
- `event: str` (required)
- `details: Dict[Any]` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `SecurityManager.get_security_status`

获取安全状态

**Signature:** `def get_security_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `add_filter`

添加过滤器

**Signature:** `def add_filter(self, filter_func) -> None:`

**Parameters:**

- `self: Any` (required)
- `filter_func: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `apply_filters`

应用所有过滤器

**Signature:** `def apply_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `log_security_event`

记录安全事件

**Signature:** `def log_security_event(self, event: str, details: Dict[str, Any]) -> None:`

**Parameters:**

- `self: Any` (required)
- `event: str` (required)
- `details: Dict[Any]` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_security_status`

获取安全状态

**Signature:** `def get_security_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

---

### core.security_factory

基础设施层 - 安全管理组件

security_factory 模块

安全管理相关的文件
提供安全管理相关的功能实现。

#### API Endpoints

#### `create_security_manager`

便捷函数：创建安全管理器

Args:
    manager_type: 管理器类型
    config: 配置参数
    **kwargs: 其他参数

Returns:
    安全管理器实例

**Signature:** `def create_security_manager(manager_type: str = 'enhanced',   config: Optional[Dict[str, Any]] = None, **kwargs):`

**Parameters:**

- `manager_type: str` (default: 'enhanced')
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_security_factory_info`

获取安全工厂信息

Returns:
    工厂信息字典

**Signature:** `def get_security_factory_info() -> Dict[str, Any]:`

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_security_component`

创建安全组件

Args:
    component_type: 组件类型
    config: 配置参数
    **kwargs: 其他参数

Returns:
    安全组件实例

Raises:
    ValueError: 不支持的组件类型

**Signature:** `def create_security_component(cls,   component_type: str, config: Optional[Dict[str, Any]] = None, **kwargs):`

**Parameters:**

- `cls: Any` (required)
- `component_type: str` (required)
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `None`

**Decorators:** `classmethod`

**Async:** No | **Visibility:** public

#### `create_default_security_stack`

创建默认安全组件栈

Args:
    config: 配置参数

Returns:
    包含所有安全组件的字典

**Signature:** `def create_default_security_stack(cls,   config: Optional[Dict[str, Any]] = None):`

**Parameters:**

- `cls: Any` (required)
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `None`

**Decorators:** `classmethod`

**Async:** No | **Visibility:** public

#### `get_component_info`

获取所有支持组件的详细信息

Returns:
    组件信息字典

**Signature:** `def get_component_info(cls) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `cls: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `classmethod`

**Async:** No | **Visibility:** public

#### `validate_security_config`

验证安全配置

Args:
    config: 配置字典

Returns:
    验证结果字典

**Signature:** `def validate_security_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `cls: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Decorators:** `classmethod`

**Async:** No | **Visibility:** public

---

### core.types

RQA2025 安全模块参数对象

提供参数对象模式，解决长参数列表问题
提高代码可读性和维护性

#### API Endpoints

#### `is_expired`

检查会话是否过期

**Signature:** `def is_expired(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

---

## Security

### components.base_security_component

基础安全模块
提供统一的安全接口和基础实现

#### API Endpoints

#### `encrypt`

Encrypt data

**Signature:** `def encrypt(self, data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `decrypt`

Decrypt data

**Signature:** `def decrypt(self, encrypted_data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `hash`

Hash data

**Signature:** `def hash(self, data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `verify_hash`

Verify hash

**Signature:** `def verify_hash(self, data: str, hash_value: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `hash_value: str` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `generate_token`

Generate token

**Signature:** `def generate_token(self, data: Dict[str, Any], expires_in: int = 3600) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)
- `expires_in: int` (default: 3600)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `verify_token`

Verify token

**Signature:** `def verify_token(self, token: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `sanitize_input`

Sanitize input

**Signature:** `def sanitize_input(self, input_data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `input_data: str` (required)

**Returns:** `str`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `encrypt`

Encrypt data

**Signature:** `def encrypt(self, data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `decrypt`

Decrypt data

**Signature:** `def decrypt(self, encrypted_data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `hash`

Hash data

**Signature:** `def hash(self, data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `verify_hash`

Verify hash

**Signature:** `def verify_hash(self, data: str, hash_value: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `hash_value: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `generate_token`

Generate token

**Signature:** `def generate_token(self, data: Dict[str, Any], expires_in: int = 3600) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)
- `expires_in: int` (default: 3600)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `verify_token`

Verify token

**Signature:** `def verify_token(self, token: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `sanitize_input`

Sanitize input

**Signature:** `def sanitize_input(self, input_data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `input_data: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `set_security_level`

设置安全级别

**Signature:** `def set_security_level(self, level: SecurityLevel) -> None:`

**Parameters:**

- `self: Any` (required)
- `level: SecurityLevel` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_security_level`

获取安全级别

**Signature:** `def get_security_level(self) -> SecurityLevel:`

**Parameters:**

- `self: Any` (required)

**Returns:** `SecurityLevel`

**Async:** No | **Visibility:** public

#### `validate_password`

验证密码强度

**Signature:** `def validate_password(self, password: str) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `password: str` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `generate_secure_password`

生成安全密码

**Signature:** `def generate_secure_password(self, length: int = 12) -> str:`

**Parameters:**

- `self: Any` (required)
- `length: int` (default: 12)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `encrypt`

高级加密

**Signature:** `def encrypt(self, data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `decrypt`

高级解密

**Signature:** `def decrypt(self, encrypted_data: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `check_rate_limit`

检查速率限制

**Signature:** `def check_rate_limit(self, identifier: str, max_attempts: int = 5, window: int = 300) -> bool:`

**Parameters:**

- `self: Any` (required)
- `identifier: str` (required)
- `max_attempts: int` (default: 5)
- `window: int` (default: 300)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `add_to_blacklist`

添加到黑名单

**Signature:** `def add_to_blacklist(self, identifier: str) -> None:`

**Parameters:**

- `self: Any` (required)
- `identifier: str` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `is_blacklisted`

检查是否在黑名单中

**Signature:** `def is_blacklisted(self, identifier: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `identifier: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `remove_from_blacklist`

从黑名单移除

**Signature:** `def remove_from_blacklist(self, identifier: str) -> None:`

**Parameters:**

- `self: Any` (required)
- `identifier: str` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

---

### components.security_component

#### API Endpoints

#### `create_security_security_component_1`

**Signature:** `def create_security_security_component_1(): return SecurityComponentFactory.create_component(1)   def create_security_security_component_7(): return SecurityComponentFactory.create_component(7)   def create_security_security_component_13(): return SecurityComponentFactory.create_component(13)   def create_security_security_component_19(): return SecurityComponentFactory.create_component(19)   def create_security_security_component_25(): return SecurityComponentFactory.create_component(25)   def create_security_security_component_31(): return SecurityComponentFactory.create_component(31)   def create_security_security_component_37(): return SecurityComponentFactory.create_component(37)   def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_7`

**Signature:** `def create_security_security_component_7(): return SecurityComponentFactory.create_component(7)   def create_security_security_component_13(): return SecurityComponentFactory.create_component(13)   def create_security_security_component_19(): return SecurityComponentFactory.create_component(19)   def create_security_security_component_25(): return SecurityComponentFactory.create_component(25)   def create_security_security_component_31(): return SecurityComponentFactory.create_component(31)   def create_security_security_component_37(): return SecurityComponentFactory.create_component(37)   def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_13`

**Signature:** `def create_security_security_component_13(): return SecurityComponentFactory.create_component(13)   def create_security_security_component_19(): return SecurityComponentFactory.create_component(19)   def create_security_security_component_25(): return SecurityComponentFactory.create_component(25)   def create_security_security_component_31(): return SecurityComponentFactory.create_component(31)   def create_security_security_component_37(): return SecurityComponentFactory.create_component(37)   def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_19`

**Signature:** `def create_security_security_component_19(): return SecurityComponentFactory.create_component(19)   def create_security_security_component_25(): return SecurityComponentFactory.create_component(25)   def create_security_security_component_31(): return SecurityComponentFactory.create_component(31)   def create_security_security_component_37(): return SecurityComponentFactory.create_component(37)   def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_25`

**Signature:** `def create_security_security_component_25(): return SecurityComponentFactory.create_component(25)   def create_security_security_component_31(): return SecurityComponentFactory.create_component(31)   def create_security_security_component_37(): return SecurityComponentFactory.create_component(37)   def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_31`

**Signature:** `def create_security_security_component_31(): return SecurityComponentFactory.create_component(31)   def create_security_security_component_37(): return SecurityComponentFactory.create_component(37)   def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_37`

**Signature:** `def create_security_security_component_37(): return SecurityComponentFactory.create_component(37)   def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_43`

**Signature:** `def create_security_security_component_43(): return SecurityComponentFactory.create_component(43)   def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_49`

**Signature:** `def create_security_security_component_49(): return SecurityComponentFactory.create_component(49)   def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_security_security_component_55`

**Signature:** `def create_security_security_component_55(): return SecurityComponentFactory.create_component(55)   __all__ = [ "ISecurityComponent", "SecurityComponent", "SecurityComponentFactory", "create_security_security_component_1", "create_security_security_component_7", "create_security_security_component_13", "create_security_security_component_19", "create_security_security_component_25", "create_security_security_component_31", "create_security_security_component_37", "create_security_security_component_43", "create_security_security_component_49", "create_security_security_component_55", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_component`

创建组件

**Signature:** `def create_component(self, component_type: str, config: Dict[str, Any]):`

**Parameters:**

- `self: Any` (required)
- `component_type: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_security_id`

获取security ID

**Signature:** `def get_security_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_security_id`

获取security ID

**Signature:** `def get_security_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_component`

创建指定ID的security组件

**Signature:** `def create_component(security_id: int) -> SecurityComponent:`

**Parameters:**

- `security_id: int` (required)

**Returns:** `SecurityComponent`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_available_securitys`

获取所有可用的security ID

**Signature:** `def get_available_securitys() -> List[int]:`

**Returns:** `List[int]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_all_securitys`

创建所有可用security

**Signature:** `def create_all_securitys() -> Dict[int, SecurityComponent]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_factory_info`

获取工厂信息

**Signature:** `def get_factory_info() -> Dict[str, Any]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

---

### config.security_config

RQA2025 安全配置管理器

负责安全模块的配置管理和持久化
分离了AccessControlManager的配置职责

#### API Endpoints

#### `SecurityConfigManager.load_config`

加载配置

Args:
    params: 配置操作参数

Returns:
    加载的配置数据

**Signature:** `def load_config(self, params: ConfigOperationParams) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `params: ConfigOperationParams` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `SecurityConfigManager.save_config`

保存配置

Args:
    config_data: 要保存的配置数据
    params: 配置操作参数

Returns:
    是否保存成功

**Signature:** `def save_config(self, config_data: Dict[str, Any], params: ConfigOperationParams) -> bool:`

**Parameters:**

- `self: Any` (required)
- `config_data: Dict[Any]` (required)
- `params: ConfigOperationParams` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `AuditConfigManager.load_audit_rules`

加载审计规则

**Signature:** `def load_audit_rules(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `AuditConfigManager.save_audit_rules`

保存审计规则

**Signature:** `def save_audit_rules(self, rules: Dict[str, Any]) -> None:`

**Parameters:**

- `self: Any` (required)
- `rules: Dict[Any]` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `load_config`

加载配置

Args:
    params: 配置操作参数

Returns:
    加载的配置数据

**Signature:** `def load_config(self, params: ConfigOperationParams) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `params: ConfigOperationParams` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `save_config`

保存配置

Args:
    config_data: 要保存的配置数据
    params: 配置操作参数

Returns:
    是否保存成功

**Signature:** `def save_config(self, config_data: Dict[str, Any], params: ConfigOperationParams) -> bool:`

**Parameters:**

- `self: Any` (required)
- `config_data: Dict[Any]` (required)
- `params: ConfigOperationParams` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `load_audit_rules`

加载审计规则

**Signature:** `def load_audit_rules(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `save_audit_rules`

保存审计规则

**Signature:** `def save_audit_rules(self, rules: Dict[str, Any]) -> None:`

**Parameters:**

- `self: Any` (required)
- `rules: Dict[Any]` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

---

## Monitoring

### monitoring.health_checker

RQA2025 健康检查器

提供全面的系统健康状态检查和监控
支持多种健康指标的实时监控和报告

#### API Endpoints

#### `should_run`

检查是否应该运行

**Signature:** `def should_run(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `run_check`

运行健康检查

**Signature:** `def run_check(self) -> Tuple[HealthStatus, str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Tuple[Any]`

**Async:** No | **Visibility:** public

#### `to_dict`

转换为字典

**Signature:** `def to_dict(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `add_check`

添加健康检查

Args:
    check: 健康检查对象

**Signature:** `def add_check(self, check: HealthCheck):`

**Parameters:**

- `self: Any` (required)
- `check: HealthCheck` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `remove_check`

移除健康检查

Args:
    check_name: 检查名称

**Signature:** `def remove_check(self, check_name: str):`

**Parameters:**

- `self: Any` (required)
- `check_name: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `run_health_check`

运行健康检查

Args:
    check_name: 指定的检查名称，如果为None则运行所有检查

Returns:
    系统健康状态

**Signature:** `def run_health_check(self, check_name: Optional[str] = None) -> SystemHealth:`

**Parameters:**

- `self: Any` (required)
- `check_name: Optional[str]` (default: None)

**Returns:** `SystemHealth`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭健康检查器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_health_report`

获取健康报告

Returns:
    健康报告字典

**Signature:** `def get_health_report(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `is_healthy`

检查系统是否健康

Returns:
    是否健康

**Signature:** `def is_healthy(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

---

### monitoring.performance_monitor

RQA2025 性能监控器

专门负责监控安全模块的性能指标
提供实时性能统计和优化建议

#### API Endpoints

#### `monitor_performance`

性能监控装饰器

**Signature:** `def monitor_performance(operation_name: str, monitor: Optional[PerformanceMonitor] = None):`

**Parameters:**

- `operation_name: str` (required)
- `monitor: Optional[PerformanceMonitor]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_performance_monitor`

获取全局性能监控器

**Signature:** `def get_performance_monitor() -> PerformanceMonitor:`

**Returns:** `PerformanceMonitor`

**Async:** No | **Visibility:** public

#### `record_performance`

记录性能数据

**Signature:** `def record_performance(operation_name: str, duration: float, is_error: bool = False) -> None:`

**Parameters:**

- `operation_name: str` (required)
- `duration: float` (required)
- `is_error: bool` (default: False)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `record_security_operation`

记录安全操作性能

Args:
    operation: 操作名称
    duration: 执行时间(秒)
    user_id: 用户ID
    resource: 资源标识
    is_error: 是否出错

**Signature:** `def record_security_operation(operation: str, duration: float, user_id: Optional[str] = None, resource: Optional[str] = None, is_error: bool = False) -> None:`

**Parameters:**

- `operation: str` (required)
- `duration: float` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `is_error: bool` (default: False)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_security_performance_report`

获取安全性能报告

Returns:
    安全性能报告

**Signature:** `def get_security_performance_report() -> Dict[str, Any]:`

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `record_call`

记录一次调用

**Signature:** `def record_call(self, duration: float, is_error: bool = False) -> None:`

**Parameters:**

- `self: Any` (required)
- `duration: float` (required)
- `is_error: bool` (default: False)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_error_rate`

获取错误率

**Signature:** `def get_error_rate(self) -> float:`

**Parameters:**

- `self: Any` (required)

**Returns:** `float`

**Async:** No | **Visibility:** public

#### `to_dict`

转换为字典

**Signature:** `def to_dict(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `record_operation`

记录操作性能

**Signature:** `def record_operation(self, operation_name: str, duration: float, is_error: bool = False, user_id: Optional[str] = None, resource: Optional[str] = None) -> None:`

**Parameters:**

- `self: Any` (required)
- `operation_name: str` (required)
- `duration: float` (required)
- `is_error: bool` (default: False)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_metrics`

获取性能指标

**Signature:** `def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `operation_name: Optional[str]` (default: None)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_system_stats`

获取系统统计信息

**Signature:** `def get_system_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_performance_report`

生成性能报告

**Signature:** `def get_performance_report(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `reset_metrics`

重置性能指标

**Signature:** `def reset_metrics(self, operation_name: Optional[str] = None) -> None:`

**Parameters:**

- `self: Any` (required)
- `operation_name: Optional[str]` (default: None)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `decorator`

**Signature:** `def decorator(func: Callable) -> Callable:`

**Parameters:**

- `func: Callable` (required)

**Returns:** `Callable`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭性能监控器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `collect_stats`

**Signature:** `def collect_stats():`

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `wrapper`

**Signature:** `def wrapper(*args, **kwargs):`

**Returns:** `None`

**Async:** No | **Visibility:** public

---

## Other

### access.access_control

RQA2025 访问控制管理器 - 重构版

基于组件化架构的访问控制系统
协调各个组件提供统一的访问控制服务

#### API Endpoints

#### `AccessControlManager.create_user`

创建用户

Args:
    username: 用户名
    email: 邮箱
    roles: 角色列表

Returns:
    用户ID

**Signature:** `def create_user(self, username: str, email: Optional[str] = None, roles: Optional[List[str]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `email: Optional[str]` (default: None)
- `roles: Optional[List[str]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AccessControlManager.assign_role_to_user`

为用户分配角色

Args:
    user_id: 用户ID
    role_id: 角色ID

Returns:
    是否分配成功

**Signature:** `def assign_role_to_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `AccessControlManager.revoke_role_from_user`

从用户撤销角色

Args:
    user_id: 用户ID
    role_id: 角色ID

Returns:
    是否撤销成功

**Signature:** `def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `AccessControlManager.create_role`

创建角色

Args:
    name: 角色名称
    description: 角色描述
    permissions: 权限列表

Returns:
    角色ID

**Signature:** `def create_role(self, name: str, description: str = "", permissions: Optional[List[str]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `description: str` (default: '')
- `permissions: Optional[List[str]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AccessControlManager.check_access`

检查用户对资源的访问权限

Args:
    user_id: 用户ID
    resource: 资源标识
    permission: 请求的权限
    context: 额外的上下文信息

Returns:
    访问决策

**Signature:** `def check_access(self, user_id: str, resource: str, permission: str, context: Optional[Dict[str, Any]] = None) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `permission: str` (required)
- `context: Optional[Dict[Any]]` (default: None)

**Returns:** `AccessDecision`

**Async:** No | **Visibility:** public

#### `AccessControlManager.check_access_async`

异步检查用户对资源的访问权限

Args:
    user_id: 用户ID
    resource: 资源标识
    permission: 请求的权限
    context: 额外的上下文信息

Returns:
    访问决策

**Signature:** `async def check_access_async(self, user_id: str, resource: str, permission: str, context: Optional[Dict[str, Any]] = None) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `permission: str` (required)
- `context: Optional[Dict[Any]]` (default: None)

**Returns:** `AccessDecision`

**Async:** Yes | **Visibility:** public

#### `AccessControlManager.create_access_policy`

创建访问策略

Args:
    name: 策略名称
    resource_pattern: 资源匹配模式
    permissions: 权限集合
    roles: 角色集合
    description: 策略描述
    conditions: 附加条件

Returns:
    策略ID

**Signature:** `def create_access_policy(self, name: str, resource_pattern: str, permissions: Set[str], roles: Set[UserRole], description: str = "", conditions: Optional[Dict[str, Any]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `resource_pattern: str` (required)
- `permissions: Set[str]` (required)
- `roles: Set[UserRole]` (required)
- `description: str` (default: '')
- `conditions: Optional[Dict[Any]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AccessControlManager.get_audit_logs`

获取审计日志

Args:
    user_id: 用户ID过滤
    limit: 结果数量限制

Returns:
    审计日志列表

**Signature:** `def get_audit_logs(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `limit: int` (default: 100)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `AccessControlManager.get_access_statistics`

获取访问统计信息

Returns:
    统计信息

**Signature:** `def get_access_statistics(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `AccessControlManager.clear_cache`

清除所有权限缓存

**Signature:** `def clear_cache(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `AccessControlManager.get_cache_stats`

获取缓存统计信息

Returns:
    缓存统计数据

**Signature:** `def get_cache_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `AccessControlManager.shutdown`

关闭访问控制管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_user`

创建用户

Args:
    username: 用户名
    email: 邮箱
    roles: 角色列表

Returns:
    用户ID

**Signature:** `def create_user(self, username: str, email: Optional[str] = None, roles: Optional[List[str]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `email: Optional[str]` (default: None)
- `roles: Optional[List[str]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `assign_role_to_user`

为用户分配角色

Args:
    user_id: 用户ID
    role_id: 角色ID

Returns:
    是否分配成功

**Signature:** `def assign_role_to_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `revoke_role_from_user`

从用户撤销角色

Args:
    user_id: 用户ID
    role_id: 角色ID

Returns:
    是否撤销成功

**Signature:** `def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_role`

创建角色

Args:
    name: 角色名称
    description: 角色描述
    permissions: 权限列表

Returns:
    角色ID

**Signature:** `def create_role(self, name: str, description: str = "", permissions: Optional[List[str]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `description: str` (default: '')
- `permissions: Optional[List[str]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `check_access`

检查用户对资源的访问权限

Args:
    user_id: 用户ID
    resource: 资源标识
    permission: 请求的权限
    context: 额外的上下文信息

Returns:
    访问决策

**Signature:** `def check_access(self, user_id: str, resource: str, permission: str, context: Optional[Dict[str, Any]] = None) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `permission: str` (required)
- `context: Optional[Dict[Any]]` (default: None)

**Returns:** `AccessDecision`

**Async:** No | **Visibility:** public

#### `check_access_async`

异步检查用户对资源的访问权限

Args:
    user_id: 用户ID
    resource: 资源标识
    permission: 请求的权限
    context: 额外的上下文信息

Returns:
    访问决策

**Signature:** `async def check_access_async(self, user_id: str, resource: str, permission: str, context: Optional[Dict[str, Any]] = None) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `permission: str` (required)
- `context: Optional[Dict[Any]]` (default: None)

**Returns:** `AccessDecision`

**Async:** Yes | **Visibility:** public

#### `create_access_policy`

创建访问策略

Args:
    name: 策略名称
    resource_pattern: 资源匹配模式
    permissions: 权限集合
    roles: 角色集合
    description: 策略描述
    conditions: 附加条件

Returns:
    策略ID

**Signature:** `def create_access_policy(self, name: str, resource_pattern: str, permissions: Set[str], roles: Set[UserRole], description: str = "", conditions: Optional[Dict[str, Any]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `resource_pattern: str` (required)
- `permissions: Set[str]` (required)
- `roles: Set[UserRole]` (required)
- `description: str` (default: '')
- `conditions: Optional[Dict[Any]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `get_audit_logs`

获取审计日志

Args:
    user_id: 用户ID过滤
    limit: 结果数量限制

Returns:
    审计日志列表

**Signature:** `def get_audit_logs(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `limit: int` (default: 100)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_access_statistics`

获取访问统计信息

Returns:
    统计信息

**Signature:** `def get_access_statistics(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `clear_cache`

清除所有权限缓存

**Signature:** `def clear_cache(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_cache_stats`

获取缓存统计信息

Returns:
    缓存统计数据

**Signature:** `def get_cache_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭访问控制管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

---

### access.access_control_component

#### API Endpoints

#### `RBACManager.create_user`

创建用户

**Signature:** `def create_user(self, user_id: str, username: str, email: str,   roles: Set[UserRole], password: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `username: str` (required)
- `email: str` (required)
- `roles: Set[UserRole]` (required)
- `password: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RBACManager.authenticate_user`

用户认证

**Signature:** `def authenticate_user(self, user_id: str, password: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `password: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RBACManager.check_permission`

检查用户权限

**Signature:** `def check_permission(self, user_id: str, resource: str, action: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `action: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RBACManager.add_role_to_user`

为用户添加角色

**Signature:** `def add_role_to_user(self, user_id: str, role: UserRole) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role: UserRole` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RBACManager.remove_role_from_user`

从用户移除角色

**Signature:** `def remove_role_from_user(self, user_id: str, role: UserRole) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role: UserRole` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RBACManager.get_user`

获取用户信息

**Signature:** `def get_user(self, user_id: str) -> Optional[User]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[User]`

**Async:** No | **Visibility:** public

#### `RBACManager.list_users`

列出所有用户

**Signature:** `def list_users(self) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `SessionManager.create_session`

创建会话

**Signature:** `def create_session(self, user_id: str, ip_address: str = "",   user_agent: str = "") -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `ip_address: str` (default: '')
- `user_agent: str` (default: '')

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `SessionManager.validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `SessionManager.destroy_session`

销毁会话

**Signature:** `def destroy_session(self, session_id: str):`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `SessionManager.cleanup_expired_sessions`

清理过期会话

**Signature:** `def cleanup_expired_sessions(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_access_control_system`

获取全局访问控制系统实例

**Signature:** `def get_access_control_system() -> AccessControlSystem:`

**Returns:** `AccessControlSystem`

**Async:** No | **Visibility:** public

#### `authenticate_user`

用户认证

**Signature:** `def authenticate_user(user_id: str, password: str) -> Optional[str]:`

**Parameters:**

- `user_id: str` (required)
- `password: str` (required)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `check_user_permission`

检查用户权限

**Signature:** `def check_user_permission(session_id: str, resource: str, action: str) -> bool:`

**Parameters:**

- `session_id: str` (required)
- `resource: str` (required)
- `action: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_system_user`

创建系统用户

**Signature:** `def create_system_user(user_id: str, username: str, email: str, roles: List[str], password: str) -> bool:`

**Parameters:**

- `user_id: str` (required)
- `username: str` (required)
- `email: str` (required)
- `roles: List[str]` (required)
- `password: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `is_expired`

检查会话是否过期

**Signature:** `def is_expired(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `extend_session`

延长会话时间

**Signature:** `def extend_session(self, minutes: int = 60) -> None:`

**Parameters:**

- `self: Any` (required)
- `minutes: int` (default: 60)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `create_user`

创建用户

**Signature:** `def create_user(self, user_id: str, username: str, email: str,   roles: Set[UserRole], password: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `username: str` (required)
- `email: str` (required)
- `roles: Set[UserRole]` (required)
- `password: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `authenticate_user`

用户认证

**Signature:** `def authenticate_user(self, user_id: str, password: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `password: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `check_permission`

检查用户权限

**Signature:** `def check_permission(self, user_id: str, resource: str, action: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `action: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `add_role_to_user`

为用户添加角色

**Signature:** `def add_role_to_user(self, user_id: str, role: UserRole) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role: UserRole` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `remove_role_from_user`

从用户移除角色

**Signature:** `def remove_role_from_user(self, user_id: str, role: UserRole) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role: UserRole` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_user`

获取用户信息

**Signature:** `def get_user(self, user_id: str) -> Optional[User]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[User]`

**Async:** No | **Visibility:** public

#### `list_users`

列出所有用户

**Signature:** `def list_users(self) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_session`

创建会话

**Signature:** `def create_session(self, user_id: str, ip_address: str = "",   user_agent: str = "") -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `ip_address: str` (default: '')
- `user_agent: str` (default: '')

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `destroy_session`

销毁会话

**Signature:** `def destroy_session(self, session_id: str):`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `cleanup_expired_sessions`

清理过期会话

**Signature:** `def cleanup_expired_sessions(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `authenticate`

用户认证并创建会话

**Signature:** `def authenticate(self, user_id: str, password: str) -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `password: str` (required)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `authorize`

授权检查

**Signature:** `def authorize(self, session_id: str, resource: str, action: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)
- `resource: str` (required)
- `action: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_user`

创建用户

**Signature:** `def create_user(self, user_id: str, username: str, email: str,   roles: List[str], password: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `username: str` (required)
- `email: str` (required)
- `roles: List[str]` (required)
- `password: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_user_info`

获取用户信息

**Signature:** `def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `update_user_roles`

更新用户角色

**Signature:** `def update_user_roles(self, user_id: str, roles: List[str]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `roles: List[str]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `logout`

用户登出

**Signature:** `def logout(self, session_id: str):`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `list_users`

列出所有用户

**Signature:** `def list_users(self) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `check_permission`

检查权限（直接调用，不需要会话）

**Signature:** `def check_permission(self, user_id: str, resource: str, action: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `action: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `cleanup_sessions`

清理过期会话

**Signature:** `def cleanup_sessions(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `health_check`

健康检查

**Signature:** `def health_check(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `collect_permissions`

**Signature:** `def collect_permissions(roles: Set[UserRole]):`

**Parameters:**

- `roles: Set[UserRole]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

---

### access.permission_checker

RQA2025 权限检查器

专门负责权限验证和访问控制逻辑
从AccessControlManager中分离出来，提高代码组织性

#### API Endpoints

#### `to_dict`

转换为字典

**Signature:** `def to_dict(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `to_dict`

转换为字典

**Signature:** `def to_dict(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `check_access`

检查访问权限

**Signature:** `def check_access(self, request: AccessRequest, user_permissions: Set[str], policies: Optional[List['AccessPolicy']] = None) -> AccessResult:`

**Parameters:**

- `self: Any` (required)
- `request: AccessRequest` (required)
- `user_permissions: Set[str]` (required)
- `policies: Optional[List[AccessPolicy]]` (default: None)

**Returns:** `AccessResult`

**Async:** No | **Visibility:** public

#### `get_stats`

获取统计信息

**Signature:** `def get_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `clear_cache`

清除缓存

**Signature:** `def clear_cache(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `set_cache_enabled`

设置缓存启用状态

**Signature:** `def set_cache_enabled(self, enabled: bool) -> None:`

**Parameters:**

- `self: Any` (required)
- `enabled: bool` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `check_access_async`

异步检查访问权限

**Signature:** `async def check_access_async(self, request: AccessRequest, user_permissions: Set[str], policies: Optional[List['AccessPolicy']] = None) -> AccessResult:`

**Parameters:**

- `self: Any` (required)
- `request: AccessRequest` (required)
- `user_permissions: Set[str]` (required)
- `policies: Optional[List[AccessPolicy]]` (default: None)

**Returns:** `AccessResult`

**Async:** Yes | **Visibility:** public

#### `batch_check_access_async`

异步批量检查访问权限

**Signature:** `async def batch_check_access_async(self, requests: List[AccessRequest], user_permissions: Dict[str, Set[str]], policies: Optional[List['AccessPolicy']] = None, max_concurrency: int = 10) -> List[AccessResult]:`

**Parameters:**

- `self: Any` (required)
- `requests: List[AccessRequest]` (required)
- `user_permissions: Dict[Any]` (required)
- `policies: Optional[List[AccessPolicy]]` (default: None)
- `max_concurrency: int` (default: 10)

**Returns:** `List[AccessResult]`

**Async:** Yes | **Visibility:** public

#### `batch_check_access`

批量检查访问权限

**Signature:** `def batch_check_access(self, requests: List[AccessRequest], user_permissions: Dict[str, Set[str]], policies: Optional[List['AccessPolicy']] = None) -> List[AccessResult]:`

**Parameters:**

- `self: Any` (required)
- `requests: List[AccessRequest]` (required)
- `user_permissions: Dict[Any]` (required)
- `policies: Optional[List[AccessPolicy]]` (default: None)

**Returns:** `List[AccessResult]`

**Async:** No | **Visibility:** public

#### `check_single`

**Signature:** `async def check_single(request: AccessRequest) -> AccessResult:`

**Parameters:**

- `request: AccessRequest` (required)

**Returns:** `AccessResult`

**Async:** Yes | **Visibility:** public

---

### access.policy_manager

RQA2025 策略管理器

负责访问控制策略的管理
分离了AccessControlManager的策略职责

#### API Endpoints

#### `PolicyManager.create_policy`

创建访问策略

Args:
    params: 策略创建参数

Returns:
    创建的策略对象

**Signature:** `def create_policy(self, params: PolicyCreationParams) -> 'AccessPolicy':`

**Parameters:**

- `self: Any` (required)
- `params: PolicyCreationParams` (required)

**Returns:** `AccessPolicy`

**Async:** No | **Visibility:** public

#### `PolicyManager.evaluate_policies`

评估适用的策略

Args:
    user: 用户对象
    resource: 资源
    permission: 权限

Returns:
    适用的策略列表

**Signature:** `def evaluate_policies(self, user: 'User', resource: str, permission: str) -> List['AccessPolicy']:`

**Parameters:**

- `self: Any` (required)
- `user: User` (required)
- `resource: str` (required)
- `permission: str` (required)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `PolicyManager.check_policy_access`

检查策略访问权限

Args:
    policies: 适用的策略列表
    permission: 请求的权限
    context: 访问上下文

Returns:
    是否允许访问

**Signature:** `def check_policy_access(self, policies: List['AccessPolicy'], permission: str, context: Dict) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policies: List[AccessPolicy]` (required)
- `permission: str` (required)
- `context: Dict` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PolicyManager.update_policy`

更新策略

**Signature:** `def update_policy(self, policy_id: str, **kwargs) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PolicyManager.delete_policy`

删除策略

**Signature:** `def delete_policy(self, policy_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PolicyManager.list_policies`

列出策略

**Signature:** `def list_policies(self, active_only: bool = True) -> List['AccessPolicy']:`

**Parameters:**

- `self: Any` (required)
- `active_only: bool` (default: True)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `PolicyManager.get_policy`

获取策略

**Signature:** `def get_policy(self, policy_id: str) -> Optional['AccessPolicy']:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `Optional[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `SessionManager.create_session`

创建会话

**Signature:** `def create_session(self, user_id: str, **kwargs) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `SessionManager.get_session`

获取会话

**Signature:** `def get_session(self, session_id: str) -> Optional['UserSession']:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[UserSession]`

**Async:** No | **Visibility:** public

#### `SessionManager.invalidate_session`

使会话失效

**Signature:** `def invalidate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `CacheManager.clear`

清除所有缓存

**Signature:** `def clear(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `create_policy`

创建访问策略

Args:
    params: 策略创建参数

Returns:
    创建的策略对象

**Signature:** `def create_policy(self, params: PolicyCreationParams) -> 'AccessPolicy':`

**Parameters:**

- `self: Any` (required)
- `params: PolicyCreationParams` (required)

**Returns:** `AccessPolicy`

**Async:** No | **Visibility:** public

#### `evaluate_policies`

评估适用的策略

Args:
    user: 用户对象
    resource: 资源
    permission: 权限

Returns:
    适用的策略列表

**Signature:** `def evaluate_policies(self, user: 'User', resource: str, permission: str) -> List['AccessPolicy']:`

**Parameters:**

- `self: Any` (required)
- `user: User` (required)
- `resource: str` (required)
- `permission: str` (required)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `check_policy_access`

检查策略访问权限

Args:
    policies: 适用的策略列表
    permission: 请求的权限
    context: 访问上下文

Returns:
    是否允许访问

**Signature:** `def check_policy_access(self, policies: List['AccessPolicy'], permission: str, context: Dict) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policies: List[AccessPolicy]` (required)
- `permission: str` (required)
- `context: Dict` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `update_policy`

更新策略

**Signature:** `def update_policy(self, policy_id: str, **kwargs) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `delete_policy`

删除策略

**Signature:** `def delete_policy(self, policy_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `list_policies`

列出策略

**Signature:** `def list_policies(self, active_only: bool = True) -> List['AccessPolicy']:`

**Parameters:**

- `self: Any` (required)
- `active_only: bool` (default: True)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `get_policy`

获取策略

**Signature:** `def get_policy(self, policy_id: str) -> Optional['AccessPolicy']:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `Optional[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `create_session`

创建会话

**Signature:** `def create_session(self, user_id: str, **kwargs) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `get_session`

获取会话

**Signature:** `def get_session(self, session_id: str) -> Optional['UserSession']:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[UserSession]`

**Async:** No | **Visibility:** public

#### `invalidate_session`

使会话失效

**Signature:** `def invalidate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `clear`

清除所有缓存

**Signature:** `def clear(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

---

### access.components.access_checker

RQA2025 访问控制组件 - 访问检查器

负责权限检查的核心逻辑和访问决策

#### API Endpoints

#### `check_access`

检查用户对资源的访问权限

Args:
    user_id: 用户ID
    resource: 资源标识
    permission: 请求的权限
    context: 额外的上下文信息

Returns:
    访问决策

**Signature:** `def check_access(self, user_id: str, resource: str, permission: str, context: Optional[Dict[str, Any]] = None) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `permission: str` (required)
- `context: Optional[Dict[Any]]` (default: None)

**Returns:** `AccessDecision`

**Async:** No | **Visibility:** public

#### `check_access_async`

异步检查用户对资源的访问权限

Args:
    user_id: 用户ID
    resource: 资源标识
    permission: 请求的权限
    context: 额外的上下文信息

Returns:
    访问决策

**Signature:** `async def check_access_async(self, user_id: str, resource: str, permission: str, context: Optional[Dict[str, Any]] = None) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `permission: str` (required)
- `context: Optional[Dict[Any]]` (default: None)

**Returns:** `AccessDecision`

**Async:** Yes | **Visibility:** public

#### `check_access_request`

检查访问请求

Args:
    request: 访问请求对象

Returns:
    访问决策

**Signature:** `def check_access_request(self, request: AccessRequest) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `request: AccessRequest` (required)

**Returns:** `AccessDecision`

**Async:** No | **Visibility:** public

#### `batch_check_access`

批量检查访问权限

Args:
    requests: 访问请求列表

Returns:
    访问决策列表

**Signature:** `def batch_check_access(self, requests: List[AccessRequest]) -> List[AccessDecision]:`

**Parameters:**

- `self: Any` (required)
- `requests: List[AccessRequest]` (required)

**Returns:** `List[AccessDecision]`

**Async:** No | **Visibility:** public

#### `batch_check_access_async`

异步批量检查访问权限

Args:
    requests: 访问请求列表

Returns:
    访问决策列表

**Signature:** `async def batch_check_access_async(self, requests: List[AccessRequest]) -> List[AccessDecision]:`

**Parameters:**

- `self: Any` (required)
- `requests: List[AccessRequest]` (required)

**Returns:** `List[AccessDecision]`

**Async:** Yes | **Visibility:** public

#### `clear_cache`

清除所有权限缓存

**Signature:** `def clear_cache(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_cache_stats`

获取缓存统计信息

Returns:
    缓存统计数据

**Signature:** `def get_cache_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `invalidate_user_cache`

使指定用户的缓存失效

Args:
    user_id: 用户ID

**Signature:** `def invalidate_user_cache(self, user_id: str):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

---

### access.components.audit_logger

RQA2025 访问控制组件 - 审计日志器

负责访问控制相关的审计日志记录和管理

#### API Endpoints

#### `log_access_check`

记录访问检查事件

Args:
    request: 访问请求
    decision: 访问决策
    details: 额外详情

**Signature:** `def log_access_check(self, request: AccessRequest, decision: AccessDecision, details: Optional[Dict[str, Any]] = None):`

**Parameters:**

- `self: Any` (required)
- `request: AccessRequest` (required)
- `decision: AccessDecision` (required)
- `details: Optional[Dict[Any]]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `log_user_action`

记录用户操作事件

Args:
    user_id: 用户ID
    action: 操作类型
    resource: 资源标识
    details: 操作详情

**Signature:** `def log_user_action(self, user_id: str, action: str, resource: str, details: Optional[Dict[str, Any]] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `action: str` (required)
- `resource: str` (required)
- `details: Optional[Dict[Any]]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `log_security_event`

记录安全事件

Args:
    event_type: 事件类型
    severity: 严重程度
    user_id: 用户ID
    description: 事件描述
    details: 事件详情

**Signature:** `def log_security_event(self, event_type: str, severity: str, user_id: str, description: str, details: Optional[Dict[str, Any]] = None):`

**Parameters:**

- `self: Any` (required)
- `event_type: str` (required)
- `severity: str` (required)
- `user_id: str` (required)
- `description: str` (required)
- `details: Optional[Dict[Any]]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `query_audit_logs`

查询审计日志

Args:
    user_id: 用户ID过滤
    action: 操作类型过滤
    resource: 资源过滤
    start_time: 开始时间
    end_time: 结束时间
    limit: 结果数量限制

Returns:
    审计事件列表

**Signature:** `def query_audit_logs(self, user_id: Optional[str] = None, action: Optional[str] = None, resource: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 100) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `action: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)
- `limit: int` (default: 100)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `get_audit_statistics`

获取审计统计信息

Args:
    days: 统计天数

Returns:
    统计信息

**Signature:** `def get_audit_statistics(self, days: int = 7) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `days: int` (default: 7)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `export_audit_logs`

导出审计日志

Args:
    file_path: 导出文件路径
    user_id: 用户ID过滤
    start_time: 开始时间
    end_time: 结束时间

Returns:
    是否导出成功

**Signature:** `def export_audit_logs(self, file_path: Path, user_id: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `file_path: Path` (required)
- `user_id: Optional[str]` (default: None)
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭审计日志器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

---

### access.components.cache_manager

RQA2025 访问控制组件 - 缓存管理器

负责权限检查结果的缓存管理，提高系统性能

#### API Endpoints

#### `CacheManager.delete`

删除缓存条目

Args:
    user_id: 用户ID
    resource: 资源标识（可选）
    permission: 权限名（可选）

**Signature:** `def delete(self, user_id: str, resource: Optional[str] = None, permission: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: Optional[str]` (default: None)
- `permission: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `CacheManager.clear`

清空所有缓存

**Signature:** `def clear(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `CacheManager.cleanup`

手动执行缓存清理

**Signature:** `def cleanup(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `CacheManager.get_stats`

获取缓存统计信息

Returns:
    统计信息字典

**Signature:** `def get_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `CacheManager.get_entries_for_user`

获取指定用户的所有缓存条目

Args:
    user_id: 用户ID

Returns:
    用户的缓存条目字典

**Signature:** `def get_entries_for_user(self, user_id: str) -> Dict[str, Dict[str, AccessDecision]]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `CacheManager.shutdown`

关闭缓存管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `is_expired`

检查是否过期

**Signature:** `def is_expired(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `touch`

更新访问时间和计数

**Signature:** `def touch(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `delete`

删除缓存条目

Args:
    user_id: 用户ID
    resource: 资源标识（可选）
    permission: 权限名（可选）

**Signature:** `def delete(self, user_id: str, resource: Optional[str] = None, permission: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: Optional[str]` (default: None)
- `permission: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `clear`

清空所有缓存

**Signature:** `def clear(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `cleanup`

手动执行缓存清理

**Signature:** `def cleanup(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_stats`

获取缓存统计信息

Returns:
    统计信息字典

**Signature:** `def get_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_entries_for_user`

获取指定用户的所有缓存条目

Args:
    user_id: 用户ID

Returns:
    用户的缓存条目字典

**Signature:** `def get_entries_for_user(self, user_id: str) -> Dict[str, Dict[str, AccessDecision]]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭缓存管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

---

### access.components.config_manager

RQA2025 访问控制组件 - 配置管理器

负责访问控制相关配置的持久化和管理

#### API Endpoints

#### `ConfigManager.get_config`

获取配置值

Args:
    key: 配置键，支持点分隔的嵌套键，如 "cache.enabled"

Returns:
    配置值

**Signature:** `def get_config(self, key: Optional[str] = None) -> Any:`

**Parameters:**

- `self: Any` (required)
- `key: Optional[str]` (default: None)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `ConfigManager.set_config`

设置配置值

Args:
    key: 配置键，支持点分隔的嵌套键
    value: 配置值

Returns:
    是否设置成功

**Signature:** `def set_config(self, key: str, value: Any) -> bool:`

**Parameters:**

- `self: Any` (required)
- `key: str` (required)
- `value: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `ConfigManager.update_config`

批量更新配置

Args:
    updates: 配置更新字典

Returns:
    是否更新成功

**Signature:** `def update_config(self, updates: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `updates: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `ConfigManager.reset_config`

重置配置到默认值

Args:
    section: 要重置的配置节，如果为None则重置所有配置

Returns:
    是否重置成功

**Signature:** `def reset_config(self, section: Optional[str] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `section: Optional[str]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `ConfigManager.validate_config`

验证配置有效性

Returns:
    验证结果字典

**Signature:** `def validate_config(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `ConfigManager.export_config`

导出配置到文件

Args:
    file_path: 导出文件路径

Returns:
    是否导出成功

**Signature:** `def export_config(self, file_path: Path) -> bool:`

**Parameters:**

- `self: Any` (required)
- `file_path: Path` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `ConfigManager.import_config`

从文件导入配置

Args:
    file_path: 导入文件路径
    merge: 是否合并到现有配置，False则完全替换

Returns:
    是否导入成功

**Signature:** `def import_config(self, file_path: Path, merge: bool = True) -> bool:`

**Parameters:**

- `self: Any` (required)
- `file_path: Path` (required)
- `merge: bool` (default: True)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `ConfigManager.get_config_summary`

获取配置摘要信息

Returns:
    配置摘要

**Signature:** `def get_config_summary(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `ConfigManager.add_config_change_callback`

添加配置变更回调函数

Args:
    callback: 回调函数，参数为新的配置字典

**Signature:** `def add_config_change_callback(self, callback: Callable[[Dict[str, Any]], None]):`

**Parameters:**

- `self: Any` (required)
- `callback: Callable[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `ConfigManager.remove_config_change_callback`

移除配置变更回调函数

Args:
    callback: 要移除的回调函数

**Signature:** `def remove_config_change_callback(self, callback: Callable[[Dict[str, Any]], None]):`

**Parameters:**

- `self: Any` (required)
- `callback: Callable[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `ConfigManager.trigger_manual_reload`

手动触发配置重新加载

Returns:
    重新加载是否成功

**Signature:** `def trigger_manual_reload(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `ConfigManager.shutdown`

关闭配置管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_config`

获取配置值

Args:
    key: 配置键，支持点分隔的嵌套键，如 "cache.enabled"

Returns:
    配置值

**Signature:** `def get_config(self, key: Optional[str] = None) -> Any:`

**Parameters:**

- `self: Any` (required)
- `key: Optional[str]` (default: None)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `set_config`

设置配置值

Args:
    key: 配置键，支持点分隔的嵌套键
    value: 配置值

Returns:
    是否设置成功

**Signature:** `def set_config(self, key: str, value: Any) -> bool:`

**Parameters:**

- `self: Any` (required)
- `key: str` (required)
- `value: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `update_config`

批量更新配置

Args:
    updates: 配置更新字典

Returns:
    是否更新成功

**Signature:** `def update_config(self, updates: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `updates: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `reset_config`

重置配置到默认值

Args:
    section: 要重置的配置节，如果为None则重置所有配置

Returns:
    是否重置成功

**Signature:** `def reset_config(self, section: Optional[str] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `section: Optional[str]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `validate_config`

验证配置有效性

Returns:
    验证结果字典

**Signature:** `def validate_config(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `export_config`

导出配置到文件

Args:
    file_path: 导出文件路径

Returns:
    是否导出成功

**Signature:** `def export_config(self, file_path: Path) -> bool:`

**Parameters:**

- `self: Any` (required)
- `file_path: Path` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `import_config`

从文件导入配置

Args:
    file_path: 导入文件路径
    merge: 是否合并到现有配置，False则完全替换

Returns:
    是否导入成功

**Signature:** `def import_config(self, file_path: Path, merge: bool = True) -> bool:`

**Parameters:**

- `self: Any` (required)
- `file_path: Path` (required)
- `merge: bool` (default: True)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_config_summary`

获取配置摘要信息

Returns:
    配置摘要

**Signature:** `def get_config_summary(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `add_config_change_callback`

添加配置变更回调函数

Args:
    callback: 回调函数，参数为新的配置字典

**Signature:** `def add_config_change_callback(self, callback: Callable[[Dict[str, Any]], None]):`

**Parameters:**

- `self: Any` (required)
- `callback: Callable[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `remove_config_change_callback`

移除配置变更回调函数

Args:
    callback: 要移除的回调函数

**Signature:** `def remove_config_change_callback(self, callback: Callable[[Dict[str, Any]], None]):`

**Parameters:**

- `self: Any` (required)
- `callback: Callable[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `trigger_manual_reload`

手动触发配置重新加载

Returns:
    重新加载是否成功

**Signature:** `def trigger_manual_reload(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭配置管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `update_nested_dict`

递归更新嵌套字典

**Signature:** `def update_nested_dict(target: Dict[str, Any], source: Dict[str, Any]):`

**Parameters:**

- `target: Dict[Any]` (required)
- `source: Dict[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

---

### access.components.policy_manager

RQA2025 访问控制组件 - 策略管理器

负责访问策略的定义、管理和评估

#### API Endpoints

#### `PolicyManager.create_policy`

创建访问策略

Args:
    name: 策略名称
    resource_pattern: 资源匹配模式
    permissions: 权限集合
    roles: 角色集合
    description: 策略描述
    conditions: 附加条件

Returns:
    策略ID

**Signature:** `def create_policy(self, name: str, resource_pattern: str, permissions: Set[str], roles: Set[UserRole], description: str = "", conditions: Optional[Dict[str, Any]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `resource_pattern: str` (required)
- `permissions: Set[str]` (required)
- `roles: Set[UserRole]` (required)
- `description: str` (default: '')
- `conditions: Optional[Dict[Any]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `PolicyManager.get_policy`

获取策略信息

Args:
    policy_id: 策略ID

Returns:
    策略对象或None

**Signature:** `def get_policy(self, policy_id: str) -> Optional[AccessPolicy]:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `Optional[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `PolicyManager.update_policy`

更新策略信息

Args:
    policy_id: 策略ID
    updates: 更新内容

Returns:
    是否更新成功

**Signature:** `def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)
- `updates: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PolicyManager.delete_policy`

删除策略

Args:
    policy_id: 策略ID

Returns:
    是否删除成功

**Signature:** `def delete_policy(self, policy_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PolicyManager.evaluate_policies`

评估访问策略

Args:
    request: 访问请求
    user_permissions: 用户权限集合

Returns:
    访问决策

**Signature:** `def evaluate_policies(self, request: AccessRequest, user_permissions: Set[str]) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `request: AccessRequest` (required)
- `user_permissions: Set[str]` (required)

**Returns:** `AccessDecision`

**Async:** No | **Visibility:** public

#### `PolicyManager.list_policies`

获取所有策略列表

Returns:
    策略列表

**Signature:** `def list_policies(self) -> List[AccessPolicy]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `PolicyManager.get_policies_for_resource`

获取适用于指定资源的策略

Args:
    resource: 资源标识

Returns:
    策略列表

**Signature:** `def get_policies_for_resource(self, resource: str) -> List[AccessPolicy]:`

**Parameters:**

- `self: Any` (required)
- `resource: str` (required)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `create_policy`

创建访问策略

Args:
    name: 策略名称
    resource_pattern: 资源匹配模式
    permissions: 权限集合
    roles: 角色集合
    description: 策略描述
    conditions: 附加条件

Returns:
    策略ID

**Signature:** `def create_policy(self, name: str, resource_pattern: str, permissions: Set[str], roles: Set[UserRole], description: str = "", conditions: Optional[Dict[str, Any]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `resource_pattern: str` (required)
- `permissions: Set[str]` (required)
- `roles: Set[UserRole]` (required)
- `description: str` (default: '')
- `conditions: Optional[Dict[Any]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `get_policy`

获取策略信息

Args:
    policy_id: 策略ID

Returns:
    策略对象或None

**Signature:** `def get_policy(self, policy_id: str) -> Optional[AccessPolicy]:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `Optional[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `update_policy`

更新策略信息

Args:
    policy_id: 策略ID
    updates: 更新内容

Returns:
    是否更新成功

**Signature:** `def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)
- `updates: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `delete_policy`

删除策略

Args:
    policy_id: 策略ID

Returns:
    是否删除成功

**Signature:** `def delete_policy(self, policy_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `policy_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `evaluate_policies`

评估访问策略

Args:
    request: 访问请求
    user_permissions: 用户权限集合

Returns:
    访问决策

**Signature:** `def evaluate_policies(self, request: AccessRequest, user_permissions: Set[str]) -> AccessDecision:`

**Parameters:**

- `self: Any` (required)
- `request: AccessRequest` (required)
- `user_permissions: Set[str]` (required)

**Returns:** `AccessDecision`

**Async:** No | **Visibility:** public

#### `list_policies`

获取所有策略列表

Returns:
    策略列表

**Signature:** `def list_policies(self) -> List[AccessPolicy]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

#### `get_policies_for_resource`

获取适用于指定资源的策略

Args:
    resource: 资源标识

Returns:
    策略列表

**Signature:** `def get_policies_for_resource(self, resource: str) -> List[AccessPolicy]:`

**Parameters:**

- `self: Any` (required)
- `resource: str` (required)

**Returns:** `List[AccessPolicy]`

**Async:** No | **Visibility:** public

---

### audit.audit_auditor

基础设施层 - 配置管理组件

security_auditor 模块

配置管理相关的文件
提供配置管理相关的功能实现。

#### API Endpoints

#### `record_login`

记录登录事件

**Signature:** `def record_login(self, user_id: str, success: bool, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `success: bool` (required)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_logout`

记录登出事件

**Signature:** `def record_logout(self, user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_access`

记录访问事件

**Signature:** `def record_access(self, user_id: Optional[str] = None, resource: Optional[str] = None, action: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_modification`

记录修改事件

**Signature:** `def record_modification(self, user_id: Optional[str] = None, resource: Optional[str] = None, action: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_deletion`

记录删除事件

**Signature:** `def record_deletion(self, user_id: Optional[str] = None, resource: Optional[str] = None, action: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_configuration_change`

记录配置变更事件

**Signature:** `def record_configuration_change(self, user_id: Optional[str] = None, resource: Optional[str] = None, action: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_security_violation`

记录安全违规事件

**Signature:** `def record_security_violation(self, user_id: Optional[str] = None, resource: Optional[str] = None, action: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_system_startup`

记录系统启动事件

**Signature:** `def record_system_startup(self, user_id: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `record_system_shutdown`

记录系统关机事件

**Signature:** `def record_system_shutdown(self, user_id: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_audit_events`

获取审计事件

**Signature:** `def get_audit_events(self, event_type: Optional[AuditEventType] = None, user_id: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 100) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `event_type: Optional[AuditEventType]` (default: None)
- `user_id: Optional[str]` (default: None)
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)
- `limit: int` (default: 100)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `get_compliance_rules`

获取所有合规规则

**Signature:** `def get_compliance_rules(self) -> Dict[str, ComplianceRule]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_compliance_rule`

获取单个合规规则

**Signature:** `def get_compliance_rule(self, rule_id: str) -> Optional[ComplianceRule]:`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `Optional[ComplianceRule]`

**Async:** No | **Visibility:** public

#### `update_compliance_rule`

更新合规规则

**Signature:** `def update_compliance_rule(self, rule_id: str, enabled: bool = None, status: str = None):`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)
- `enabled: bool` (default: None)
- `status: str` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `check_compliance`

检查事件是否符合所有合规规则

**Signature:** `def check_compliance(self, event: AuditEvent) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_compliance_report`

生成合规报告

**Signature:** `def get_compliance_report(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `export_audit_events`

导出审计事件

**Signature:** `def export_audit_events(self, output_file: str):`

**Parameters:**

- `self: Any` (required)
- `output_file: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `export_compliance_report`

导出合规报告

**Signature:** `def export_compliance_report(self, output_file: str):`

**Parameters:**

- `self: Any` (required)
- `output_file: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_recommendations`

获取修复建议

Args:
    report: 安全报告

Returns:
    按严重程度分组的修复建议

**Signature:** `def get_recommendations(self, report: Dict[str, Any]) -> Dict[str, List[str]]:`

**Parameters:**

- `self: Any` (required)
- `report: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

---

### audit.audit_events

RQA2025 审计事件管理器

专门负责审计事件的定义、创建和管理
从AuditLoggingManager中分离出来，提高代码组织性

#### API Endpoints

#### `AuditEventManager.create_event`

创建审计事件

**Signature:** `def create_event(self, event_type: AuditEventType, severity: AuditSeverity, user_id: Optional[str] = None, resource: Optional[str] = None, action: str = "", result: str = "") -> AuditEvent:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `severity: AuditSeverity` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: str` (default: '')
- `result: str` (default: '')

**Returns:** `AuditEvent`

**Async:** No | **Visibility:** public

#### `AuditEventManager.add_event`

添加审计事件

**Signature:** `def add_event(self, event: AuditEvent) -> None:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `AuditEventManager.get_events`

获取审计事件

**Signature:** `def get_events(self, filter_obj: Optional[AuditEventFilter] = None, limit: Optional[int] = None) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `filter_obj: Optional[AuditEventFilter]` (default: None)
- `limit: Optional[int]` (default: None)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `AuditEventManager.get_events_async`

异步获取审计事件

**Signature:** `async def get_events_async(self, filter_obj: Optional[AuditEventFilter] = None, limit: Optional[int] = None) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `filter_obj: Optional[AuditEventFilter]` (default: None)
- `limit: Optional[int]` (default: None)

**Returns:** `List[AuditEvent]`

**Async:** Yes | **Visibility:** public

#### `AuditEventManager.create_event_async`

异步创建审计事件

**Signature:** `async def create_event_async(self, event_type: AuditEventType, severity: AuditSeverity, user_id: Optional[str] = None, resource: Optional[str] = None, action: str = "", result: str = "") -> AuditEvent:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `severity: AuditSeverity` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: str` (default: '')
- `result: str` (default: '')

**Returns:** `AuditEvent`

**Async:** Yes | **Visibility:** public

#### `AuditEventManager.clear_events`

清除所有事件

**Signature:** `def clear_events(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `AuditEventManager.get_event_count`

获取事件总数

**Signature:** `def get_event_count(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `AuditEventManager.get_events_by_type`

按类型获取事件

**Signature:** `def get_events_by_type(self, event_type: AuditEventType) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `AuditEventManager.get_events_by_severity`

按严重程度获取事件

**Signature:** `def get_events_by_severity(self, severity: AuditSeverity) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `severity: AuditSeverity` (required)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `AuditEventManager.get_recent_events`

获取最近的事件

**Signature:** `def get_recent_events(self, hours: int = 24) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `hours: int` (default: 24)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `to_dict`

转换为字典格式

**Signature:** `def to_dict(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `from_dict`

从字典创建事件

**Signature:** `def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':`

**Parameters:**

- `cls: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `AuditEvent`

**Decorators:** `classmethod`

**Async:** No | **Visibility:** public

#### `new_event`

创建新的事件

**Signature:** `def new_event(self, event_type: AuditEventType, severity: AuditSeverity) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `severity: AuditSeverity` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_user`

设置用户ID

**Signature:** `def with_user(self, user_id: Optional[str]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_session`

设置会话ID

**Signature:** `def with_session(self, session_id: Optional[str]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `session_id: Optional[str]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_resource`

设置资源

**Signature:** `def with_resource(self, resource: Optional[str]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `resource: Optional[str]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_action`

设置操作

**Signature:** `def with_action(self, action: str) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `action: str` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_result`

设置结果

**Signature:** `def with_result(self, result: str) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `result: str` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_details`

设置详细信息

**Signature:** `def with_details(self, details: Dict[str, Any]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `details: Dict[Any]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_ip_address`

设置IP地址

**Signature:** `def with_ip_address(self, ip_address: Optional[str]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `ip_address: Optional[str]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_user_agent`

设置用户代理

**Signature:** `def with_user_agent(self, user_agent: Optional[str]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `user_agent: Optional[str]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_location`

设置位置

**Signature:** `def with_location(self, location: Optional[str]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `location: Optional[str]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_risk_score`

设置风险分数

**Signature:** `def with_risk_score(self, risk_score: float) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `risk_score: float` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `with_tags`

设置标签

**Signature:** `def with_tags(self, tags: Set[str]) -> 'AuditEventBuilder':`

**Parameters:**

- `self: Any` (required)
- `tags: Set[str]` (required)

**Returns:** `AuditEventBuilder`

**Async:** No | **Visibility:** public

#### `build`

构建事件

**Signature:** `def build(self) -> AuditEvent:`

**Parameters:**

- `self: Any` (required)

**Returns:** `AuditEvent`

**Async:** No | **Visibility:** public

#### `by_event_type`

按事件类型过滤

**Signature:** `def by_event_type(self, event_types: List[AuditEventType]) -> 'AuditEventFilter':`

**Parameters:**

- `self: Any` (required)
- `event_types: List[AuditEventType]` (required)

**Returns:** `AuditEventFilter`

**Async:** No | **Visibility:** public

#### `by_severity`

按严重程度过滤

**Signature:** `def by_severity(self, severities: List[AuditSeverity]) -> 'AuditEventFilter':`

**Parameters:**

- `self: Any` (required)
- `severities: List[AuditSeverity]` (required)

**Returns:** `AuditEventFilter`

**Async:** No | **Visibility:** public

#### `by_user`

按用户过滤

**Signature:** `def by_user(self, user_ids: List[str]) -> 'AuditEventFilter':`

**Parameters:**

- `self: Any` (required)
- `user_ids: List[str]` (required)

**Returns:** `AuditEventFilter`

**Async:** No | **Visibility:** public

#### `by_time_range`

按时间范围过滤

**Signature:** `def by_time_range(self, start_time: datetime, end_time: datetime) -> 'AuditEventFilter':`

**Parameters:**

- `self: Any` (required)
- `start_time: datetime` (required)
- `end_time: datetime` (required)

**Returns:** `AuditEventFilter`

**Async:** No | **Visibility:** public

#### `by_resource`

按资源过滤

**Signature:** `def by_resource(self, resources: List[str]) -> 'AuditEventFilter':`

**Parameters:**

- `self: Any` (required)
- `resources: List[str]` (required)

**Returns:** `AuditEventFilter`

**Async:** No | **Visibility:** public

#### `by_risk_score`

按风险分数过滤

**Signature:** `def by_risk_score(self, min_score: float, max_score: float) -> 'AuditEventFilter':`

**Parameters:**

- `self: Any` (required)
- `min_score: float` (required)
- `max_score: float` (required)

**Returns:** `AuditEventFilter`

**Async:** No | **Visibility:** public

#### `matches`

检查事件是否匹配过滤条件

**Signature:** `def matches(self, event: AuditEvent) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_event`

创建审计事件

**Signature:** `def create_event(self, event_type: AuditEventType, severity: AuditSeverity, user_id: Optional[str] = None, resource: Optional[str] = None, action: str = "", result: str = "") -> AuditEvent:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `severity: AuditSeverity` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: str` (default: '')
- `result: str` (default: '')

**Returns:** `AuditEvent`

**Async:** No | **Visibility:** public

#### `add_event`

添加审计事件

**Signature:** `def add_event(self, event: AuditEvent) -> None:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_events`

获取审计事件

**Signature:** `def get_events(self, filter_obj: Optional[AuditEventFilter] = None, limit: Optional[int] = None) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `filter_obj: Optional[AuditEventFilter]` (default: None)
- `limit: Optional[int]` (default: None)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `get_events_async`

异步获取审计事件

**Signature:** `async def get_events_async(self, filter_obj: Optional[AuditEventFilter] = None, limit: Optional[int] = None) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `filter_obj: Optional[AuditEventFilter]` (default: None)
- `limit: Optional[int]` (default: None)

**Returns:** `List[AuditEvent]`

**Async:** Yes | **Visibility:** public

#### `create_event_async`

异步创建审计事件

**Signature:** `async def create_event_async(self, event_type: AuditEventType, severity: AuditSeverity, user_id: Optional[str] = None, resource: Optional[str] = None, action: str = "", result: str = "") -> AuditEvent:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `severity: AuditSeverity` (required)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `action: str` (default: '')
- `result: str` (default: '')

**Returns:** `AuditEvent`

**Async:** Yes | **Visibility:** public

#### `clear_events`

清除所有事件

**Signature:** `def clear_events(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_event_count`

获取事件总数

**Signature:** `def get_event_count(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_events_by_type`

按类型获取事件

**Signature:** `def get_events_by_type(self, event_type: AuditEventType) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `get_events_by_severity`

按严重程度获取事件

**Signature:** `def get_events_by_severity(self, severity: AuditSeverity) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `severity: AuditSeverity` (required)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `get_recent_events`

获取最近的事件

**Signature:** `def get_recent_events(self, hours: int = 24) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `hours: int` (default: 24)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

---

### audit.audit_logging_manager

RQA2025 审计日志管理器

实现全面的操作审计和合规日志功能
提供安全监控、异常检测和合规报告

#### API Endpoints

#### `AuditLoggingManager.log_event`

记录审计事件

Args:
    event_type: 事件类型
    severity: 严重程度
    user_id: 用户ID
    action: 操作
    result: 结果
    resource: 资源
    session_id: 会话ID
    details: 详细信息
    ip_address: IP地址
    user_agent: 用户代理
    location: 地理位置
    risk_score: 风险分数
    tags: 标签

Returns:
    事件ID

**Signature:** `def log_event(self, event_type: AuditEventType, severity: AuditSeverity,   user_id: Optional[str], action: str, result: str, resource: Optional[str] = None, session_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None, location: Optional[str] = None, risk_score: float = 0.0, tags: Optional[Set[str]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `severity: AuditSeverity` (required)
- `user_id: Optional[str]` (required)
- `action: str` (required)
- `result: str` (required)
- `resource: Optional[str]` (default: None)
- `session_id: Optional[str]` (default: None)
- `details: Optional[Dict[Any]]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)
- `location: Optional[str]` (default: None)
- `risk_score: float` (default: 0.0)
- `tags: Optional[Set[str]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.log_security_event`

记录安全事件

Args:
    user_id: 用户ID
    action: 操作
    result: 结果
    details: 详细信息
    ip_address: IP地址
    risk_score: 风险分数

Returns:
    事件ID

**Signature:** `def log_security_event(self, user_id: Optional[str], action: str, result: str,   details: Optional[Dict[str, Any]] = None, ip_address: Optional[str] = None, risk_score: float = 0.0) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (required)
- `action: str` (required)
- `result: str` (required)
- `details: Optional[Dict[Any]]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `risk_score: float` (default: 0.0)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.log_access_event`

记录访问事件

Args:
    user_id: 用户ID
    resource: 资源
    action: 操作
    result: 结果
    session_id: 会话ID
    ip_address: IP地址
    risk_score: 风险分数

Returns:
    事件ID

**Signature:** `def log_access_event(self, user_id: str, resource: str, action: str, result: str,   session_id: Optional[str] = None, ip_address: Optional[str] = None, risk_score: float = 0.0) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `action: str` (required)
- `result: str` (required)
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `risk_score: float` (default: 0.0)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.log_data_operation`

记录数据操作事件

Args:
    user_id: 用户ID
    operation: 操作
    resource: 资源
    result: 结果
    details: 详细信息

Returns:
    事件ID

**Signature:** `def log_data_operation(self, user_id: str, operation: str, resource: str,   result: str, details: Optional[Dict[str, Any]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `operation: str` (required)
- `resource: str` (required)
- `result: str` (required)
- `details: Optional[Dict[Any]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.query_events`

查询审计事件

Args:
    start_time: 开始时间
    end_time: 结束时间
    event_type: 事件类型
    user_id: 用户ID
    resource: 资源
    result: 结果
    limit: 限制条数

Returns:
    审计事件列表

**Signature:** `def query_events(self, start_time: Optional[datetime] = None,   end_time: Optional[datetime] = None, event_type: Optional[AuditEventType] = None, user_id: Optional[str] = None, resource: Optional[str] = None, result: Optional[str] = None, limit: int = 100) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)
- `event_type: Optional[AuditEventType]` (default: None)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `result: Optional[str]` (default: None)
- `limit: int` (default: 100)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.get_security_report`

获取安全报告

Args:
    days: 报告天数

Returns:
    安全报告

**Signature:** `def get_security_report(self, days: int = 7) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `days: int` (default: 7)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.get_compliance_report`

生成合规报告

Args:
    report_type: 报告类型
    days: 报告天数

Returns:
    合规报告

**Signature:** `def get_compliance_report(self, report_type: str = "general", days: int = 30) -> ComplianceReport:`

**Parameters:**

- `self: Any` (required)
- `report_type: str` (default: 'general')
- `days: int` (default: 30)

**Returns:** `ComplianceReport`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.add_audit_rule`

添加审计规则

Args:
    rule: 审计规则

**Signature:** `def add_audit_rule(self, rule: AuditRule):`

**Parameters:**

- `self: Any` (required)
- `rule: AuditRule` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.remove_audit_rule`

移除审计规则

Args:
    rule_id: 规则ID

**Signature:** `def remove_audit_rule(self, rule_id: str):`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.enable_audit_rule`

启用审计规则

Args:
    rule_id: 规则ID

**Signature:** `def enable_audit_rule(self, rule_id: str):`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.disable_audit_rule`

禁用审计规则

Args:
    rule_id: 规则ID

**Signature:** `def disable_audit_rule(self, rule_id: str):`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.get_statistics`

获取统计信息

Returns:
    统计信息

**Signature:** `def get_statistics(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.cleanup_old_logs`

清理旧的日志文件

Args:
    days_to_keep: 保留天数

**Signature:** `def cleanup_old_logs(self, days_to_keep: int = 90):`

**Parameters:**

- `self: Any` (required)
- `days_to_keep: int` (default: 90)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `AuditLoggingManager.shutdown`

关闭审计日志管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `to_dict`

转换为字典

**Signature:** `def to_dict(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `from_dict`

从字典创建

**Signature:** `def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':`

**Parameters:**

- `cls: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `AuditEvent`

**Decorators:** `classmethod`

**Async:** No | **Visibility:** public

#### `matches_event`

检查事件是否匹配规则

**Signature:** `def matches_event(self, event: AuditEvent) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `should_trigger`

检查是否应该触发规则

**Signature:** `def should_trigger(self, event: AuditEvent) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `trigger`

触发规则

**Signature:** `def trigger(self, event: AuditEvent) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `log_event`

记录审计事件

Args:
    event_type: 事件类型
    severity: 严重程度
    user_id: 用户ID
    action: 操作
    result: 结果
    resource: 资源
    session_id: 会话ID
    details: 详细信息
    ip_address: IP地址
    user_agent: 用户代理
    location: 地理位置
    risk_score: 风险分数
    tags: 标签

Returns:
    事件ID

**Signature:** `def log_event(self, event_type: AuditEventType, severity: AuditSeverity,   user_id: Optional[str], action: str, result: str, resource: Optional[str] = None, session_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None, location: Optional[str] = None, risk_score: float = 0.0, tags: Optional[Set[str]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `severity: AuditSeverity` (required)
- `user_id: Optional[str]` (required)
- `action: str` (required)
- `result: str` (required)
- `resource: Optional[str]` (default: None)
- `session_id: Optional[str]` (default: None)
- `details: Optional[Dict[Any]]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)
- `location: Optional[str]` (default: None)
- `risk_score: float` (default: 0.0)
- `tags: Optional[Set[str]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `log_security_event`

记录安全事件

Args:
    user_id: 用户ID
    action: 操作
    result: 结果
    details: 详细信息
    ip_address: IP地址
    risk_score: 风险分数

Returns:
    事件ID

**Signature:** `def log_security_event(self, user_id: Optional[str], action: str, result: str,   details: Optional[Dict[str, Any]] = None, ip_address: Optional[str] = None, risk_score: float = 0.0) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (required)
- `action: str` (required)
- `result: str` (required)
- `details: Optional[Dict[Any]]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `risk_score: float` (default: 0.0)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `log_access_event`

记录访问事件

Args:
    user_id: 用户ID
    resource: 资源
    action: 操作
    result: 结果
    session_id: 会话ID
    ip_address: IP地址
    risk_score: 风险分数

Returns:
    事件ID

**Signature:** `def log_access_event(self, user_id: str, resource: str, action: str, result: str,   session_id: Optional[str] = None, ip_address: Optional[str] = None, risk_score: float = 0.0) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `resource: str` (required)
- `action: str` (required)
- `result: str` (required)
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `risk_score: float` (default: 0.0)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `log_data_operation`

记录数据操作事件

Args:
    user_id: 用户ID
    operation: 操作
    resource: 资源
    result: 结果
    details: 详细信息

Returns:
    事件ID

**Signature:** `def log_data_operation(self, user_id: str, operation: str, resource: str,   result: str, details: Optional[Dict[str, Any]] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `operation: str` (required)
- `resource: str` (required)
- `result: str` (required)
- `details: Optional[Dict[Any]]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `query_events`

查询审计事件

Args:
    start_time: 开始时间
    end_time: 结束时间
    event_type: 事件类型
    user_id: 用户ID
    resource: 资源
    result: 结果
    limit: 限制条数

Returns:
    审计事件列表

**Signature:** `def query_events(self, start_time: Optional[datetime] = None,   end_time: Optional[datetime] = None, event_type: Optional[AuditEventType] = None, user_id: Optional[str] = None, resource: Optional[str] = None, result: Optional[str] = None, limit: int = 100) -> List[AuditEvent]:`

**Parameters:**

- `self: Any` (required)
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)
- `event_type: Optional[AuditEventType]` (default: None)
- `user_id: Optional[str]` (default: None)
- `resource: Optional[str]` (default: None)
- `result: Optional[str]` (default: None)
- `limit: int` (default: 100)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `get_security_report`

获取安全报告

Args:
    days: 报告天数

Returns:
    安全报告

**Signature:** `def get_security_report(self, days: int = 7) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `days: int` (default: 7)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_compliance_report`

生成合规报告

Args:
    report_type: 报告类型
    days: 报告天数

Returns:
    合规报告

**Signature:** `def get_compliance_report(self, report_type: str = "general", days: int = 30) -> ComplianceReport:`

**Parameters:**

- `self: Any` (required)
- `report_type: str` (default: 'general')
- `days: int` (default: 30)

**Returns:** `ComplianceReport`

**Async:** No | **Visibility:** public

#### `add_audit_rule`

添加审计规则

Args:
    rule: 审计规则

**Signature:** `def add_audit_rule(self, rule: AuditRule):`

**Parameters:**

- `self: Any` (required)
- `rule: AuditRule` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `remove_audit_rule`

移除审计规则

Args:
    rule_id: 规则ID

**Signature:** `def remove_audit_rule(self, rule_id: str):`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `enable_audit_rule`

启用审计规则

Args:
    rule_id: 规则ID

**Signature:** `def enable_audit_rule(self, rule_id: str):`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `disable_audit_rule`

禁用审计规则

Args:
    rule_id: 规则ID

**Signature:** `def disable_audit_rule(self, rule_id: str):`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_statistics`

获取统计信息

Returns:
    统计信息

**Signature:** `def get_statistics(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `cleanup_old_logs`

清理旧的日志文件

Args:
    days_to_keep: 保留天数

**Signature:** `def cleanup_old_logs(self, days_to_keep: int = 90):`

**Parameters:**

- `self: Any` (required)
- `days_to_keep: int` (default: 90)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭审计日志管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

---

### audit.audit_manager

RQA2025 审计管理器

负责审计日志和合规报告
分离了AuditLoggingManager的审计职责

#### API Endpoints

#### `AuditManager.log_event`

记录审计事件

Args:
    params: 审计事件参数

Returns:
    事件ID

**Signature:** `def log_event(self, params: AuditEventParams) -> str:`

**Parameters:**

- `self: Any` (required)
- `params: AuditEventParams` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `AuditManager.query_events`

查询审计事件

Args:
    params: 查询过滤参数

Returns:
    匹配的事件列表

**Signature:** `def query_events(self, params: QueryFilterParams) -> List[Dict]:`

**Parameters:**

- `self: Any` (required)
- `params: QueryFilterParams` (required)

**Returns:** `List[Dict]`

**Async:** No | **Visibility:** public

#### `AuditManager.generate_security_report`

生成安全报告

Args:
    params: 报告生成参数

Returns:
    报告数据

**Signature:** `def generate_security_report(self, params: ReportGenerationParams) -> Dict:`

**Parameters:**

- `self: Any` (required)
- `params: ReportGenerationParams` (required)

**Returns:** `Dict`

**Async:** No | **Visibility:** public

#### `AuditManager.get_compliance_report`

生成合规报告

Args:
    compliance_type: 合规类型

Returns:
    合规报告数据

**Signature:** `def get_compliance_report(self, compliance_type: str = "general") -> Dict:`

**Parameters:**

- `self: Any` (required)
- `compliance_type: str` (default: 'general')

**Returns:** `Dict`

**Async:** No | **Visibility:** public

#### `log_event`

记录审计事件

Args:
    params: 审计事件参数

Returns:
    事件ID

**Signature:** `def log_event(self, params: AuditEventParams) -> str:`

**Parameters:**

- `self: Any` (required)
- `params: AuditEventParams` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `query_events`

查询审计事件

Args:
    params: 查询过滤参数

Returns:
    匹配的事件列表

**Signature:** `def query_events(self, params: QueryFilterParams) -> List[Dict]:`

**Parameters:**

- `self: Any` (required)
- `params: QueryFilterParams` (required)

**Returns:** `List[Dict]`

**Async:** No | **Visibility:** public

#### `generate_security_report`

生成安全报告

Args:
    params: 报告生成参数

Returns:
    报告数据

**Signature:** `def generate_security_report(self, params: ReportGenerationParams) -> Dict:`

**Parameters:**

- `self: Any` (required)
- `params: ReportGenerationParams` (required)

**Returns:** `Dict`

**Async:** No | **Visibility:** public

#### `get_compliance_report`

生成合规报告

Args:
    compliance_type: 合规类型

Returns:
    合规报告数据

**Signature:** `def get_compliance_report(self, compliance_type: str = "general") -> Dict:`

**Parameters:**

- `self: Any` (required)
- `compliance_type: str` (default: 'general')

**Returns:** `Dict`

**Async:** No | **Visibility:** public

---

### audit.audit_reporting

RQA2025 审计报告生成器

专门负责审计报告的生成、格式化和导出
从AuditLoggingManager中分离出来，提高代码组织性

#### API Endpoints

#### `generate_report`

生成审计报告

**Signature:** `def generate_report(self, events: List['AuditEvent'], report_type: str = "summary", start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, **kwargs) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `events: List[AuditEvent]` (required)
- `report_type: str` (default: 'summary')
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `generate_compliance_report`

生成合规报告

**Signature:** `def generate_compliance_report(self, events: List['AuditEvent'], report_type: str = "general", days: int = 30) -> ComplianceReport:`

**Parameters:**

- `self: Any` (required)
- `events: List[AuditEvent]` (required)
- `report_type: str` (default: 'general')
- `days: int` (default: 30)

**Returns:** `ComplianceReport`

**Async:** No | **Visibility:** public

#### `export_report`

导出报告

**Signature:** `def export_report(self, report_data: Dict[str, Any], format_type: str = "json", output_path: Optional[Path] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `report_data: Dict[Any]` (required)
- `format_type: str` (default: 'json')
- `output_path: Optional[Path]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

---

### audit.audit_rules

RQA2025 审计规则引擎

专门负责审计规则的定义、执行和触发逻辑
从AuditLoggingManager中分离出来，提高代码组织性

#### API Endpoints

#### `matches`

检查事件是否匹配条件

**Signature:** `def matches(self, event: 'AuditEvent') -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `matches`

检查事件是否匹配规则

**Signature:** `def matches(self, event: 'AuditEvent') -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `execute_actions`

执行规则动作

**Signature:** `def execute_actions(self, event: 'AuditEvent') -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `add_rule`

添加规则

**Signature:** `def add_rule(self, rule: AuditRule) -> None:`

**Parameters:**

- `self: Any` (required)
- `rule: AuditRule` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `remove_rule`

移除规则

**Signature:** `def remove_rule(self, rule_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `enable_rule`

启用规则

**Signature:** `def enable_rule(self, rule_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `disable_rule`

禁用规则

**Signature:** `def disable_rule(self, rule_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `evaluate_event`

评估事件并执行匹配的规则

**Signature:** `def evaluate_event(self, event: 'AuditEvent') -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_rule`

获取规则

**Signature:** `def get_rule(self, rule_id: str) -> Optional[AuditRule]:`

**Parameters:**

- `self: Any` (required)
- `rule_id: str` (required)

**Returns:** `Optional[AuditRule]`

**Async:** No | **Visibility:** public

#### `list_rules`

列出规则

**Signature:** `def list_rules(self, enabled_only: bool = False) -> List[AuditRule]:`

**Parameters:**

- `self: Any` (required)
- `enabled_only: bool` (default: False)

**Returns:** `List[AuditRule]`

**Async:** No | **Visibility:** public

#### `get_stats`

获取统计信息

**Signature:** `def get_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_rule_group`

创建规则组

**Signature:** `def create_rule_group(self, group_name: str, rule_ids: List[str]) -> None:`

**Parameters:**

- `self: Any` (required)
- `group_name: str` (required)
- `rule_ids: List[str]` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `evaluate_event_with_group`

使用规则组评估事件

**Signature:** `def evaluate_event_with_group(self, event: 'AuditEvent', group_name: str) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)
- `group_name: str` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `clear_stats`

清除统计信息

**Signature:** `def clear_stats(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `create_failed_login_rule`

创建失败登录规则

**Signature:** `def create_failed_login_rule() -> AuditRule:`

**Returns:** `AuditRule`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_high_risk_operation_rule`

创建高风险操作规则

**Signature:** `def create_high_risk_operation_rule() -> AuditRule:`

**Returns:** `AuditRule`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_suspicious_resource_access_rule`

创建可疑资源访问规则

**Signature:** `def create_suspicious_resource_access_rule() -> AuditRule:`

**Returns:** `AuditRule`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_compliance_violation_rule`

创建合规违规规则

**Signature:** `def create_compliance_violation_rule() -> AuditRule:`

**Returns:** `AuditRule`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

---

### audit.audit_storage

RQA2025 审计存储管理器

专门负责审计事件的存储、检索和持久化
从AuditLoggingManager中分离出来，提高代码组织性

#### API Endpoints

#### `AuditStorageManager.store_event`

存储审计事件

**Signature:** `def store_event(self, event: 'AuditEvent') -> None:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `AuditStorageManager.get_events`

检索审计事件

**Signature:** `def get_events(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, event_type: Optional[str] = None, limit: Optional[int] = None) -> List['AuditEvent']:`

**Parameters:**

- `self: Any` (required)
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)
- `event_type: Optional[str]` (default: None)
- `limit: Optional[int]` (default: None)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `AuditStorageManager.archive_old_events`

归档旧事件

**Signature:** `def archive_old_events(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `AuditStorageManager.get_storage_stats`

获取存储统计信息

**Signature:** `def get_storage_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `AuditStorageManager.cleanup_storage`

清理存储空间

**Signature:** `def cleanup_storage(self, days_to_keep: Optional[int] = None) -> int:`

**Parameters:**

- `self: Any` (required)
- `days_to_keep: Optional[int]` (default: None)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `store_event`

存储审计事件

**Signature:** `def store_event(self, event: 'AuditEvent') -> None:`

**Parameters:**

- `self: Any` (required)
- `event: AuditEvent` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_events`

检索审计事件

**Signature:** `def get_events(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, event_type: Optional[str] = None, limit: Optional[int] = None) -> List['AuditEvent']:`

**Parameters:**

- `self: Any` (required)
- `start_time: Optional[datetime]` (default: None)
- `end_time: Optional[datetime]` (default: None)
- `event_type: Optional[str]` (default: None)
- `limit: Optional[int]` (default: None)

**Returns:** `List[AuditEvent]`

**Async:** No | **Visibility:** public

#### `archive_old_events`

归档旧事件

**Signature:** `def archive_old_events(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_storage_stats`

获取存储统计信息

**Signature:** `def get_storage_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `cleanup_storage`

清理存储空间

**Signature:** `def cleanup_storage(self, days_to_keep: Optional[int] = None) -> int:`

**Parameters:**

- `self: Any` (required)
- `days_to_keep: Optional[int]` (default: None)

**Returns:** `int`

**Async:** No | **Visibility:** public

---

### audit.audit_system

RQA2025 安全审计系统

实现交易操作审计日志和安全事件记录

#### API Endpoints

#### `get_audit_system`

获取全局审计系统实例

**Signature:** `def get_audit_system(log_directory: str = "logs / audit") -> AuditSystem:`

**Parameters:**

- `log_directory: str` (default: 'logs / audit')

**Returns:** `AuditSystem`

**Async:** No | **Visibility:** public

#### `audit_trade_execution`

审计交易执行

**Signature:** `def audit_trade_execution(user_id: str, trade_details: Dict[str, Any], result: str = "success",   session_id: Optional[str] = None, ip_address: Optional[str] = None):`

**Parameters:**

- `user_id: str` (required)
- `trade_details: Dict[Any]` (required)
- `result: str` (default: 'success')
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `audit_order_operation`

审计订单操作

**Signature:** `def audit_order_operation(user_id: str, order_type: str, order_details: Dict[str, Any],   operation: str, result: str = "success", session_id: Optional[str] = None, ip_address: Optional[str] = None):`

**Parameters:**

- `user_id: str` (required)
- `order_type: str` (required)
- `order_details: Dict[Any]` (required)
- `operation: str` (required)
- `result: str` (default: 'success')
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `audit_security_event`

审计安全事件

**Signature:** `def audit_security_event(event_type: str, severity: SecurityLevel, source_ip: Optional[str] = None,   user_id: Optional[str] = None, description: str = "", details: Dict[str, Any] = None):`

**Parameters:**

- `event_type: str` (required)
- `severity: SecurityLevel` (required)
- `source_ip: Optional[str]` (default: None)
- `user_id: Optional[str]` (default: None)
- `description: str` (default: '')
- `details: Dict[Any]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `check_security_status`

检查安全状态

**Signature:** `def check_security_status(ip_address: str) -> Dict[str, Any]:`

**Parameters:**

- `ip_address: str` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `log_event`

记录审计事件

**Signature:** `def log_event(self, event_type: AuditEventType, user_id: Optional[str] = None,   session_id: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None, resource: str = "", action: str = "", result: str = "success", details: Dict[str, Any] = None, security_level: SecurityLevel = SecurityLevel.LOW):`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `user_id: Optional[str]` (default: None)
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)
- `resource: str` (default: '')
- `action: str` (default: '')
- `result: str` (default: 'success')
- `details: Dict[Any]` (default: None)
- `security_level: SecurityLevel` (default: ...)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `verify_event_integrity`

验证事件完整性

**Signature:** `def verify_event_integrity(self, event_dict: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event_dict: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_events_by_user`

获取指定用户的审计事件

**Signature:** `def get_events_by_user(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `days: int` (default: 7)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_events_by_type`

获取指定类型的审计事件

**Signature:** `def get_events_by_type(self, event_type: AuditEventType, days: int = 7) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `event_type: AuditEventType` (required)
- `days: int` (default: 7)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_high_risk_events`

获取高风险审计事件

**Signature:** `def get_high_risk_events(self, risk_threshold: float = 10.0, days: int = 7) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `risk_threshold: float` (default: 10.0)
- `days: int` (default: 7)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `record_security_event`

记录安全事件

**Signature:** `def record_security_event(self, event_type: str, severity: SecurityLevel,   source_ip: Optional[str] = None, user_id: Optional[str] = None, description: str = "", details: Dict[str, Any] = None):`

**Parameters:**

- `self: Any` (required)
- `event_type: str` (required)
- `severity: SecurityLevel` (required)
- `source_ip: Optional[str]` (default: None)
- `user_id: Optional[str]` (default: None)
- `description: str` (default: '')
- `details: Dict[Any]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `is_ip_blocked`

检查IP是否被封禁

**Signature:** `def is_ip_blocked(self, ip_address: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `ip_address: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_security_events`

获取安全事件

**Signature:** `def get_security_events(self, severity_filter: Optional[SecurityLevel] = None,   hours: int = 24) -> List[SecurityEvent]:`

**Parameters:**

- `self: Any` (required)
- `severity_filter: Optional[SecurityLevel]` (default: None)
- `hours: int` (default: 24)

**Returns:** `List[SecurityEvent]`

**Async:** No | **Visibility:** public

#### `resolve_security_event`

解决安全事件

**Signature:** `def resolve_security_event(self, event_id: str, notes: str = ""):`

**Parameters:**

- `self: Any` (required)
- `event_id: str` (required)
- `notes: str` (default: '')

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `log_trade_execution`

记录交易执行

**Signature:** `def log_trade_execution(self, user_id: str, trade_details: Dict[str, Any],   result: str = "success", session_id: Optional[str] = None, ip_address: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `trade_details: Dict[Any]` (required)
- `result: str` (default: 'success')
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `log_order_operation`

记录订单操作

**Signature:** `def log_order_operation(self, user_id: str, order_type: str, order_details: Dict[str, Any],   operation: str, result: str = "success", session_id: Optional[str] = None, ip_address: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `order_type: str` (required)
- `order_details: Dict[Any]` (required)
- `operation: str` (required)
- `result: str` (default: 'success')
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `log_security_event`

记录安全事件

**Signature:** `def log_security_event(self, event_type: str, severity: SecurityLevel,   source_ip: Optional[str] = None, user_id: Optional[str] = None, description: str = "", details: Dict[str, Any] = None):`

**Parameters:**

- `self: Any` (required)
- `event_type: str` (required)
- `severity: SecurityLevel` (required)
- `source_ip: Optional[str]` (default: None)
- `user_id: Optional[str]` (default: None)
- `description: str` (default: '')
- `details: Dict[Any]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `log_user_authentication`

记录用户认证

**Signature:** `def log_user_authentication(self, user_id: str, action: str, result: str = "success",   session_id: Optional[str] = None, ip_address: Optional[str] = None, user_agent: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `action: str` (required)
- `result: str` (default: 'success')
- `session_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `log_system_operation`

记录系统操作

**Signature:** `def log_system_operation(self, operation: str, details: Dict[str, Any] = None,   user_id: Optional[str] = None, ip_address: Optional[str] = None):`

**Parameters:**

- `self: Any` (required)
- `operation: str` (required)
- `details: Dict[Any]` (default: None)
- `user_id: Optional[str]` (default: None)
- `ip_address: Optional[str]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `check_security_status`

检查安全状态

**Signature:** `def check_security_status(self, ip_address: str) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `ip_address: str` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_audit_report`

生成审计报告

**Signature:** `def get_audit_report(self, user_id: Optional[str] = None, event_type: Optional[AuditEventType] = None,   days: int = 7) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `user_id: Optional[str]` (default: None)
- `event_type: Optional[AuditEventType]` (default: None)
- `days: int` (default: 7)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `health_check`

健康检查

**Signature:** `def health_check(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

---

### auth.authentication

#### API Endpoints

#### `MultiFactorAuthenticationService.register_authenticator`

注册认证器

**Signature:** `def register_authenticator(self, method: AuthMethod, authenticator: IAuthenticator):`

**Parameters:**

- `self: Any` (required)
- `method: AuthMethod` (required)
- `authenticator: IAuthenticator` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `MultiFactorAuthenticationService.create_user`

创建用户

**Signature:** `def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.VIEWER) -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `email: str` (required)
- `password: str` (required)
- `role: UserRole` (default: ...)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `MultiFactorAuthenticationService.authenticate_user`

认证用户

**Signature:** `def authenticate_user(self, username: str, credentials: Dict[str, Any], required_factors: List[AuthMethod] = None) -> AuthResult:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `credentials: Dict[Any]` (required)
- `required_factors: List[AuthMethod]` (default: None)

**Returns:** `AuthResult`

**Async:** No | **Visibility:** public

#### `MultiFactorAuthenticationService.verify_token`

验证JWT令牌

**Signature:** `def verify_token(self, token: str) -> Optional[User]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `Optional[User]`

**Async:** No | **Visibility:** public

#### `MultiFactorAuthenticationService.setup_mfa`

设置多因素认证

**Signature:** `def setup_mfa(self, user_id: str, method: AuthMethod, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `method: AuthMethod` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `MultiFactorAuthenticationService.get_totp_secret`

获取用户的TOTP密钥

**Signature:** `def get_totp_secret(self, user_id: str) -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `MultiFactorAuthenticationService.generate_current_totp`

生成用户当前的TOTP代码

**Signature:** `def generate_current_totp(self, user_id: str) -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `MultiFactorAuthenticationService.logout`

用户登出

**Signature:** `def logout(self, token: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `AuthorizationService.check_permission`

检查权限

**Signature:** `def check_permission(self, token: str, permission: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)
- `permission: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `AuthorizationService.get_user_permissions`

获取用户权限

**Signature:** `def get_user_permissions(self, token: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `authenticate`

执行认证

**Signature:** `def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:`

**Parameters:**

- `self: Any` (required)
- `credentials: Dict[Any]` (required)

**Returns:** `AuthResult`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `setup`

设置认证方法

**Signature:** `def setup(self, user_id: str, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `verify`

验证令牌

**Signature:** `def verify(self, user_id: str, token: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `token: str` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `authenticate`

执行密码认证

**Signature:** `def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:`

**Parameters:**

- `self: Any` (required)
- `credentials: Dict[Any]` (required)

**Returns:** `AuthResult`

**Async:** No | **Visibility:** public

#### `setup`

设置密码

**Signature:** `def setup(self, user_id: str, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `verify`

验证密码（不适用）

**Signature:** `def verify(self, user_id: str, token: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `token: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `authenticate`

执行TOTP认证

**Signature:** `def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:`

**Parameters:**

- `self: Any` (required)
- `credentials: Dict[Any]` (required)

**Returns:** `AuthResult`

**Async:** No | **Visibility:** public

#### `setup`

设置TOTP密钥

**Signature:** `def setup(self, user_id: str, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `verify`

验证TOTP令牌

**Signature:** `def verify(self, user_id: str, token: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `token: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `register_authenticator`

注册认证器

**Signature:** `def register_authenticator(self, method: AuthMethod, authenticator: IAuthenticator):`

**Parameters:**

- `self: Any` (required)
- `method: AuthMethod` (required)
- `authenticator: IAuthenticator` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_user`

创建用户

**Signature:** `def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.VIEWER) -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `email: str` (required)
- `password: str` (required)
- `role: UserRole` (default: ...)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `authenticate_user`

认证用户

**Signature:** `def authenticate_user(self, username: str, credentials: Dict[str, Any], required_factors: List[AuthMethod] = None) -> AuthResult:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `credentials: Dict[Any]` (required)
- `required_factors: List[AuthMethod]` (default: None)

**Returns:** `AuthResult`

**Async:** No | **Visibility:** public

#### `verify_token`

验证JWT令牌

**Signature:** `def verify_token(self, token: str) -> Optional[User]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `Optional[User]`

**Async:** No | **Visibility:** public

#### `setup_mfa`

设置多因素认证

**Signature:** `def setup_mfa(self, user_id: str, method: AuthMethod, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `method: AuthMethod` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_totp_secret`

获取用户的TOTP密钥

**Signature:** `def get_totp_secret(self, user_id: str) -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `generate_current_totp`

生成用户当前的TOTP代码

**Signature:** `def generate_current_totp(self, user_id: str) -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `logout`

用户登出

**Signature:** `def logout(self, token: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `check_permission`

检查权限

**Signature:** `def check_permission(self, token: str, permission: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)
- `permission: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_user_permissions`

获取用户权限

**Signature:** `def get_user_permissions(self, token: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

---

### auth.role_manager

RQA2025 角色管理器

专门负责角色的创建、管理和权限分配
从AccessControlManager中分离出来，提高代码组织性

#### API Endpoints

#### `RoleManager.create_role`

创建角色

**Signature:** `def create_role(self, role_id: str, name: str, description: str = "", permissions: Optional[Set[str]] = None, parent_roles: Optional[Set[str]] = None) -> Role:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)
- `name: str` (required)
- `description: str` (default: '')
- `permissions: Optional[Set[str]]` (default: None)
- `parent_roles: Optional[Set[str]]` (default: None)

**Returns:** `Role`

**Async:** No | **Visibility:** public

#### `RoleManager.update_role`

更新角色

**Signature:** `def update_role(self, role_id: str, **kwargs: Any) -> bool:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RoleManager.delete_role`

删除角色

**Signature:** `def delete_role(self, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RoleManager.get_role`

获取角色

**Signature:** `def get_role(self, role_id: str) -> Optional[Role]:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `Optional[Role]`

**Async:** No | **Visibility:** public

#### `RoleManager.list_roles`

列出角色

**Signature:** `def list_roles(self, active_only: bool = True) -> List[Role]:`

**Parameters:**

- `self: Any` (required)
- `active_only: bool` (default: True)

**Returns:** `List[Role]`

**Async:** No | **Visibility:** public

#### `RoleManager.assign_role_to_user`

为用户分配角色

**Signature:** `def assign_role_to_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RoleManager.revoke_role_from_user`

撤销用户的角色

**Signature:** `def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RoleManager.get_user_roles`

获取用户的角色

**Signature:** `def get_user_roles(self, user_id: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `RoleManager.get_role_permissions`

获取角色的所有权限

**Signature:** `def get_role_permissions(self, role_id: str) -> Set[str]:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `Set[str]`

**Async:** No | **Visibility:** public

#### `RoleManager.check_role_permission`

检查角色是否有指定权限

**Signature:** `def check_role_permission(self, role_id: str, permission: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)
- `permission: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `RoleManager.get_roles_with_permission`

获取拥有指定权限的所有角色

**Signature:** `def get_roles_with_permission(self, permission: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `permission: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `RoleManager.create_role_from_template`

从模板创建角色

**Signature:** `def create_role_from_template(self, role_enum: UserRole) -> Optional[Role]:`

**Parameters:**

- `self: Any` (required)
- `role_enum: UserRole` (required)

**Returns:** `Optional[Role]`

**Async:** No | **Visibility:** public

#### `RoleManager.get_role_hierarchy`

获取角色层次结构

**Signature:** `def get_role_hierarchy(self) -> Dict[str, List[str]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `RoleManager.validate_role_hierarchy`

验证角色层次结构的有效性

**Signature:** `def validate_role_hierarchy(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `RoleManager.get_role_stats`

获取角色统计信息

**Signature:** `def get_role_stats(self) -> Dict[str, any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `add_permission`

添加权限

**Signature:** `def add_permission(self, permission: str) -> None:`

**Parameters:**

- `self: Any` (required)
- `permission: str` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `remove_permission`

移除权限

**Signature:** `def remove_permission(self, permission: str) -> None:`

**Parameters:**

- `self: Any` (required)
- `permission: str` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `add_parent_role`

添加父角色

**Signature:** `def add_parent_role(self, role_id: str) -> None:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `remove_parent_role`

移除父角色

**Signature:** `def remove_parent_role(self, role_id: str) -> None:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_all_permissions`

获取所有权限（包括继承的权限）

**Signature:** `def get_all_permissions(self, all_roles: Dict[str, 'Role']) -> Set[str]:`

**Parameters:**

- `self: Any` (required)
- `all_roles: Dict[Any]` (required)

**Returns:** `Set[str]`

**Async:** No | **Visibility:** public

#### `create_role`

创建角色

**Signature:** `def create_role(self, role_id: str, name: str, description: str = "", permissions: Optional[Set[str]] = None, parent_roles: Optional[Set[str]] = None) -> Role:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)
- `name: str` (required)
- `description: str` (default: '')
- `permissions: Optional[Set[str]]` (default: None)
- `parent_roles: Optional[Set[str]]` (default: None)

**Returns:** `Role`

**Async:** No | **Visibility:** public

#### `update_role`

更新角色

**Signature:** `def update_role(self, role_id: str, **kwargs: Any) -> bool:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `delete_role`

删除角色

**Signature:** `def delete_role(self, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_role`

获取角色

**Signature:** `def get_role(self, role_id: str) -> Optional[Role]:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `Optional[Role]`

**Async:** No | **Visibility:** public

#### `list_roles`

列出角色

**Signature:** `def list_roles(self, active_only: bool = True) -> List[Role]:`

**Parameters:**

- `self: Any` (required)
- `active_only: bool` (default: True)

**Returns:** `List[Role]`

**Async:** No | **Visibility:** public

#### `assign_role_to_user`

为用户分配角色

**Signature:** `def assign_role_to_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `revoke_role_from_user`

撤销用户的角色

**Signature:** `def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_user_roles`

获取用户的角色

**Signature:** `def get_user_roles(self, user_id: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `get_role_permissions`

获取角色的所有权限

**Signature:** `def get_role_permissions(self, role_id: str) -> Set[str]:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)

**Returns:** `Set[str]`

**Async:** No | **Visibility:** public

#### `check_role_permission`

检查角色是否有指定权限

**Signature:** `def check_role_permission(self, role_id: str, permission: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `role_id: str` (required)
- `permission: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_roles_with_permission`

获取拥有指定权限的所有角色

**Signature:** `def get_roles_with_permission(self, permission: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `permission: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `create_role_from_template`

从模板创建角色

**Signature:** `def create_role_from_template(self, role_enum: UserRole) -> Optional[Role]:`

**Parameters:**

- `self: Any` (required)
- `role_enum: UserRole` (required)

**Returns:** `Optional[Role]`

**Async:** No | **Visibility:** public

#### `get_role_hierarchy`

获取角色层次结构

**Signature:** `def get_role_hierarchy(self) -> Dict[str, List[str]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `validate_role_hierarchy`

验证角色层次结构的有效性

**Signature:** `def validate_role_hierarchy(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `get_role_stats`

获取角色统计信息

**Signature:** `def get_role_stats(self) -> Dict[str, any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `check_cycle`

**Signature:** `def check_cycle(role_id: str) -> bool:`

**Parameters:**

- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

---

### auth.session_manager

RQA2025 会话管理器

专门负责用户会话的创建、验证和管理
从AccessControlManager中分离出来，提高代码组织性

#### API Endpoints

#### `SessionManager.create_session`

创建新会话

**Signature:** `def create_session(self, user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None, **metadata) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `SessionManager.get_session`

获取会话

**Signature:** `def get_session(self, session_id: str) -> Optional[UserSession]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[UserSession]`

**Async:** No | **Visibility:** public

#### `SessionManager.validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `SessionManager.extend_session`

延长会话

**Signature:** `def extend_session(self, session_id: str, minutes: int = 60) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)
- `minutes: int` (default: 60)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `SessionManager.invalidate_session`

使会话失效

**Signature:** `def invalidate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `SessionManager.invalidate_user_sessions`

使指定用户的所有会话失效

**Signature:** `def invalidate_user_sessions(self, user_id: str) -> int:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `SessionManager.get_user_sessions`

获取用户的所有活动会话

**Signature:** `def get_user_sessions(self, user_id: str) -> List[UserSession]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `List[UserSession]`

**Async:** No | **Visibility:** public

#### `SessionManager.cleanup_expired_sessions`

清理所有过期会话

**Signature:** `def cleanup_expired_sessions(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `SessionManager.get_session_stats`

获取会话统计信息

**Signature:** `def get_session_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `SessionManager.get_session_info`

获取会话详细信息

**Signature:** `def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `SessionManager.update_session_metadata`

更新会话元数据

**Signature:** `def update_session_metadata(self, session_id: str, **metadata) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `is_expired`

检查会话是否过期

**Signature:** `def is_expired(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `extend_session`

延长会话时间

**Signature:** `def extend_session(self, minutes: int = 60) -> None:`

**Parameters:**

- `self: Any` (required)
- `minutes: int` (default: 60)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_remaining_time`

获取剩余时间

**Signature:** `def get_remaining_time(self) -> timedelta:`

**Parameters:**

- `self: Any` (required)

**Returns:** `timedelta`

**Async:** No | **Visibility:** public

#### `to_dict`

转换为字典

**Signature:** `def to_dict(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_session`

创建新会话

**Signature:** `def create_session(self, user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None, **metadata) -> str:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `get_session`

获取会话

**Signature:** `def get_session(self, session_id: str) -> Optional[UserSession]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[UserSession]`

**Async:** No | **Visibility:** public

#### `validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)
- `ip_address: Optional[str]` (default: None)
- `user_agent: Optional[str]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `extend_session`

延长会话

**Signature:** `def extend_session(self, session_id: str, minutes: int = 60) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)
- `minutes: int` (default: 60)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `invalidate_session`

使会话失效

**Signature:** `def invalidate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `invalidate_user_sessions`

使指定用户的所有会话失效

**Signature:** `def invalidate_user_sessions(self, user_id: str) -> int:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_user_sessions`

获取用户的所有活动会话

**Signature:** `def get_user_sessions(self, user_id: str) -> List[UserSession]:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `List[UserSession]`

**Async:** No | **Visibility:** public

#### `cleanup_expired_sessions`

清理所有过期会话

**Signature:** `def cleanup_expired_sessions(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_session_stats`

获取会话统计信息

**Signature:** `def get_session_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_session_info`

获取会话详细信息

**Signature:** `def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `update_session_metadata`

更新会话元数据

**Signature:** `def update_session_metadata(self, session_id: str, **metadata) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

---

### auth.user_manager

RQA2025 用户管理器

负责用户和角色的管理
分离了AccessControlManager的用户管理职责

#### API Endpoints

#### `UserManager.create_user`

创建用户

Args:
    params: 用户创建参数

Returns:
    创建的用户对象

**Signature:** `def create_user(self, params: UserCreationParams) -> 'User':`

**Parameters:**

- `self: Any` (required)
- `params: UserCreationParams` (required)

**Returns:** `User`

**Async:** No | **Visibility:** public

#### `UserManager.get_user`

获取用户

**Signature:** `def get_user(self, user_id: str) -> Optional['User']:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[User]`

**Async:** No | **Visibility:** public

#### `UserManager.update_user`

更新用户信息

**Signature:** `def update_user(self, user_id: str, **kwargs) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `UserManager.delete_user`

删除用户

**Signature:** `def delete_user(self, user_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `UserManager.list_users`

列出用户

**Signature:** `def list_users(self, active_only: bool = True) -> List['User']:`

**Parameters:**

- `self: Any` (required)
- `active_only: bool` (default: True)

**Returns:** `List[User]`

**Async:** No | **Visibility:** public

#### `UserManager.create_role`

创建角色

**Signature:** `def create_role(self, name: str, permissions: Set[str], description: str = "") -> 'Role':`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `permissions: Set[str]` (required)
- `description: str` (default: '')

**Returns:** `Role`

**Async:** No | **Visibility:** public

#### `UserManager.assign_role_to_user`

为用户分配角色

**Signature:** `def assign_role_to_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `UserManager.revoke_role_from_user`

撤销用户角色

**Signature:** `def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PermissionManager.check_permission`

检查权限

Args:
    params: 访问检查参数

Returns:
    是否有权限

**Signature:** `def check_permission(self, params: AccessCheckParams) -> bool:`

**Parameters:**

- `self: Any` (required)
- `params: AccessCheckParams` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_user`

创建用户

Args:
    params: 用户创建参数

Returns:
    创建的用户对象

**Signature:** `def create_user(self, params: UserCreationParams) -> 'User':`

**Parameters:**

- `self: Any` (required)
- `params: UserCreationParams` (required)

**Returns:** `User`

**Async:** No | **Visibility:** public

#### `get_user`

获取用户

**Signature:** `def get_user(self, user_id: str) -> Optional['User']:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `Optional[User]`

**Async:** No | **Visibility:** public

#### `update_user`

更新用户信息

**Signature:** `def update_user(self, user_id: str, **kwargs) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `delete_user`

删除用户

**Signature:** `def delete_user(self, user_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `list_users`

列出用户

**Signature:** `def list_users(self, active_only: bool = True) -> List['User']:`

**Parameters:**

- `self: Any` (required)
- `active_only: bool` (default: True)

**Returns:** `List[User]`

**Async:** No | **Visibility:** public

#### `create_role`

创建角色

**Signature:** `def create_role(self, name: str, permissions: Set[str], description: str = "") -> 'Role':`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)
- `permissions: Set[str]` (required)
- `description: str` (default: '')

**Returns:** `Role`

**Async:** No | **Visibility:** public

#### `assign_role_to_user`

为用户分配角色

**Signature:** `def assign_role_to_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `revoke_role_from_user`

撤销用户角色

**Signature:** `def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `user_id: str` (required)
- `role_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `check_permission`

检查权限

Args:
    params: 访问检查参数

Returns:
    是否有权限

**Signature:** `def check_permission(self, params: AccessCheckParams) -> bool:`

**Parameters:**

- `self: Any` (required)
- `params: AccessCheckParams` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

---

### components.audit_component

#### API Endpoints

#### `create_component`

创建组件

**Signature:** `def create_component(self, component_type: str, config: Dict[str, Any]):`

**Parameters:**

- `self: Any` (required)
- `component_type: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_audit_id`

获取audit ID

**Signature:** `def get_audit_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_audit_id`

获取audit ID

**Signature:** `def get_audit_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_component`

创建指定ID的audit组件

**Signature:** `def create_component(audit_id: int) -> AuditComponent:`

**Parameters:**

- `audit_id: int` (required)

**Returns:** `AuditComponent`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_available_audits`

获取所有可用的audit ID

**Signature:** `def get_available_audits() -> List[int]:`

**Returns:** `List[int]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_all_audits`

创建所有可用audit

**Signature:** `def create_all_audits() -> Dict[int, AuditComponent]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_factory_info`

获取工厂信息

**Signature:** `def get_factory_info() -> Dict[str, Any]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

---

### components.auth_component

#### API Endpoints

#### `create_auth_auth_component_2`

**Signature:** `def create_auth_auth_component_2(): return AuthComponentFactory.create_component(2)   def create_auth_auth_component_8(): return AuthComponentFactory.create_component(8)   def create_auth_auth_component_14(): return AuthComponentFactory.create_component(14)   def create_auth_auth_component_20(): return AuthComponentFactory.create_component(20)   def create_auth_auth_component_26(): return AuthComponentFactory.create_component(26)   def create_auth_auth_component_32(): return AuthComponentFactory.create_component(32)   def create_auth_auth_component_38(): return AuthComponentFactory.create_component(38)   def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_8`

**Signature:** `def create_auth_auth_component_8(): return AuthComponentFactory.create_component(8)   def create_auth_auth_component_14(): return AuthComponentFactory.create_component(14)   def create_auth_auth_component_20(): return AuthComponentFactory.create_component(20)   def create_auth_auth_component_26(): return AuthComponentFactory.create_component(26)   def create_auth_auth_component_32(): return AuthComponentFactory.create_component(32)   def create_auth_auth_component_38(): return AuthComponentFactory.create_component(38)   def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_14`

**Signature:** `def create_auth_auth_component_14(): return AuthComponentFactory.create_component(14)   def create_auth_auth_component_20(): return AuthComponentFactory.create_component(20)   def create_auth_auth_component_26(): return AuthComponentFactory.create_component(26)   def create_auth_auth_component_32(): return AuthComponentFactory.create_component(32)   def create_auth_auth_component_38(): return AuthComponentFactory.create_component(38)   def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_20`

**Signature:** `def create_auth_auth_component_20(): return AuthComponentFactory.create_component(20)   def create_auth_auth_component_26(): return AuthComponentFactory.create_component(26)   def create_auth_auth_component_32(): return AuthComponentFactory.create_component(32)   def create_auth_auth_component_38(): return AuthComponentFactory.create_component(38)   def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_26`

**Signature:** `def create_auth_auth_component_26(): return AuthComponentFactory.create_component(26)   def create_auth_auth_component_32(): return AuthComponentFactory.create_component(32)   def create_auth_auth_component_38(): return AuthComponentFactory.create_component(38)   def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_32`

**Signature:** `def create_auth_auth_component_32(): return AuthComponentFactory.create_component(32)   def create_auth_auth_component_38(): return AuthComponentFactory.create_component(38)   def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_38`

**Signature:** `def create_auth_auth_component_38(): return AuthComponentFactory.create_component(38)   def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_44`

**Signature:** `def create_auth_auth_component_44(): return AuthComponentFactory.create_component(44)   def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_50`

**Signature:** `def create_auth_auth_component_50(): return AuthComponentFactory.create_component(50)   def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_auth_auth_component_56`

**Signature:** `def create_auth_auth_component_56(): return AuthComponentFactory.create_component(56)   __all__ = [ "IAuthComponent", "AuthComponent", "AuthComponentFactory", "create_auth_auth_component_2", "create_auth_auth_component_8", "create_auth_auth_component_14", "create_auth_auth_component_20", "create_auth_auth_component_26", "create_auth_auth_component_32", "create_auth_auth_component_38", "create_auth_auth_component_44", "create_auth_auth_component_50", "create_auth_auth_component_56", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_component`

创建组件

**Signature:** `def create_component(self, component_type: str, config: Dict[str, Any]):`

**Parameters:**

- `self: Any` (required)
- `component_type: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_auth_id`

获取auth ID

**Signature:** `def get_auth_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_auth_id`

获取auth ID

**Signature:** `def get_auth_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_component`

创建指定ID的auth组件

**Signature:** `def create_component(auth_id: int) -> AuthComponent:`

**Parameters:**

- `auth_id: int` (required)

**Returns:** `AuthComponent`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_available_auths`

获取所有可用的auth ID

**Signature:** `def get_available_auths() -> List[int]:`

**Returns:** `List[int]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_all_auths`

创建所有可用auth

**Signature:** `def create_all_auths() -> Dict[int, AuthComponent]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_factory_info`

获取工厂信息

**Signature:** `def get_factory_info() -> Dict[str, Any]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

---

### components.encrypt_component

#### API Endpoints

#### `create_encrypt_encrypt_component_3`

**Signature:** `def create_encrypt_encrypt_component_3(): return EncryptComponentFactory.create_component(3)   def create_encrypt_encrypt_component_9(): return EncryptComponentFactory.create_component(9)   def create_encrypt_encrypt_component_15(): return EncryptComponentFactory.create_component(15)   def create_encrypt_encrypt_component_21(): return EncryptComponentFactory.create_component(21)   def create_encrypt_encrypt_component_27(): return EncryptComponentFactory.create_component(27)   def create_encrypt_encrypt_component_33(): return EncryptComponentFactory.create_component(33)   def create_encrypt_encrypt_component_39(): return EncryptComponentFactory.create_component(39)   def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_9`

**Signature:** `def create_encrypt_encrypt_component_9(): return EncryptComponentFactory.create_component(9)   def create_encrypt_encrypt_component_15(): return EncryptComponentFactory.create_component(15)   def create_encrypt_encrypt_component_21(): return EncryptComponentFactory.create_component(21)   def create_encrypt_encrypt_component_27(): return EncryptComponentFactory.create_component(27)   def create_encrypt_encrypt_component_33(): return EncryptComponentFactory.create_component(33)   def create_encrypt_encrypt_component_39(): return EncryptComponentFactory.create_component(39)   def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_15`

**Signature:** `def create_encrypt_encrypt_component_15(): return EncryptComponentFactory.create_component(15)   def create_encrypt_encrypt_component_21(): return EncryptComponentFactory.create_component(21)   def create_encrypt_encrypt_component_27(): return EncryptComponentFactory.create_component(27)   def create_encrypt_encrypt_component_33(): return EncryptComponentFactory.create_component(33)   def create_encrypt_encrypt_component_39(): return EncryptComponentFactory.create_component(39)   def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_21`

**Signature:** `def create_encrypt_encrypt_component_21(): return EncryptComponentFactory.create_component(21)   def create_encrypt_encrypt_component_27(): return EncryptComponentFactory.create_component(27)   def create_encrypt_encrypt_component_33(): return EncryptComponentFactory.create_component(33)   def create_encrypt_encrypt_component_39(): return EncryptComponentFactory.create_component(39)   def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_27`

**Signature:** `def create_encrypt_encrypt_component_27(): return EncryptComponentFactory.create_component(27)   def create_encrypt_encrypt_component_33(): return EncryptComponentFactory.create_component(33)   def create_encrypt_encrypt_component_39(): return EncryptComponentFactory.create_component(39)   def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_33`

**Signature:** `def create_encrypt_encrypt_component_33(): return EncryptComponentFactory.create_component(33)   def create_encrypt_encrypt_component_39(): return EncryptComponentFactory.create_component(39)   def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_39`

**Signature:** `def create_encrypt_encrypt_component_39(): return EncryptComponentFactory.create_component(39)   def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_45`

**Signature:** `def create_encrypt_encrypt_component_45(): return EncryptComponentFactory.create_component(45)   def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_51`

**Signature:** `def create_encrypt_encrypt_component_51(): return EncryptComponentFactory.create_component(51)   def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_encrypt_encrypt_component_57`

**Signature:** `def create_encrypt_encrypt_component_57(): return EncryptComponentFactory.create_component(57)   __all__ = [ "IEncryptComponent", "EncryptComponent", "EncryptComponentFactory", "create_encrypt_encrypt_component_3", "create_encrypt_encrypt_component_9", "create_encrypt_encrypt_component_15", "create_encrypt_encrypt_component_21", "create_encrypt_encrypt_component_27", "create_encrypt_encrypt_component_33", "create_encrypt_encrypt_component_39", "create_encrypt_encrypt_component_45", "create_encrypt_encrypt_component_51", "create_encrypt_encrypt_component_57", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_component`

创建组件

**Signature:** `def create_component(self, component_type: str, config: Dict[str, Any]):`

**Parameters:**

- `self: Any` (required)
- `component_type: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_encrypt_id`

获取encrypt ID

**Signature:** `def get_encrypt_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_encrypt_id`

获取encrypt ID

**Signature:** `def get_encrypt_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_component`

创建指定ID的encrypt组件

**Signature:** `def create_component(encrypt_id: int) -> EncryptComponent:`

**Parameters:**

- `encrypt_id: int` (required)

**Returns:** `EncryptComponent`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_available_encrypts`

获取所有可用的encrypt ID

**Signature:** `def get_available_encrypts() -> List[int]:`

**Returns:** `List[int]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_all_encrypts`

创建所有可用encrypt

**Signature:** `def create_all_encrypts() -> Dict[int, EncryptComponent]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_factory_info`

获取工厂信息

**Signature:** `def get_factory_info() -> Dict[str, Any]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

---

### components.policy_component

#### API Endpoints

#### `create_policy_policy_component_5`

**Signature:** `def create_policy_policy_component_5(): return PolicyComponentFactory.create_component(5)   def create_policy_policy_component_11(): return PolicyComponentFactory.create_component(11)   def create_policy_policy_component_17(): return PolicyComponentFactory.create_component(17)   def create_policy_policy_component_23(): return PolicyComponentFactory.create_component(23)   def create_policy_policy_component_29(): return PolicyComponentFactory.create_component(29)   def create_policy_policy_component_35(): return PolicyComponentFactory.create_component(35)   def create_policy_policy_component_41(): return PolicyComponentFactory.create_component(41)   def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_11`

**Signature:** `def create_policy_policy_component_11(): return PolicyComponentFactory.create_component(11)   def create_policy_policy_component_17(): return PolicyComponentFactory.create_component(17)   def create_policy_policy_component_23(): return PolicyComponentFactory.create_component(23)   def create_policy_policy_component_29(): return PolicyComponentFactory.create_component(29)   def create_policy_policy_component_35(): return PolicyComponentFactory.create_component(35)   def create_policy_policy_component_41(): return PolicyComponentFactory.create_component(41)   def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_17`

**Signature:** `def create_policy_policy_component_17(): return PolicyComponentFactory.create_component(17)   def create_policy_policy_component_23(): return PolicyComponentFactory.create_component(23)   def create_policy_policy_component_29(): return PolicyComponentFactory.create_component(29)   def create_policy_policy_component_35(): return PolicyComponentFactory.create_component(35)   def create_policy_policy_component_41(): return PolicyComponentFactory.create_component(41)   def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_23`

**Signature:** `def create_policy_policy_component_23(): return PolicyComponentFactory.create_component(23)   def create_policy_policy_component_29(): return PolicyComponentFactory.create_component(29)   def create_policy_policy_component_35(): return PolicyComponentFactory.create_component(35)   def create_policy_policy_component_41(): return PolicyComponentFactory.create_component(41)   def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_29`

**Signature:** `def create_policy_policy_component_29(): return PolicyComponentFactory.create_component(29)   def create_policy_policy_component_35(): return PolicyComponentFactory.create_component(35)   def create_policy_policy_component_41(): return PolicyComponentFactory.create_component(41)   def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_35`

**Signature:** `def create_policy_policy_component_35(): return PolicyComponentFactory.create_component(35)   def create_policy_policy_component_41(): return PolicyComponentFactory.create_component(41)   def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_41`

**Signature:** `def create_policy_policy_component_41(): return PolicyComponentFactory.create_component(41)   def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_47`

**Signature:** `def create_policy_policy_component_47(): return PolicyComponentFactory.create_component(47)   def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_53`

**Signature:** `def create_policy_policy_component_53(): return PolicyComponentFactory.create_component(53)   def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_policy_policy_component_59`

**Signature:** `def create_policy_policy_component_59(): return PolicyComponentFactory.create_component(59)   __all__ = [ "IPolicyComponent", "PolicyComponent", "PolicyComponentFactory", "create_policy_policy_component_5", "create_policy_policy_component_11", "create_policy_policy_component_17", "create_policy_policy_component_23", "create_policy_policy_component_29", "create_policy_policy_component_35", "create_policy_policy_component_41", "create_policy_policy_component_47", "create_policy_policy_component_53", "create_policy_policy_component_59", ] `

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `create_component`

创建组件

**Signature:** `def create_component(self, component_type: str, config: Dict[str, Any]):`

**Parameters:**

- `self: Any` (required)
- `component_type: str` (required)
- `config: Dict[Any]` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_policy_id`

获取policy ID

**Signature:** `def get_policy_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_policy_id`

获取policy ID

**Signature:** `def get_policy_id(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_info`

获取组件信息

**Signature:** `def get_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `process`

处理数据

**Signature:** `def process(self, data: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_status`

获取组件状态

**Signature:** `def get_status(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_component`

创建指定ID的policy组件

**Signature:** `def create_component(policy_id: int) -> PolicyComponent:`

**Parameters:**

- `policy_id: int` (required)

**Returns:** `PolicyComponent`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_available_policys`

获取所有可用的policy ID

**Signature:** `def get_available_policys() -> List[int]:`

**Returns:** `List[int]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_all_policys`

创建所有可用policy

**Signature:** `def create_all_policys() -> Dict[int, PolicyComponent]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `get_factory_info`

获取工厂信息

**Signature:** `def get_factory_info() -> Dict[str, Any]:`

**Returns:** `Dict[Any]`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

---

### crypto.algorithms

RQA2025 加密算法组件

提供各种加密算法的具体实现
职责单一，便于测试和维护

#### API Endpoints

#### `increment_usage`

增加使用次数

**Signature:** `def increment_usage(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `encrypt`

加密数据

**Signature:** `def encrypt(self, data: bytes, key: EncryptionKey, params: EncryptionParams) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: bytes` (required)
- `key: EncryptionKey` (required)
- `params: EncryptionParams` (required)

**Returns:** `EncryptionResult`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `decrypt`

解密数据

**Signature:** `def decrypt(self, result: EncryptionResult, key: EncryptionKey) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `result: EncryptionResult` (required)
- `key: EncryptionKey` (required)

**Returns:** `bytes`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `generate_key`

生成密钥

**Signature:** `def generate_key(self, key_size: int) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `key_size: int` (required)

**Returns:** `bytes`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `encrypt`

AES-GCM加密

**Signature:** `def encrypt(self, data: bytes, key: EncryptionKey, params: EncryptionParams) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: bytes` (required)
- `key: EncryptionKey` (required)
- `params: EncryptionParams` (required)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `decrypt`

AES-GCM解密

**Signature:** `def decrypt(self, result: EncryptionResult, key: EncryptionKey) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `result: EncryptionResult` (required)
- `key: EncryptionKey` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `generate_key`

生成AES密钥

**Signature:** `def generate_key(self, key_size: int) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `key_size: int` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `encrypt`

AES-CBC加密

**Signature:** `def encrypt(self, data: bytes, key: EncryptionKey, params: EncryptionParams) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: bytes` (required)
- `key: EncryptionKey` (required)
- `params: EncryptionParams` (required)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `decrypt`

AES-CBC解密

**Signature:** `def decrypt(self, result: EncryptionResult, key: EncryptionKey) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `result: EncryptionResult` (required)
- `key: EncryptionKey` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `generate_key`

生成AES密钥

**Signature:** `def generate_key(self, key_size: int) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `key_size: int` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `encrypt`

RSA-OAEP加密

**Signature:** `def encrypt(self, data: bytes, key: EncryptionKey, params: EncryptionParams) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: bytes` (required)
- `key: EncryptionKey` (required)
- `params: EncryptionParams` (required)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `decrypt`

RSA-OAEP解密

**Signature:** `def decrypt(self, result: EncryptionResult, key: EncryptionKey) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `result: EncryptionResult` (required)
- `key: EncryptionKey` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `generate_key`

生成RSA密钥对

**Signature:** `def generate_key(self, key_size: int) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `key_size: int` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `encrypt`

ChaCha20加密

**Signature:** `def encrypt(self, data: bytes, key: EncryptionKey, params: EncryptionParams) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: bytes` (required)
- `key: EncryptionKey` (required)
- `params: EncryptionParams` (required)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `decrypt`

ChaCha20解密

**Signature:** `def decrypt(self, result: EncryptionResult, key: EncryptionKey) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `result: EncryptionResult` (required)
- `key: EncryptionKey` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `generate_key`

生成ChaCha20密钥

**Signature:** `def generate_key(self, key_size: int) -> bytes:`

**Parameters:**

- `self: Any` (required)
- `key_size: int` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

---

### crypto.encryption

RQA2025 数据加密管理器

实现端到端的数据加密和解密功能
支持多种加密算法和密钥管理

#### API Endpoints

#### `DataEncryptionManager.encrypt_data`

加密数据

Args:
    data: 要加密的数据
    algorithm: 加密算法
    key_id: 密钥ID，如果为None则使用当前活动密钥
    metadata: 元数据

Returns:
    加密结果

**Signature:** `def encrypt_data(self, data: Union[str, bytes], algorithm: str = "AES - 256 - GCM",   key_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: Union[Any]` (required)
- `algorithm: str` (default: 'AES - 256 - GCM')
- `key_id: Optional[str]` (default: None)
- `metadata: Optional[Dict[Any]]` (default: None)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.decrypt_data`

解密数据

Args:
    encrypted_result: 加密结果

Returns:
    解密结果

**Signature:** `def decrypt_data(self, encrypted_result: EncryptionResult) -> DecryptionResult:`

**Parameters:**

- `self: Any` (required)
- `encrypted_result: EncryptionResult` (required)

**Returns:** `DecryptionResult`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.generate_key`

生成新密钥

Args:
    algorithm: 密钥算法
    expires_in_days: 过期天数

Returns:
    密钥ID

**Signature:** `def generate_key(self, algorithm: str = "AES - 256", expires_in_days: Optional[int] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `algorithm: str` (default: 'AES - 256')
- `expires_in_days: Optional[int]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.rotate_keys`

轮换密钥

Returns:
    新生成的密钥ID列表

**Signature:** `def rotate_keys(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.get_audit_logs`

获取审计日志

Args:
    limit: 限制条数

Returns:
    审计日志列表

**Signature:** `def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `limit: int` (default: 100)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.get_encryption_stats`

获取加密统计信息

Returns:
    统计信息

**Signature:** `def get_encryption_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.encrypt_batch`

批量加密数据

Args:
    data_list: 数据列表 [{'data': bytes, 'metadata': dict}]
    algorithm: 加密算法

Returns:
    加密结果列表

**Signature:** `def encrypt_batch(self, data_list: List[Dict[str, Any]],   algorithm: str = "AES - 256 - GCM") -> List[EncryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `data_list: List[Dict[Any]]` (required)
- `algorithm: str` (default: 'AES - 256 - GCM')

**Returns:** `List[EncryptionResult]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.decrypt_batch`

批量解密数据

Args:
    encrypted_results: 加密结果列表

Returns:
    解密结果列表

**Signature:** `def decrypt_batch(self, encrypted_results: List[EncryptionResult]) -> List[DecryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_results: List[EncryptionResult]` (required)

**Returns:** `List[DecryptionResult]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.cleanup_expired_keys`

清理过期密钥

Returns:
    清理的密钥数量

**Signature:** `def cleanup_expired_keys(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.export_keys`

导出密钥

Args:
    export_path: 导出路径
    include_private: 是否包含私钥信息

**Signature:** `def export_keys(self, export_path: str, include_private: bool = False):`

**Parameters:**

- `self: Any` (required)
- `export_path: str` (required)
- `include_private: bool` (default: False)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.shutdown`

关闭加密管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `is_expired`

检查密钥是否过期

**Signature:** `def is_expired(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `can_use`

检查密钥是否可以使用

**Signature:** `def can_use(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `increment_usage`

增加使用计数

**Signature:** `def increment_usage(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `encrypt_data`

加密数据

Args:
    data: 要加密的数据
    algorithm: 加密算法
    key_id: 密钥ID，如果为None则使用当前活动密钥
    metadata: 元数据

Returns:
    加密结果

**Signature:** `def encrypt_data(self, data: Union[str, bytes], algorithm: str = "AES - 256 - GCM",   key_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: Union[Any]` (required)
- `algorithm: str` (default: 'AES - 256 - GCM')
- `key_id: Optional[str]` (default: None)
- `metadata: Optional[Dict[Any]]` (default: None)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `decrypt_data`

解密数据

Args:
    encrypted_result: 加密结果

Returns:
    解密结果

**Signature:** `def decrypt_data(self, encrypted_result: EncryptionResult) -> DecryptionResult:`

**Parameters:**

- `self: Any` (required)
- `encrypted_result: EncryptionResult` (required)

**Returns:** `DecryptionResult`

**Async:** No | **Visibility:** public

#### `generate_key`

生成新密钥

Args:
    algorithm: 密钥算法
    expires_in_days: 过期天数

Returns:
    密钥ID

**Signature:** `def generate_key(self, algorithm: str = "AES - 256", expires_in_days: Optional[int] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `algorithm: str` (default: 'AES - 256')
- `expires_in_days: Optional[int]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `rotate_keys`

轮换密钥

Returns:
    新生成的密钥ID列表

**Signature:** `def rotate_keys(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `get_audit_logs`

获取审计日志

Args:
    limit: 限制条数

Returns:
    审计日志列表

**Signature:** `def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `limit: int` (default: 100)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_encryption_stats`

获取加密统计信息

Returns:
    统计信息

**Signature:** `def get_encryption_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `encrypt_batch`

批量加密数据

Args:
    data_list: 数据列表 [{'data': bytes, 'metadata': dict}]
    algorithm: 加密算法

Returns:
    加密结果列表

**Signature:** `def encrypt_batch(self, data_list: List[Dict[str, Any]],   algorithm: str = "AES - 256 - GCM") -> List[EncryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `data_list: List[Dict[Any]]` (required)
- `algorithm: str` (default: 'AES - 256 - GCM')

**Returns:** `List[EncryptionResult]`

**Async:** No | **Visibility:** public

#### `decrypt_batch`

批量解密数据

Args:
    encrypted_results: 加密结果列表

Returns:
    解密结果列表

**Signature:** `def decrypt_batch(self, encrypted_results: List[EncryptionResult]) -> List[DecryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_results: List[EncryptionResult]` (required)

**Returns:** `List[DecryptionResult]`

**Async:** No | **Visibility:** public

#### `cleanup_expired_keys`

清理过期密钥

Returns:
    清理的密钥数量

**Signature:** `def cleanup_expired_keys(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `export_keys`

导出密钥

Args:
    export_path: 导出路径
    include_private: 是否包含私钥信息

**Signature:** `def export_keys(self, export_path: str, include_private: bool = False):`

**Parameters:**

- `self: Any` (required)
- `export_path: str` (required)
- `include_private: bool` (default: False)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭加密管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `simple_xor_encrypt`

**Signature:** `def simple_xor_encrypt(data: bytes, key: bytes) -> bytes:`

**Parameters:**

- `data: bytes` (required)
- `key: bytes` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `simple_xor_decrypt`

**Signature:** `def simple_xor_decrypt(data: bytes, key: bytes) -> bytes:`

**Parameters:**

- `data: bytes` (required)
- `key: bytes` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

---

### crypto.encryption_service

#### API Endpoints

#### `KeyManager.store_key`

存储密钥

**Signature:** `def store_key(self, key_id: str, key: bytes, metadata: Dict[str, Any] = None):`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)
- `key: bytes` (required)
- `metadata: Dict[Any]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `KeyManager.get_key`

获取密钥

**Signature:** `def get_key(self, key_id: str) -> Optional[bytes]:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `Optional[bytes]`

**Async:** No | **Visibility:** public

#### `KeyManager.rotate_key`

轮换密钥

**Signature:** `def rotate_key(self, key_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `KeyManager.list_keys`

列出所有密钥

**Signature:** `def list_keys(self) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `EncryptionService.encrypt`

加密数据

**Signature:** `def encrypt(self, data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `EncryptionService.decrypt`

解密数据

**Signature:** `def decrypt(self, encrypted_data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `EncryptionService.encrypt_json`

加密JSON数据

**Signature:** `def encrypt_json(self, data: Dict[str, Any], key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `EncryptionService.decrypt_json`

解密JSON数据

**Signature:** `def decrypt_json(self, encrypted_data: str, key_id: str = "master") -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `EncryptionService.generate_signature`

生成数字签名

**Signature:** `def generate_signature(self, data: str, key_id: str = "api") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `key_id: str` (default: 'api')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `EncryptionService.verify_signature`

验证数字签名

**Signature:** `def verify_signature(self, data: str, signature: str, key_id: str = "api") -> bool:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `signature: str` (required)
- `key_id: str` (default: 'api')

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `EncryptionService.create_token`

创建安全令牌

**Signature:** `def create_token(self, payload: Dict[str, Any], expiration_minutes: int = 60,   key_id: str = "session") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `payload: Dict[Any]` (required)
- `expiration_minutes: int` (default: 60)
- `key_id: str` (default: 'session')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `EncryptionService.verify_token`

验证安全令牌

**Signature:** `def verify_token(self, token: str, key_id: str = "session") -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)
- `key_id: str` (default: 'session')

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `EncryptionService.rotate_key`

轮换密钥

**Signature:** `def rotate_key(self, key_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `EncryptionService.get_key_info`

获取密钥信息

**Signature:** `def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `EncryptionService.list_keys`

列出所有密钥

**Signature:** `def list_keys(self) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `EncryptionService.encrypt_file`

加密文件

**Signature:** `def encrypt_file(self, input_file: str, output_file: str, key_id: str = "file") -> bool:`

**Parameters:**

- `self: Any` (required)
- `input_file: str` (required)
- `output_file: str` (required)
- `key_id: str` (default: 'file')

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `EncryptionService.decrypt_file`

解密文件

**Signature:** `def decrypt_file(self, input_file: str, output_file: str, key_id: str = "file") -> bool:`

**Parameters:**

- `self: Any` (required)
- `input_file: str` (required)
- `output_file: str` (required)
- `key_id: str` (default: 'file')

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `EncryptionService.health_check`

健康检查

**Signature:** `def health_check(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_encryption_service`

获取全局加密服务实例

**Signature:** `def get_encryption_service() -> EncryptionService:`

**Returns:** `EncryptionService`

**Async:** No | **Visibility:** public

#### `encrypt_data`

加密数据

**Signature:** `def encrypt_data(data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `decrypt_data`

解密数据

**Signature:** `def decrypt_data(encrypted_data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `encrypted_data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `create_secure_token`

创建安全令牌

**Signature:** `def create_secure_token(payload: Dict[str, Any], expiration_minutes: int = 60) -> Optional[str]:`

**Parameters:**

- `payload: Dict[Any]` (required)
- `expiration_minutes: int` (default: 60)

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `verify_secure_token`

验证安全令牌

**Signature:** `def verify_secure_token(token: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `token: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `store_key`

存储密钥

**Signature:** `def store_key(self, key_id: str, key: bytes, metadata: Dict[str, Any] = None):`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)
- `key: bytes` (required)
- `metadata: Dict[Any]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_key`

获取密钥

**Signature:** `def get_key(self, key_id: str) -> Optional[bytes]:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `Optional[bytes]`

**Async:** No | **Visibility:** public

#### `rotate_key`

轮换密钥

**Signature:** `def rotate_key(self, key_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `list_keys`

列出所有密钥

**Signature:** `def list_keys(self) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `encrypt`

加密数据

**Signature:** `def encrypt(self, data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `decrypt`

解密数据

**Signature:** `def decrypt(self, encrypted_data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `encrypt_dict`

加密字典数据

**Signature:** `def encrypt_dict(self, data: Dict[str, Any], key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `decrypt_dict`

解密字典数据

**Signature:** `def decrypt_dict(self, encrypted_data: str, key_id: str = "master") -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `generate_signature`

生成数据签名

**Signature:** `def generate_signature(self, data: str, key_id: str = "api") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `key_id: str` (default: 'api')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `verify_signature`

验证数据签名

**Signature:** `def verify_signature(self, data: str, signature: str, key_id: str = "api") -> bool:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `signature: str` (required)
- `key_id: str` (default: 'api')

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_secure_token`

创建安全令牌

**Signature:** `def create_secure_token(self, payload: Dict[str, Any], expiration_minutes: int = 60,   key_id: str = "session") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `payload: Dict[Any]` (required)
- `expiration_minutes: int` (default: 60)
- `key_id: str` (default: 'session')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `verify_secure_token`

验证安全令牌

**Signature:** `def verify_secure_token(self, token: str, key_id: str = "session") -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)
- `key_id: str` (default: 'session')

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `encrypt`

加密数据

**Signature:** `def encrypt(self, data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `decrypt`

解密数据

**Signature:** `def decrypt(self, encrypted_data: str, key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `encrypt_json`

加密JSON数据

**Signature:** `def encrypt_json(self, data: Dict[str, Any], key_id: str = "master") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: Dict[Any]` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `decrypt_json`

解密JSON数据

**Signature:** `def decrypt_json(self, encrypted_data: str, key_id: str = "master") -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_data: str` (required)
- `key_id: str` (default: 'master')

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `generate_signature`

生成数字签名

**Signature:** `def generate_signature(self, data: str, key_id: str = "api") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `key_id: str` (default: 'api')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `verify_signature`

验证数字签名

**Signature:** `def verify_signature(self, data: str, signature: str, key_id: str = "api") -> bool:`

**Parameters:**

- `self: Any` (required)
- `data: str` (required)
- `signature: str` (required)
- `key_id: str` (default: 'api')

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_token`

创建安全令牌

**Signature:** `def create_token(self, payload: Dict[str, Any], expiration_minutes: int = 60,   key_id: str = "session") -> Optional[str]:`

**Parameters:**

- `self: Any` (required)
- `payload: Dict[Any]` (required)
- `expiration_minutes: int` (default: 60)
- `key_id: str` (default: 'session')

**Returns:** `Optional[str]`

**Async:** No | **Visibility:** public

#### `verify_token`

验证安全令牌

**Signature:** `def verify_token(self, token: str, key_id: str = "session") -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `token: str` (required)
- `key_id: str` (default: 'session')

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `rotate_key`

轮换密钥

**Signature:** `def rotate_key(self, key_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_key_info`

获取密钥信息

**Signature:** `def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `list_keys`

列出所有密钥

**Signature:** `def list_keys(self) -> Dict[str, Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `encrypt_file`

加密文件

**Signature:** `def encrypt_file(self, input_file: str, output_file: str, key_id: str = "file") -> bool:`

**Parameters:**

- `self: Any` (required)
- `input_file: str` (required)
- `output_file: str` (required)
- `key_id: str` (default: 'file')

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `decrypt_file`

解密文件

**Signature:** `def decrypt_file(self, input_file: str, output_file: str, key_id: str = "file") -> bool:`

**Parameters:**

- `self: Any` (required)
- `input_file: str` (required)
- `output_file: str` (required)
- `key_id: str` (default: 'file')

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `health_check`

健康检查

**Signature:** `def health_check(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

---

### crypto.key_management

RQA2025 密钥管理器

负责密钥的生成、存储、轮换和生命周期管理
职责单一，专注于密钥管理

#### API Endpoints

#### `KeyManager.generate_key`

生成新密钥

Args:
    algorithm: 加密算法
    key_size: 密钥大小

Returns:
    密钥ID

**Signature:** `def generate_key(self, algorithm: str = "AES256-GCM", key_size: int = 256) -> str:`

**Parameters:**

- `self: Any` (required)
- `algorithm: str` (default: 'AES256-GCM')
- `key_size: int` (default: 256)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `KeyManager.get_key`

获取密钥

Args:
    key_id: 密钥ID

Returns:
    密钥对象，如果不存在则返回None

**Signature:** `def get_key(self, key_id: str) -> Optional[EncryptionKey]:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `Optional[EncryptionKey]`

**Async:** No | **Visibility:** public

#### `KeyManager.get_or_create_key`

获取密钥或创建新密钥

Args:
    key_id: 指定的密钥ID（可选）
    algorithm: 加密算法

Returns:
    密钥对象

**Signature:** `def get_or_create_key(self, key_id: Optional[str] = None, algorithm: str = "AES256-GCM") -> EncryptionKey:`

**Parameters:**

- `self: Any` (required)
- `key_id: Optional[str]` (default: None)
- `algorithm: str` (default: 'AES256-GCM')

**Returns:** `EncryptionKey`

**Async:** No | **Visibility:** public

#### `KeyManager.rotate_key`

轮换密钥

Args:
    key_id: 要轮换的密钥ID
    new_algorithm: 新的算法（可选）

Returns:
    新密钥ID

**Signature:** `def rotate_key(self, key_id: str, new_algorithm: Optional[str] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)
- `new_algorithm: Optional[str]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `KeyManager.list_keys`

列出所有密钥信息

**Signature:** `def list_keys(self) -> Dict[str, Dict]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `KeyManager.cleanup_expired_keys`

清理过期的密钥

Args:
    retention_days: 保留天数

Returns:
    清理的密钥数量

**Signature:** `def cleanup_expired_keys(self, retention_days: int = 90) -> int:`

**Parameters:**

- `self: Any` (required)
- `retention_days: int` (default: 90)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `generate_key`

生成新密钥

Args:
    algorithm: 加密算法
    key_size: 密钥大小

Returns:
    密钥ID

**Signature:** `def generate_key(self, algorithm: str = "AES256-GCM", key_size: int = 256) -> str:`

**Parameters:**

- `self: Any` (required)
- `algorithm: str` (default: 'AES256-GCM')
- `key_size: int` (default: 256)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `get_key`

获取密钥

Args:
    key_id: 密钥ID

Returns:
    密钥对象，如果不存在则返回None

**Signature:** `def get_key(self, key_id: str) -> Optional[EncryptionKey]:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)

**Returns:** `Optional[EncryptionKey]`

**Async:** No | **Visibility:** public

#### `get_or_create_key`

获取密钥或创建新密钥

Args:
    key_id: 指定的密钥ID（可选）
    algorithm: 加密算法

Returns:
    密钥对象

**Signature:** `def get_or_create_key(self, key_id: Optional[str] = None, algorithm: str = "AES256-GCM") -> EncryptionKey:`

**Parameters:**

- `self: Any` (required)
- `key_id: Optional[str]` (default: None)
- `algorithm: str` (default: 'AES256-GCM')

**Returns:** `EncryptionKey`

**Async:** No | **Visibility:** public

#### `rotate_key`

轮换密钥

Args:
    key_id: 要轮换的密钥ID
    new_algorithm: 新的算法（可选）

Returns:
    新密钥ID

**Signature:** `def rotate_key(self, key_id: str, new_algorithm: Optional[str] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `key_id: str` (required)
- `new_algorithm: Optional[str]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `list_keys`

列出所有密钥信息

**Signature:** `def list_keys(self) -> Dict[str, Dict]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `cleanup_expired_keys`

清理过期的密钥

Args:
    retention_days: 保留天数

Returns:
    清理的密钥数量

**Signature:** `def cleanup_expired_keys(self, retention_days: int = 90) -> int:`

**Parameters:**

- `self: Any` (required)
- `retention_days: int` (default: 90)

**Returns:** `int`

**Async:** No | **Visibility:** public

---

### filters.event_filters

事件过滤器模块
提供配置事件过滤功能

#### API Endpoints

#### `should_process`

判断是否应该处理事件

Args:
    event: 事件数据

Returns:
    是否应该处理

**Signature:** `def should_process(self, event: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_filter_info`

获取过滤器信息

Returns:
    过滤器信息

**Signature:** `def get_filter_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `should_process`

判断是否应该处理事件

**Signature:** `def should_process(self, event: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_filter_info`

获取过滤器信息

**Signature:** `def get_filter_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `should_process`

判断是否应该处理事件

**Signature:** `def should_process(self, event: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `sanitize_data`

清理敏感数据

**Signature:** `def sanitize_data(self, data: Any) -> Any:`

**Parameters:**

- `self: Any` (required)
- `data: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_filter_info`

获取过滤器信息

**Signature:** `def get_filter_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `should_process`

判断是否应该处理事件

**Signature:** `def should_process(self, event: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_filter_info`

获取过滤器信息

**Signature:** `def get_filter_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `should_process`

判断是否应该处理事件

**Signature:** `def should_process(self, event: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_filter_info`

获取过滤器信息

**Signature:** `def get_filter_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `should_process`

判断是否应该处理事件

**Signature:** `def should_process(self, event: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_filter_info`

获取过滤器信息

**Signature:** `def get_filter_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `add_filter`

添加过滤器

**Signature:** `def add_filter(self, filter_obj: IEventFilter) -> None:`

**Parameters:**

- `self: Any` (required)
- `filter_obj: IEventFilter` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `remove_filter`

移除过滤器

**Signature:** `def remove_filter(self, filter_obj: IEventFilter) -> bool:`

**Parameters:**

- `self: Any` (required)
- `filter_obj: IEventFilter` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `clear_filters`

清空所有过滤器

**Signature:** `def clear_filters(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `should_process`

判断是否应该处理事件

**Signature:** `def should_process(self, event: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `process_event`

处理事件

Args:
    event: 原始事件

Returns:
    处理后的事件，如果不应该处理则返回None

**Signature:** `def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `event: Dict[Any]` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_filter_info`

获取过滤器链信息

**Signature:** `def get_filter_info(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `create_type_filter`

创建事件类型过滤器

**Signature:** `def create_type_filter(event_types: List[str],   filter_type: FilterType = FilterType.INCLUDE) -> EventTypeFilter:`

**Parameters:**

- `event_types: List[str]` (required)
- `filter_type: FilterType` (default: ...)

**Returns:** `EventTypeFilter`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_sensitive_filter`

创建敏感数据过滤器

**Signature:** `def create_sensitive_filter(sensitive_keys: List[str],   replacement: str = "***", filter_type: FilterType = FilterType.EXCLUDE) -> SensitiveDataFilter:`

**Parameters:**

- `sensitive_keys: List[str]` (required)
- `replacement: str` (default: '***')
- `filter_type: FilterType` (default: ...)

**Returns:** `SensitiveDataFilter`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_pattern_filter`

创建模式过滤器

**Signature:** `def create_pattern_filter(pattern: str,   field: str = "message", filter_type: FilterType = FilterType.INCLUDE) -> PatternFilter:`

**Parameters:**

- `pattern: str` (required)
- `field: str` (default: 'message')
- `filter_type: FilterType` (default: ...)

**Returns:** `PatternFilter`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_time_filter`

创建时间范围过滤器

**Signature:** `def create_time_filter(start_time: Optional[str] = None,   end_time: Optional[str] = None, filter_type: FilterType = FilterType.INCLUDE) -> TimeRangeFilter:`

**Parameters:**

- `start_time: Optional[str]` (default: None)
- `end_time: Optional[str]` (default: None)
- `filter_type: FilterType` (default: ...)

**Returns:** `TimeRangeFilter`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_composite_filter`

创建复合过滤器

**Signature:** `def create_composite_filter(filters: List[IEventFilter],   operator: str = "AND") -> CompositeFilter:`

**Parameters:**

- `filters: List[IEventFilter]` (required)
- `operator: str` (default: 'AND')

**Returns:** `CompositeFilter`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

#### `create_default_filter_chain`

创建默认过滤器链

**Signature:** `def create_default_filter_chain() -> EventFilterChain:`

**Returns:** `EventFilterChain`

**Decorators:** `staticmethod`

**Async:** No | **Visibility:** public

---

### plugins.custom_auth_plugin

自定义认证插件示例

展示如何创建安全插件来扩展认证功能

#### API Endpoints

#### `create_plugin`

创建插件实例

**Signature:** `def create_plugin():`

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `plugin_info`

**Signature:** `def plugin_info(self) -> PluginInfo:`

**Parameters:**

- `self: Any` (required)

**Returns:** `PluginInfo`

**Decorators:** `property`

**Async:** No | **Visibility:** public

#### `initialize`

初始化插件

**Signature:** `def initialize(self, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭插件

**Signature:** `def shutdown(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `pre_auth_check`

预认证检查

**Signature:** `def pre_auth_check(self, username: str, **kwargs) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `post_auth_check`

后认证检查

**Signature:** `def post_auth_check(self, username: str, success: bool, **kwargs) -> None:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `success: bool` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_auth_stats`

获取认证统计信息

**Signature:** `def get_auth_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `block_user`

手动阻塞用户

**Signature:** `def block_user(self, username: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `unblock_user`

解除用户阻塞

**Signature:** `def unblock_user(self, username: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

---

### plugins.plugin_system

RQA2025 插件系统

提供插件化架构，支持动态加载和扩展安全组件

#### API Endpoints

#### `PluginManager.load_plugin`

加载插件

**Signature:** `def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PluginManager.unload_plugin`

卸载插件

**Signature:** `def unload_plugin(self, plugin_name: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PluginManager.get_plugin`

获取插件实例

**Signature:** `def get_plugin(self, plugin_name: str) -> Optional[SecurityPlugin]:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)

**Returns:** `Optional[SecurityPlugin]`

**Async:** No | **Visibility:** public

#### `PluginManager.list_plugins`

列出所有已加载的插件

**Signature:** `def list_plugins(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `PluginManager.call_capability`

调用指定能力的插件

**Signature:** `def call_capability(self, capability_name: str, *args, **kwargs) -> List[Any]:`

**Parameters:**

- `self: Any` (required)
- `capability_name: str` (required)

**Returns:** `List[Any]`

**Async:** No | **Visibility:** public

#### `PluginManager.reload_plugin`

重新加载插件

**Signature:** `def reload_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `PluginManager.discover_plugins`

发现可用的插件

**Signature:** `def discover_plugins(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `PluginManager.validate_plugin_dependencies`

验证插件依赖

**Signature:** `def validate_plugin_dependencies(self, plugin_name: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `get_plugin_manager`

获取全局插件管理器

**Signature:** `def get_plugin_manager() -> PluginManager:`

**Returns:** `PluginManager`

**Async:** No | **Visibility:** public

#### `load_security_plugin`

加载安全插件

**Signature:** `def load_security_plugin(plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:`

**Parameters:**

- `plugin_name: str` (required)
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `unload_security_plugin`

卸载安全插件

**Signature:** `def unload_security_plugin(plugin_name: str) -> bool:`

**Parameters:**

- `plugin_name: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `call_plugin_capability`

调用插件能力

**Signature:** `def call_plugin_capability(capability_name: str, *args, **kwargs) -> List[Any]:`

**Parameters:**

- `capability_name: str` (required)

**Returns:** `List[Any]`

**Async:** No | **Visibility:** public

#### `plugin_info`

插件信息

**Signature:** `def plugin_info(self) -> PluginInfo:`

**Parameters:**

- `self: Any` (required)

**Returns:** `PluginInfo`

**Decorators:** `property`, `abstractmethod`

**Async:** No | **Visibility:** public

#### `initialize`

初始化插件

**Signature:** `def initialize(self, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭插件

**Signature:** `def shutdown(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Decorators:** `abstractmethod`

**Async:** No | **Visibility:** public

#### `get_capability`

获取插件能力

**Signature:** `def get_capability(self, name: str) -> Optional[Callable]:`

**Parameters:**

- `self: Any` (required)
- `name: str` (required)

**Returns:** `Optional[Callable]`

**Async:** No | **Visibility:** public

#### `load_plugin`

加载插件

**Signature:** `def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `unload_plugin`

卸载插件

**Signature:** `def unload_plugin(self, plugin_name: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_plugin`

获取插件实例

**Signature:** `def get_plugin(self, plugin_name: str) -> Optional[SecurityPlugin]:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)

**Returns:** `Optional[SecurityPlugin]`

**Async:** No | **Visibility:** public

#### `list_plugins`

列出所有已加载的插件

**Signature:** `def list_plugins(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `call_capability`

调用指定能力的插件

**Signature:** `def call_capability(self, capability_name: str, *args, **kwargs) -> List[Any]:`

**Parameters:**

- `self: Any` (required)
- `capability_name: str` (required)

**Returns:** `List[Any]`

**Async:** No | **Visibility:** public

#### `reload_plugin`

重新加载插件

**Signature:** `def reload_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)
- `config: Optional[Dict[Any]]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `discover_plugins`

发现可用的插件

**Signature:** `def discover_plugins(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `validate_plugin_dependencies`

验证插件依赖

**Signature:** `def validate_plugin_dependencies(self, plugin_name: str) -> List[str]:`

**Parameters:**

- `self: Any` (required)
- `plugin_name: str` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

---

### plugins.examples.custom_auth_plugin

自定义认证插件示例

展示如何创建安全插件来扩展认证功能

#### API Endpoints

#### `create_plugin`

创建插件实例

**Signature:** `def create_plugin():`

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `plugin_info`

**Signature:** `def plugin_info(self) -> PluginInfo:`

**Parameters:**

- `self: Any` (required)

**Returns:** `PluginInfo`

**Decorators:** `property`

**Async:** No | **Visibility:** public

#### `initialize`

初始化插件

**Signature:** `def initialize(self, config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭插件

**Signature:** `def shutdown(self) -> None:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `pre_auth_check`

预认证检查

**Signature:** `def pre_auth_check(self, username: str, **kwargs) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `post_auth_check`

后认证检查

**Signature:** `def post_auth_check(self, username: str, success: bool, **kwargs) -> None:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `success: bool` (required)

**Returns:** `Any`

**Async:** No | **Visibility:** public

#### `get_auth_stats`

获取认证统计信息

**Signature:** `def get_auth_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `block_user`

手动阻塞用户

**Signature:** `def block_user(self, username: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `unblock_user`

解除用户阻塞

**Signature:** `def unblock_user(self, username: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

---

### services.data_encryption_service

RQA2025 数据加密管理器

实现端到端的数据加密和解密功能
支持多种加密算法和密钥管理

#### API Endpoints

#### `DataEncryptionManager.encrypt_data`

加密数据

Args:
    data: 要加密的数据
    algorithm: 加密算法
    key_id: 密钥ID，如果为None则使用当前活动密钥
    metadata: 元数据

Returns:
    加密结果

**Signature:** `def encrypt_data(self, data: Union[str, bytes], algorithm: str = "AES-256-GCM", key_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: Union[Any]` (required)
- `algorithm: str` (default: 'AES-256-GCM')
- `key_id: Optional[str]` (default: None)
- `metadata: Optional[Dict[Any]]` (default: None)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.decrypt_data`

解密数据

Args:
    encrypted_result: 加密结果

Returns:
    解密结果

**Signature:** `def decrypt_data(self, encrypted_result: EncryptionResult) -> DecryptionResult:`

**Parameters:**

- `self: Any` (required)
- `encrypted_result: EncryptionResult` (required)

**Returns:** `DecryptionResult`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.generate_key`

生成新密钥

Args:
    algorithm: 密钥算法
    expires_in_days: 过期天数

Returns:
    密钥ID

**Signature:** `def generate_key(self, algorithm: str = "AES-256", expires_in_days: Optional[int] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `algorithm: str` (default: 'AES-256')
- `expires_in_days: Optional[int]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.rotate_keys`

轮换密钥

Returns:
    新生成的密钥ID列表

**Signature:** `def rotate_keys(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.get_audit_logs`

获取审计日志

Args:
    limit: 限制条数

Returns:
    审计日志列表

**Signature:** `def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `limit: int` (default: 100)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.get_encryption_stats`

获取加密统计信息

Returns:
    统计信息

**Signature:** `def get_encryption_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.encrypt_batch`

批量加密数据

Args:
    data_list: 数据列表 [{'data': bytes, 'metadata': dict}]
    algorithm: 加密算法

Returns:
    加密结果列表

**Signature:** `def encrypt_batch(self, data_list: List[Dict[str, Any]],   algorithm: str = "AES - 256 - GCM") -> List[EncryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `data_list: List[Dict[Any]]` (required)
- `algorithm: str` (default: 'AES - 256 - GCM')

**Returns:** `List[EncryptionResult]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.decrypt_batch`

批量解密数据

Args:
    encrypted_results: 加密结果列表

Returns:
    解密结果列表

**Signature:** `def decrypt_batch(self, encrypted_results: List[EncryptionResult]) -> List[DecryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_results: List[EncryptionResult]` (required)

**Returns:** `List[DecryptionResult]`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.cleanup_expired_keys`

清理过期密钥

Returns:
    清理的密钥数量

**Signature:** `def cleanup_expired_keys(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.export_keys`

导出密钥

Args:
    export_path: 导出路径
    include_private: 是否包含私钥信息

**Signature:** `def export_keys(self, export_path: str, include_private: bool = False):`

**Parameters:**

- `self: Any` (required)
- `export_path: str` (required)
- `include_private: bool` (default: False)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `DataEncryptionManager.shutdown`

关闭加密管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `is_expired`

检查密钥是否过期

**Signature:** `def is_expired(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `can_use`

检查密钥是否可以使用

**Signature:** `def can_use(self) -> bool:`

**Parameters:**

- `self: Any` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `increment_usage`

增加使用计数

**Signature:** `def increment_usage(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `encrypt_data`

加密数据

Args:
    data: 要加密的数据
    algorithm: 加密算法
    key_id: 密钥ID，如果为None则使用当前活动密钥
    metadata: 元数据

Returns:
    加密结果

**Signature:** `def encrypt_data(self, data: Union[str, bytes], algorithm: str = "AES-256-GCM", key_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> EncryptionResult:`

**Parameters:**

- `self: Any` (required)
- `data: Union[Any]` (required)
- `algorithm: str` (default: 'AES-256-GCM')
- `key_id: Optional[str]` (default: None)
- `metadata: Optional[Dict[Any]]` (default: None)

**Returns:** `EncryptionResult`

**Async:** No | **Visibility:** public

#### `decrypt_data`

解密数据

Args:
    encrypted_result: 加密结果

Returns:
    解密结果

**Signature:** `def decrypt_data(self, encrypted_result: EncryptionResult) -> DecryptionResult:`

**Parameters:**

- `self: Any` (required)
- `encrypted_result: EncryptionResult` (required)

**Returns:** `DecryptionResult`

**Async:** No | **Visibility:** public

#### `generate_key`

生成新密钥

Args:
    algorithm: 密钥算法
    expires_in_days: 过期天数

Returns:
    密钥ID

**Signature:** `def generate_key(self, algorithm: str = "AES-256", expires_in_days: Optional[int] = None) -> str:`

**Parameters:**

- `self: Any` (required)
- `algorithm: str` (default: 'AES-256')
- `expires_in_days: Optional[int]` (default: None)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `rotate_keys`

轮换密钥

Returns:
    新生成的密钥ID列表

**Signature:** `def rotate_keys(self) -> List[str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[str]`

**Async:** No | **Visibility:** public

#### `get_audit_logs`

获取审计日志

Args:
    limit: 限制条数

Returns:
    审计日志列表

**Signature:** `def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `limit: int` (default: 100)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_encryption_stats`

获取加密统计信息

Returns:
    统计信息

**Signature:** `def get_encryption_stats(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `encrypt_batch`

批量加密数据

Args:
    data_list: 数据列表 [{'data': bytes, 'metadata': dict}]
    algorithm: 加密算法

Returns:
    加密结果列表

**Signature:** `def encrypt_batch(self, data_list: List[Dict[str, Any]],   algorithm: str = "AES - 256 - GCM") -> List[EncryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `data_list: List[Dict[Any]]` (required)
- `algorithm: str` (default: 'AES - 256 - GCM')

**Returns:** `List[EncryptionResult]`

**Async:** No | **Visibility:** public

#### `decrypt_batch`

批量解密数据

Args:
    encrypted_results: 加密结果列表

Returns:
    解密结果列表

**Signature:** `def decrypt_batch(self, encrypted_results: List[EncryptionResult]) -> List[DecryptionResult]:`

**Parameters:**

- `self: Any` (required)
- `encrypted_results: List[EncryptionResult]` (required)

**Returns:** `List[DecryptionResult]`

**Async:** No | **Visibility:** public

#### `cleanup_expired_keys`

清理过期密钥

Returns:
    清理的密钥数量

**Signature:** `def cleanup_expired_keys(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `export_keys`

导出密钥

Args:
    export_path: 导出路径
    include_private: 是否包含私钥信息

**Signature:** `def export_keys(self, export_path: str, include_private: bool = False):`

**Parameters:**

- `self: Any` (required)
- `export_path: str` (required)
- `include_private: bool` (default: False)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `shutdown`

关闭加密管理器

**Signature:** `def shutdown(self):`

**Parameters:**

- `self: Any` (required)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `simple_xor_encrypt`

**Signature:** `def simple_xor_encrypt(data: bytes, key: bytes) -> bytes:`

**Parameters:**

- `data: bytes` (required)
- `key: bytes` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

#### `simple_xor_decrypt`

**Signature:** `def simple_xor_decrypt(data: bytes, key: bytes) -> bytes:`

**Parameters:**

- `data: bytes` (required)
- `key: bytes` (required)

**Returns:** `bytes`

**Async:** No | **Visibility:** public

---

### services.web_management_service

Web管理服务
提供配置管理的Web界面功能

#### API Endpoints

#### `WebManagementService.get_dashboard_data`

获取仪表板数据"
Returns:
    Dict[str, Any]: 仪表板数据

**Signature:** `def get_dashboard_data(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `WebManagementService.get_config_tree`

将配置转换为树形结构

**Signature:** `def get_config_tree(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.update_config_value`

更新配置值

**Signature:** `def update_config_value(self, config: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)
- `path: str` (required)
- `value: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `WebManagementService.validate_config_changes`

验证配置变更

**Signature:** `def validate_config_changes(self, original_config: Dict[str, Any],   new_config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `original_config: Dict[Any]` (required)
- `new_config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `WebManagementService.get_config_statistics`

获取配置统计信息

**Signature:** `def get_config_statistics(self, config: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `WebManagementService.encrypt_sensitive_config`

加密敏感配置

**Signature:** `def encrypt_sensitive_config(self, config: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `WebManagementService.decrypt_config`

解密配置

**Signature:** `def decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `WebManagementService.get_sync_nodes`

获取同步节点

**Signature:** `def get_sync_nodes(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.sync_config_to_nodes`

同步配置到节点

**Signature:** `def sync_config_to_nodes(self, config: Dict[str, Any],   target_nodes: Optional[List[str]] = None):`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)
- `target_nodes: Optional[List[str]]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `WebManagementService.get_sync_history`

获取同步历史

**Signature:** `def get_sync_history(self, limit: int = 20) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `limit: int` (default: 20)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.get_conflicts`

获取冲突

**Signature:** `def get_conflicts(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.resolve_conflicts`

解决冲突

**Signature:** `def resolve_conflicts(self, conflicts: List[Dict[str, Any]],   strategy: str = "merge"):`

**Parameters:**

- `self: Any` (required)
- `conflicts: List[Dict[Any]]` (required)
- `strategy: str` (default: 'merge')

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `WebManagementService.authenticate_user`

用户认证

**Signature:** `def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `password: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.check_permission`

检查用户权限

**Signature:** `def check_permission(self, username: str, permission: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `permission: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `WebManagementService.create_session`

创建用户会话

**Signature:** `def create_session(self, username: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `WebManagementService.validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.invalidate_session`

使会话失效

**Signature:** `def invalidate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `WebManagementService.get_user_info`

获取用户信息

**Signature:** `def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.add_user`

添加用户

**Signature:** `def add_user(self, username: str, password: str, role: str = "user",   permissions: Optional[List[str]] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `password: str` (required)
- `role: str` (default: 'user')
- `permissions: Optional[List[str]]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `WebManagementService.update_user`

更新用户信息

**Signature:** `def update_user(self, username: str, **kwargs) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `WebManagementService.delete_user`

删除用户

**Signature:** `def delete_user(self, username: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `WebManagementService.list_users`

列出所有用户

**Signature:** `def list_users(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.list_sessions`

列出所有会话

**Signature:** `def list_sessions(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `WebManagementService.get_permissions`

获取权限定义

**Signature:** `def get_permissions(self) -> Dict[str, str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `WebManagementService.cleanup_expired_sessions`

清理过期会话

**Signature:** `def cleanup_expired_sessions(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

#### `get_dashboard_data`

获取仪表板数据"
Returns:
    Dict[str, Any]: 仪表板数据

**Signature:** `def get_dashboard_data(self) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_config_tree`

将配置转换为树形结构

**Signature:** `def get_config_tree(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `update_config_value`

更新配置值

**Signature:** `def update_config_value(self, config: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)
- `path: str` (required)
- `value: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `validate_config_changes`

验证配置变更

**Signature:** `def validate_config_changes(self, original_config: Dict[str, Any],   new_config: Dict[str, Any]) -> bool:`

**Parameters:**

- `self: Any` (required)
- `original_config: Dict[Any]` (required)
- `new_config: Dict[Any]` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_config_statistics`

获取配置统计信息

**Signature:** `def get_config_statistics(self, config: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `encrypt_sensitive_config`

加密敏感配置

**Signature:** `def encrypt_sensitive_config(self, config: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `decrypt_config`

解密配置

**Signature:** `def decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `get_sync_nodes`

获取同步节点

**Signature:** `def get_sync_nodes(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `sync_config_to_nodes`

同步配置到节点

**Signature:** `def sync_config_to_nodes(self, config: Dict[str, Any],   target_nodes: Optional[List[str]] = None):`

**Parameters:**

- `self: Any` (required)
- `config: Dict[Any]` (required)
- `target_nodes: Optional[List[str]]` (default: None)

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `get_sync_history`

获取同步历史

**Signature:** `def get_sync_history(self, limit: int = 20) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `limit: int` (default: 20)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_conflicts`

获取冲突

**Signature:** `def get_conflicts(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `resolve_conflicts`

解决冲突

**Signature:** `def resolve_conflicts(self, conflicts: List[Dict[str, Any]],   strategy: str = "merge"):`

**Parameters:**

- `self: Any` (required)
- `conflicts: List[Dict[Any]]` (required)
- `strategy: str` (default: 'merge')

**Returns:** `None`

**Async:** No | **Visibility:** public

#### `authenticate_user`

用户认证

**Signature:** `def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `password: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `check_permission`

检查用户权限

**Signature:** `def check_permission(self, username: str, permission: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `permission: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `create_session`

创建用户会话

**Signature:** `def create_session(self, username: str) -> str:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `str`

**Async:** No | **Visibility:** public

#### `validate_session`

验证会话

**Signature:** `def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `invalidate_session`

使会话失效

**Signature:** `def invalidate_session(self, session_id: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `session_id: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `get_user_info`

获取用户信息

**Signature:** `def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `Optional[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `add_user`

添加用户

**Signature:** `def add_user(self, username: str, password: str, role: str = "user",   permissions: Optional[List[str]] = None) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)
- `password: str` (required)
- `role: str` (default: 'user')
- `permissions: Optional[List[str]]` (default: None)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `update_user`

更新用户信息

**Signature:** `def update_user(self, username: str, **kwargs) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `delete_user`

删除用户

**Signature:** `def delete_user(self, username: str) -> bool:`

**Parameters:**

- `self: Any` (required)
- `username: str` (required)

**Returns:** `bool`

**Async:** No | **Visibility:** public

#### `list_users`

列出所有用户

**Signature:** `def list_users(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `list_sessions`

列出所有会话

**Signature:** `def list_sessions(self) -> List[Dict[str, Any]]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `List[Dict[Any]]`

**Async:** No | **Visibility:** public

#### `get_permissions`

获取权限定义

**Signature:** `def get_permissions(self) -> Dict[str, str]:`

**Parameters:**

- `self: Any` (required)

**Returns:** `Dict[Any]`

**Async:** No | **Visibility:** public

#### `cleanup_expired_sessions`

清理过期会话

**Signature:** `def cleanup_expired_sessions(self) -> int:`

**Parameters:**

- `self: Any` (required)

**Returns:** `int`

**Async:** No | **Visibility:** public

---
