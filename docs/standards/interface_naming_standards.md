# 接口命名规范

## 概述

本文档定义了RQA2025系统中接口的命名规范，确保代码的一致性和可读性。

## 基本原则

### 1. 接口标识

所有接口必须通过命名清晰标识其为接口：

#### ✅ 推荐的接口命名

```python
# 使用大写字母I开头 + Pascal命名
class IUserService(ABC):
class IConfigManager(ABC):
class IDataAccess(ABC):

# 或者使用Interface后缀
class UserServiceInterface(ABC):
class ConfigManagerInterface(ABC):
class DataAccessInterface(ABC):

# 对于协议，使用Protocol后缀
class ServiceProtocol(Protocol):
class CacheProtocol(Protocol):
```

#### ❌ 避免的接口命名

```python
# 没有接口标识
class UserService:          # 容易与实现类混淆
class ConfigManager:        # 不清楚是接口还是实现

# 其他语言风格
interface UserService:      # Python不支持interface关键字
```

### 2. 方法命名规范

#### 标准方法签名

```python
# ✅ 推荐的方法命名
def get_user_by_id(self, user_id: str) -> Optional[User]:
def create_user(self, user_data: Dict[str, Any]) -> User:
def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
def delete_user(self, user_id: str) -> bool:
def find_users_by_criteria(self, criteria: Dict[str, Any]) -> List[User]:
def validate_user_data(self, data: Dict[str, Any]) -> List[str]:
```

#### 方法命名规则

1. **动词 + 名词**：使用动词描述操作，名词描述对象
2. **一致性**：相同操作使用相同动词
3. **简洁性**：方法名应简洁明了，避免过长
4. **避免缩写**：除非是行业标准缩写

#### 常见动词映射

```python
# 查询操作
get_     # 获取单个对象
find_    # 查找多个对象
list_    # 列出所有对象
search_  # 搜索对象

# 修改操作
create_  # 创建新对象
update_  # 更新现有对象
delete_  # 删除对象
save_    # 保存对象

# 业务操作
process_ # 处理业务逻辑
execute_ # 执行操作
validate_# 验证数据
calculate_# 计算结果
```

### 3. 参数命名规范

#### 标准参数命名

```python
# ✅ 推荐的参数命名
def get_user_by_id(self, user_id: str) -> Optional[User]:
def create_user(self, user_data: Dict[str, Any]) -> User:
def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
def transfer_money(self, from_account: str, to_account: str, amount: float) -> bool:
def search_orders(self, criteria: Dict[str, Any], page: int = 1, size: int = 20) -> List[Order]:
```

#### 参数命名规则

1. **描述性**：参数名应描述其用途
2. **一致性**：相同概念使用相同参数名
3. **类型提示**：必须提供类型提示
4. **可选参数**：使用默认值表示可选参数

### 4. 返回值规范

#### 标准返回值类型

```python
# ✅ 推荐的返回值类型
def get_user(self, user_id: str) -> Optional[User]:          # 可能不存在
def create_user(self, data: Dict[str, Any]) -> User:         # 总是返回对象
def update_user(self, user_id: str, data: Dict[str, Any]) -> bool:  # 成功/失败
def validate_data(self, data: Dict[str, Any]) -> List[str]:  # 错误列表
def calculate_total(self, items: List[Item]) -> float:        # 数值结果
def process_batch(self, items: List[Any]) -> Dict[str, Any]: # 详细结果
```

#### 返回值设计原则

1. **明确语义**：返回值应明确表示操作结果
2. **一致性**：相似操作使用相似返回值类型
3. **异常 vs 返回值**：使用异常表示错误情况，使用返回值表示正常变体

## 接口分类规范

### 1. 服务接口

#### 标准服务接口结构

```python
class IUserService(StandardServiceInterface):
    """用户服务接口"""

    # 标准服务接口方法（继承自StandardServiceInterface）
    # service_name, service_version, initialize, shutdown, health_check, get_status, get_metrics

    # 业务特定方法
    def create_user(self, user_data: Dict[str, Any]) -> User:
    def get_user(self, user_id: str) -> Optional[User]:
    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
    def delete_user(self, user_id: str) -> bool:
    def authenticate_user(self, credentials: Dict[str, Any]) -> Optional[str]:
```

### 2. 数据访问接口

#### 数据访问接口结构

```python
class IUserRepository(DataAccessInterface):
    """用户数据访问接口"""

    # 标准数据访问方法（继承自DataAccessInterface）
    # connect, disconnect, is_connected, execute_query, execute_command, begin_transaction

    # 数据访问特定方法
    def find_by_id(self, user_id: str) -> Optional[User]:
    def find_by_email(self, email: str) -> Optional[User]:
    def save(self, user: User) -> User:
    def delete(self, user_id: str) -> bool:
    def exists(self, user_id: str) -> bool:
```

### 3. 配置接口

#### 配置接口结构

```python
class IConfigManager(ConfigurationInterface):
    """配置管理接口"""

    # 标准配置方法（继承自ConfigurationInterface）
    # load, save, get, set, has, delete, validate, reload

    # 配置管理特定方法
    def get_all(self, prefix: str = "") -> Dict[str, Any]:
    def watch(self, key: str, callback: Callable) -> bool:
    def export(self, format: str = "json") -> Union[str, Dict[str, Any]]:
```

## 文档规范

### 1. 接口文档

#### 标准接口文档格式

```python
class IUserService(StandardServiceInterface):
    """
    用户服务接口

    提供用户管理相关的核心业务功能，包括用户的创建、查询、更新和删除操作。

    设计原则:
    - 用户ID使用UUID格式保证唯一性
    - 密码加密存储，认证使用安全哈希
    - 支持软删除，保留用户数据完整性

    依赖:
    - IUserRepository: 用户数据访问
    - IEventPublisher: 事件发布
    - ICache: 用户缓存

    作者: 开发团队
    创建时间: 2025-10-06
    版本: 1.0.0
    """

    @abstractmethod
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """
        创建新用户

        执行完整的用户创建流程，包括数据验证、密码加密、数据库保存和事件发布。

        Args:
            user_data: 用户数据字典，必须包含以下字段:
                - username: 用户名，字符串，3-50字符
                - email: 邮箱地址，必须唯一
                - password: 原始密码，创建时会加密存储

        Returns:
            User: 创建成功的用户对象

        Raises:
            ValidationError: 用户数据验证失败
            BusinessLogicError: 用户名或邮箱已存在
            InfrastructureError: 数据库操作失败

        示例:
            >>> user_data = {
            ...     "username": "john_doe",
            ...     "email": "john@example.com",
            ...     "password": "secure_password"
            ... }
            >>> user = user_service.create_user(user_data)
        """
```

### 2. 方法文档

#### 方法文档模板

```python
@abstractmethod
def get_user(self, user_id: str) -> Optional[User]:
    """
    根据用户ID获取用户信息

    从数据库或缓存中获取指定用户的完整信息。如果用户不存在，返回None。

    Args:
        user_id (str): 用户的唯一标识符，UUID格式

    Returns:
        Optional[User]: 用户对象，如果用户不存在则返回None

    Raises:
        ValidationError: user_id格式无效
        InfrastructureError: 数据访问失败

    注意:
        - 此方法会首先检查缓存，如果缓存未命中则查询数据库
        - 查询结果会被缓存以提高后续访问性能
        - 软删除的用户不会被返回

    性能:
        - 平均响应时间: < 10ms (缓存命中)
        - 平均响应时间: < 50ms (缓存未命中)

    示例:
        >>> user = user_service.get_user("550e8400-e29b-41d4-a716-446655440000")
        >>> if user:
        ...     print(f"找到用户: {user.username}")
    """
```

### 3. 参数文档

#### 参数文档规范

```python
# ✅ 推荐的参数文档
def create_user(self, user_data: Dict[str, Any]) -> User:
    """
    Args:
        user_data (Dict[str, Any]): 用户数据字典
            - username (str): 用户名，必需，3-50字符
            - email (str): 邮箱地址，必需，必须唯一
            - password (str): 原始密码，必需，至少8字符
            - role (str, optional): 用户角色，默认'user'
            - metadata (Dict[str, Any], optional): 额外元数据
    """
```

## 版本控制

### 接口版本管理

```python
class IUserServiceV1(StandardServiceInterface):
    """用户服务接口 v1.0"""

    service_version = "1.0.0"

class IUserServiceV2(StandardServiceInterface):
    """用户服务接口 v2.0 - 增加OAuth支持"""

    service_version = "2.0.0"

    # v1.0方法
    def create_user(self, user_data: Dict[str, Any]) -> User: ...

    # 新增方法
    def authenticate_with_oauth(self, provider: str, token: str) -> User: ...
```

### 向后兼容性

1. **新增方法**：在新的接口版本中添加方法
2. **方法签名变更**：创建新的接口版本，避免破坏现有实现
3. **废弃方法**：使用`@deprecated`装饰器标记废弃方法

## 代码示例

### 完整接口示例

```python
"""
用户服务接口定义
符合RQA2025接口命名规范
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.core.standard_interfaces import BusinessServiceInterface
from src.core.unified_exceptions import ValidationError, BusinessLogicError


class IUserService(BusinessServiceInterface):
    """
    用户服务接口

    提供完整的用户生命周期管理功能。

    作者: 开发团队
    创建时间: 2025-10-06
    版本: 1.0.0
    """

    # 实现标准业务服务接口方法
    @property
    def service_name(self) -> str:
        return "user_service"

    @property
    def service_version(self) -> str:
        return "1.0.0"

    # 用户管理方法
    @abstractmethod
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建新用户

        Args:
            user_data: 用户数据，包含username, email, password等字段

        Returns:
            Dict[str, Any]: 创建的用户信息
        """
        pass

    @abstractmethod
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取用户

        Args:
            user_id: 用户ID

        Returns:
            Optional[Dict[str, Any]]: 用户信息或None
        """
        pass

    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        根据邮箱获取用户

        Args:
            email: 用户邮箱

        Returns:
            Optional[Dict[str, Any]]: 用户信息或None
        """
        pass

    @abstractmethod
    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """
        更新用户信息

        Args:
            user_id: 用户ID
            user_data: 要更新的用户数据

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """
        删除用户（软删除）

        Args:
            user_id: 用户ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    def authenticate_user(self, credentials: Dict[str, Any]) -> Optional[str]:
        """
        用户认证

        Args:
            credentials: 认证凭据，包含username/email和password

        Returns:
            Optional[str]: 认证成功返回用户ID，失败返回None
        """
        pass

    @abstractmethod
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        修改密码

        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码

        Returns:
            bool: 修改是否成功
        """
        pass

    @abstractmethod
    def list_users(self, filters: Optional[Dict[str, Any]] = None,
                  page: int = 1, size: int = 20) -> Dict[str, Any]:
        """
        分页查询用户列表

        Args:
            filters: 查询过滤条件
            page: 页码，从1开始
            size: 每页大小

        Returns:
            Dict[str, Any]: 包含用户列表和分页信息的字典
        """
        pass
```

## 检查清单

### 接口设计检查清单

- [ ] 接口名称以I开头或使用Interface后缀
- [ ] 方法名使用动词+名词格式
- [ ] 参数名描述性强，有完整的类型提示
- [ ] 返回值类型明确，异常情况通过异常处理
- [ ] 文档完整，包含参数、返回值、异常说明
- [ ] 接口继承自标准接口协议
- [ ] 方法签名一致性好，遵循命名规范

### 实现检查清单

- [ ] 实现类正确实现了所有接口方法
- [ ] 方法实现与接口文档一致
- [ ] 异常处理符合接口规范
- [ ] 单元测试覆盖所有接口方法
- [ ] 集成测试验证接口契约

---

*最后更新: 2025年10月6日*

