#!/usr/bin/env python3
"""
RQA2025 RBAC权限控制系统
基于角色的访问控制实现
"""

from typing import Dict, List, Set, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """权限枚举"""
    # 系统管理权限
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"

    # 用户管理权限
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"

    # 交易权限
    TRADE_EXECUTE = "trade:execute"
    TRADE_CANCEL = "trade:cancel"
    TRADE_HISTORY = "trade:history"

    # 策略权限
    STRATEGY_CREATE = "strategy:create"
    STRATEGY_READ = "strategy:read"
    STRATEGY_UPDATE = "strategy:update"
    STRATEGY_DELETE = "strategy:delete"
    STRATEGY_EXECUTE = "strategy:execute"

    # 数据权限
    DATA_MARKET_READ = "data:market:read"
    DATA_HISTORICAL_READ = "data:historical:read"
    DATA_REALTIME_READ = "data:realtime:read"

    # 报告权限
    REPORT_CREATE = "report:create"
    REPORT_READ = "report:read"
    REPORT_EXPORT = "report:export"

    # 风险管理权限
    RISK_ASSESS = "risk:assess"
    RISK_CONFIGURE = "risk:configure"
    RISK_MONITOR = "risk:monitor"


@dataclass
class Role:
    """角色定义"""
    name: str
    description: str
    permissions: Set[Permission]
    inherits_from: List[str] = None  # 继承的角色列表

    def __post_init__(self):
        if self.inherits_from is None:
            self.inherits_from = []


@dataclass
class User:
    """用户定义"""
    username: str
    roles: List[str]
    is_active: bool = True


class RBACManager:
    """RBAC权限管理器"""

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self._permission_cache: Dict[str, Set[Permission]] = {}

        # 初始化默认角色
        self._init_default_roles()

        logger.info("RBAC权限管理器初始化完成")

    def _init_default_roles(self):
        """初始化默认角色"""
        # 超级管理员角色
        self.roles["super_admin"] = Role(
            name="super_admin",
            description="超级管理员，具有所有权限",
            permissions=set(Permission),
            inherits_from=[]
        )

        # 系统管理员角色
        self.roles["admin"] = Role(
            name="admin",
            description="系统管理员",
            permissions={
                Permission.SYSTEM_ADMIN,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_MONITOR,
                Permission.USER_CREATE,
                Permission.USER_READ,
                Permission.USER_UPDATE,
                Permission.USER_DELETE,
                Permission.TRADE_EXECUTE,
                Permission.STRATEGY_CREATE,
                Permission.STRATEGY_READ,
                Permission.STRATEGY_UPDATE,
                Permission.STRATEGY_DELETE,
                Permission.DATA_MARKET_READ,
                Permission.DATA_HISTORICAL_READ,
                Permission.REPORT_CREATE,
                Permission.REPORT_READ,
                Permission.REPORT_EXPORT
            },
            inherits_from=[]
        )

        # 交易员角色
        self.roles["trader"] = Role(
            name="trader",
            description="专业交易员",
            permissions={
                Permission.TRADE_EXECUTE,
                Permission.TRADE_CANCEL,
                Permission.TRADE_HISTORY,
                Permission.STRATEGY_READ,
                Permission.STRATEGY_EXECUTE,
                Permission.DATA_MARKET_READ,
                Permission.DATA_REALTIME_READ,
                Permission.DATA_HISTORICAL_READ,
                Permission.REPORT_READ
            },
            inherits_from=[]
        )

        # 分析师角色
        self.roles["analyst"] = Role(
            name="analyst",
            description="数据分析师",
            permissions={
                Permission.DATA_MARKET_READ,
                Permission.DATA_HISTORICAL_READ,
                Permission.DATA_REALTIME_READ,
                Permission.STRATEGY_READ,
                Permission.REPORT_CREATE,
                Permission.REPORT_READ,
                Permission.REPORT_EXPORT,
                Permission.RISK_ASSESS,
                Permission.RISK_MONITOR
            },
            inherits_from=[]
        )

        # 普通用户角色
        self.roles["user"] = Role(
            name="user",
            description="普通用户",
            permissions={
                Permission.DATA_MARKET_READ,
                Permission.STRATEGY_READ,
                Permission.REPORT_READ
            },
            inherits_from=[]
        )

    def create_role(self, name: str, description: str,
                    permissions: List[Permission],
                    inherits_from: List[str] = None) -> bool:
        """
        创建新角色

        Args:
            name: 角色名称
            description: 角色描述
            permissions: 权限列表
            inherits_from: 继承的角色列表

        Returns:
            创建是否成功
        """
        if name in self.roles:
            logger.warning(f"角色 {name} 已存在")
            return False

        # 验证继承的角色是否存在
        for parent_role in inherits_from or []:
            if parent_role not in self.roles:
                logger.error(f"父角色 {parent_role} 不存在")
                return False

        role = Role(
            name=name,
            description=description,
            permissions=set(permissions),
            inherits_from=inherits_from or []
        )

        self.roles[name] = role

        # 清除权限缓存
        self._permission_cache.clear()

        logger.info(f"角色 {name} 创建成功")
        return True

    def assign_role_to_user(self, username: str, role_name: str) -> bool:
        """
        为用户分配角色

        Args:
            username: 用户名
            role_name: 角色名

        Returns:
            分配是否成功
        """
        if role_name not in self.roles:
            logger.error(f"角色 {role_name} 不存在")
            return False

        user = self.users.get(username)
        if not user:
            # 创建新用户
            user = User(username=username, roles=[])
            self.users[username] = user

        if role_name not in user.roles:
            user.roles.append(role_name)
            # 清除用户权限缓存
            self._permission_cache.pop(username, None)
            logger.info(f"角色 {role_name} 已分配给用户 {username}")
        else:
            logger.warning(f"用户 {username} 已有角色 {role_name}")

        return True

    def revoke_role_from_user(self, username: str, role_name: str) -> bool:
        """
        从用户撤销角色

        Args:
            username: 用户名
            role_name: 角色名

        Returns:
            撤销是否成功
        """
        user = self.users.get(username)
        if not user:
            logger.warning(f"用户 {username} 不存在")
            return False

        if role_name in user.roles:
            user.roles.remove(role_name)
            # 清除用户权限缓存
            self._permission_cache.pop(username, None)
            logger.info(f"角色 {role_name} 已从用户 {username} 撤销")
            return True
        else:
            logger.warning(f"用户 {username} 没有角色 {role_name}")
            return False

    def check_permission(self, username: str, permission: Permission) -> bool:
        """
        检查用户是否有指定权限

        Args:
            username: 用户名
            permission: 权限

        Returns:
            是否有权限
        """
        user = self.users.get(username)
        if not user or not user.is_active:
            return False

        # 获取用户所有权限
        user_permissions = self._get_user_permissions(username)

        return permission in user_permissions

    def check_any_permission(self, username: str, permissions: List[Permission]) -> bool:
        """
        检查用户是否有任意一个权限

        Args:
            username: 用户名
            permissions: 权限列表

        Returns:
            是否有任意权限
        """
        for permission in permissions:
            if self.check_permission(username, permission):
                return True
        return False

    def check_all_permissions(self, username: str, permissions: List[Permission]) -> bool:
        """
        检查用户是否有所有权限

        Args:
            username: 用户名
            permissions: 权限列表

        Returns:
            是否有所有权限
        """
        for permission in permissions:
            if not self.check_permission(username, permission):
                return False
        return True

    def get_user_roles(self, username: str) -> List[str]:
        """
        获取用户角色列表

        Args:
            username: 用户名

        Returns:
            角色列表
        """
        user = self.users.get(username)
        return user.roles if user else []

    def get_user_permissions(self, username: str) -> Set[Permission]:
        """
        获取用户所有权限

        Args:
            username: 用户名

        Returns:
            权限集合
        """
        return self._get_user_permissions(username)

    def _get_user_permissions(self, username: str) -> Set[Permission]:
        """
        获取用户所有权限（包括继承的权限）

        Args:
            username: 用户名

        Returns:
            权限集合
        """
        # 检查缓存
        if username in self._permission_cache:
            return self._permission_cache[username]

        user = self.users.get(username)
        if not user or not user.is_active:
            return set()

        permissions = set()

        # 递归获取所有角色的权限
        def collect_permissions(role_names: List[str], visited: Set[str]):
            for role_name in role_names:
                if role_name in visited:
                    continue  # 避免循环继承

                visited.add(role_name)
                role = self.roles.get(role_name)
                if role:
                    # 添加角色直接权限
                    permissions.update(role.permissions)
                    # 递归添加继承的权限
                    collect_permissions(role.inherits_from, visited)

        collect_permissions(user.roles, set())

        # 缓存结果
        self._permission_cache[username] = permissions

        return permissions

    def get_role_info(self, role_name: str) -> Optional[Dict]:
        """
        获取角色信息

        Args:
            role_name: 角色名

        Returns:
            角色信息字典或None
        """
        role = self.roles.get(role_name)
        if not role:
            return None

        return {
            "name": role.name,
            "description": role.description,
            "permissions": [p.value for p in role.permissions],
            "inherits_from": role.inherits_from
        }

    def list_roles(self) -> List[Dict]:
        """
        列出所有角色

        Returns:
            角色信息列表
        """
        return [self.get_role_info(name) for name in self.roles.keys()]

    def list_users(self) -> List[Dict]:
        """
        列出所有用户

        Returns:
            用户信息列表
        """
        return [
            {
                "username": user.username,
                "roles": user.roles,
                "is_active": user.is_active,
                "permissions": [p.value for p in self._get_user_permissions(user.username)]
            }
            for user in self.users.values()
        ]


# 全局RBAC管理器实例
rbac_manager = RBACManager()


def init_rbac_system():
    """初始化RBAC系统"""
    # 创建测试用户并分配角色
    rbac_manager.assign_role_to_user("admin", "admin")
    rbac_manager.assign_role_to_user("trader", "trader")
    rbac_manager.assign_role_to_user("analyst", "analyst")

    logger.info("RBAC权限控制系统初始化完成")


# 装饰器函数
def require_permission(permission: Permission):
    """
    权限检查装饰器

    Args:
        permission: 必需的权限

    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # 从kwargs中获取用户名（在实际应用中应该从JWT令牌中提取）
            username = kwargs.get('username')
            if not username:
                # 尝试从args中查找（假设第一个参数是用户名）
                if args and isinstance(args[0], str):
                    username = args[0]

            if not username:
                raise ValueError("无法确定用户名进行权限检查")

            if not rbac_manager.check_permission(username, permission):
                raise PermissionError(f"用户 {username} 没有权限: {permission.value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_any_permission(permissions: List[Permission]):
    """
    任意权限检查装饰器

    Args:
        permissions: 权限列表

    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            username = kwargs.get('username')
            if not username and args and isinstance(args[0], str):
                username = args[0]

            if not username:
                raise ValueError("无法确定用户名进行权限检查")

            if not rbac_manager.check_any_permission(username, permissions):
                permission_names = [p.value for p in permissions]
                raise PermissionError(f"用户 {username} 缺少必需权限: {permission_names}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # 初始化RBAC系统
    init_rbac_system()

    # 测试RBAC权限控制
    print("👥 测试RBAC权限控制系统")
    print("=" * 50)

    # 测试权限检查
    print("🔍 测试权限检查")

    test_cases = [
        ("admin", Permission.SYSTEM_ADMIN, True),
        ("admin", Permission.TRADE_EXECUTE, True),
        ("trader", Permission.SYSTEM_ADMIN, False),
        ("trader", Permission.TRADE_EXECUTE, True),
        ("analyst", Permission.STRATEGY_CREATE, False),
        ("analyst", Permission.DATA_MARKET_READ, True),
    ]

    for username, permission, expected in test_cases:
        result = rbac_manager.check_permission(username, permission)
        status = "✅" if result == expected else "❌"
        print(f"{status} 用户 {username} {permission.value}: {result} (期望: {expected})")

    # 测试用户角色和权限
    print("\n📋 测试用户角色和权限")
    for user_info in rbac_manager.list_users():
        print(f"用户: {user_info['username']}")
        print(f"  角色: {user_info['roles']}")
        print(f"  权限数量: {len(user_info['permissions'])}")
        print(f"  示例权限: {user_info['permissions'][:3]}")

    # 测试权限装饰器
    print("\n🎨 测试权限装饰器")

    @require_permission(Permission.TRADE_EXECUTE)
    def execute_trade(username: str, symbol: str, quantity: int):
        return f"用户 {username} 执行交易: {symbol} x {quantity}"

    try:
        result = execute_trade("trader", "AAPL", 100)
        print(f"✅ {result}")
    except PermissionError as e:
        print(f"❌ {e}")

    try:
        result = execute_trade("analyst", "AAPL", 100)
        print(f"✅ {result}")
    except PermissionError as e:
        print(f"❌ {e}")

    print("\n🎉 RBAC权限控制系统测试完成")
