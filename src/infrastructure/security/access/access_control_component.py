import os
import logging
#!/usr/bin/env python3
"""
RQA2025 访问控制系统

提供基于角色的访问控制(RBAC)和权限管理功能
"""

import hashlib
import threading
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# 导入统一基础设施集成层
try:
    from src.integration import get_trading_layer_adapter
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False


logger = logging.getLogger(__name__)


class UserRole(Enum):

    """用户角色枚举"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    GUEST = "guest"


class Permission(Enum):

    """权限枚举"""
    # 交易权限
    TRADE_EXECUTE = "trade:execute"
    TRADE_CANCEL = "trade:cancel"
    ORDER_PLACE = "order:place"
    ORDER_CANCEL = "order:cancel"

    # 数据权限
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_EXPORT = "data:export"

    # 系统权限
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    USER_MANAGE = "user:manage"

    # 审计权限
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"


@dataclass
class User:

    """用户"""
    user_id: str
    username: str
    email: str
    roles: Set[UserRole]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


@dataclass
class RoleDefinition:

    """角色定义"""
    role: UserRole
    name: str
    description: str
    permissions: Set[Permission]
    parent_roles: Set[UserRole]


@dataclass
class UserSession:

    """用户会话"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return datetime.now() > self.expires_at

    def extend_session(self, minutes: int = 60) -> None:
        """延长会话时间"""
        self.expires_at = datetime.now() + timedelta(minutes=minutes)


@dataclass
class AccessPolicy:

    """访问策略"""
    policy_id: str
    name: str
    description: str
    resource_pattern: str  # 资源匹配模式，如 "/api / trade/*"
    permissions: Set[Permission]
    roles: Set[UserRole]
    conditions: Dict[str, Any]  # 额外的访问条件


class RBACManager:

    """基于角色的访问控制管理器"""

    def __init__(self):

        self.users: Dict[str, User] = {}
        self.roles: Dict[UserRole, RoleDefinition] = {}
        self.policies: Dict[str, AccessPolicy] = {}
        self._lock = threading.RLock()

        # 初始化默认角色和权限
        self._init_default_roles()
        self._init_default_policies()

    def _init_default_roles(self):
        """初始化默认角色"""
        role_definitions = [
            RoleDefinition(
                role=UserRole.ADMIN,
                name="管理员",
                description="系统管理员，拥有所有权限",
                permissions=set(Permission),
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.TRADER,
                name="交易员",
                description="专业交易员，可以执行交易操作",
                permissions={
                    Permission.TRADE_EXECUTE,
                    Permission.TRADE_CANCEL,
                    Permission.ORDER_PLACE,
                    Permission.ORDER_CANCEL,
                    Permission.DATA_READ
                },
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.ANALYST,
                name="分析师",
                description="数据分析师，可以查看和导出数据",
                permissions={
                    Permission.DATA_READ,
                    Permission.DATA_EXPORT,
                    Permission.SYSTEM_MONITOR
                },
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.AUDITOR,
                name="审计员",
                description="审计员，可以查看审计日志",
                permissions={
                    Permission.AUDIT_READ,
                    Permission.AUDIT_EXPORT,
                    Permission.DATA_READ
                },
                parent_roles=set()
            ),
            RoleDefinition(
                role=UserRole.GUEST,
                name="访客",
                description="访客用户，只读权限",
                permissions={Permission.DATA_READ},
                parent_roles=set()
            )
        ]

        for role_def in role_definitions:
            self.roles[role_def.role] = role_def

    def _init_default_policies(self):
        """初始化默认访问策略"""
        policies = [
            AccessPolicy(
                policy_id="trade_api",
                name="交易API访问",
                description="交易相关API的访问控制",
                resource_pattern="/api / trade/*",
                permissions={Permission.TRADE_EXECUTE, Permission.ORDER_PLACE},
                roles={UserRole.ADMIN, UserRole.TRADER},
                conditions={}
            ),
            AccessPolicy(
                policy_id="data_api",
                name="数据API访问",
                description="数据相关API的访问控制",
                resource_pattern="/api / data/*",
                permissions={Permission.DATA_READ},
                roles={UserRole.ADMIN, UserRole.TRADER, UserRole.ANALYST, UserRole.AUDITOR},
                conditions={}
            ),
            AccessPolicy(
                policy_id="admin_api",
                name="管理API访问",
                description="系统管理API的访问控制",
                resource_pattern="/api / admin/*",
                permissions={Permission.SYSTEM_CONFIG, Permission.USER_MANAGE},
                roles={UserRole.ADMIN},
                conditions={}
            ),
            AccessPolicy(
                policy_id="audit_api",
                name="审计API访问",
                description="审计相关API的访问控制",
                resource_pattern="/api / audit/*",
                permissions={Permission.AUDIT_READ, Permission.AUDIT_EXPORT},
                roles={UserRole.ADMIN, UserRole.AUDITOR},
                conditions={}
            )
        ]

        for policy in policies:
            self.policies[policy.policy_id] = policy

    def create_user(self, user_id: str, username: str, email: str,


                    roles: Set[UserRole], password: str) -> bool:
        """创建用户"""
        with self._lock:
            if (user_id in self.users or not user_id or not username or not email
                    or "@" not in email or "." not in email):
                return False

            password_hash = self._hash_password(password)
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles,
                is_active=True,
                created_at=datetime.now(),
                password_hash=password_hash
            )

            self.users[user_id] = user
            logger.info(f"用户创建成功: {username} ({user_id})")
            return True

    def authenticate_user(self, user_id: str, password: str) -> bool:
        """用户认证"""
        with self._lock:
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return False

            # 检查账户是否被锁定
            if user.locked_until and datetime.now() < user.locked_until:
                return False

            # 验证密码
            if self._verify_password(password, user.password_hash):
                user.last_login = datetime.now()
                user.failed_login_attempts = 0
                return True
            else:
                user.failed_login_attempts += 1
                # 如果失败次数过多，锁定账户
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.now() + timedelta(minutes=30)
                return False

    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """检查用户权限"""
        with self._lock:
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return False

            # 获取用户的所有权限（包括继承的权限）
            user_permissions = self._get_user_permissions(user)

            # 将action转换为Permission枚举
            try:
                required_permission = Permission(action)
            except ValueError:
                return False

            # 检查是否有直接权限
            if required_permission in user_permissions:
                return True

            # 检查访问策略
            return self._check_policies(user, resource, required_permission)

    def _get_user_permissions(self, user: User) -> Set[Permission]:
        """获取用户的所有权限（包括角色继承）"""
        permissions = set()

        def collect_permissions(roles: Set[UserRole]):

            for role in roles:
                if role in self.roles:
                    role_def = self.roles[role]
                    permissions.update(role_def.permissions)
                    # 递归收集父角色的权限
                    collect_permissions(role_def.parent_roles)

        collect_permissions(user.roles)
        return permissions

    def _check_policies(self, user: User, resource: str, permission: Permission) -> bool:
        """检查访问策略"""
        for policy in self.policies.values():
            # 检查资源模式匹配
            if self._match_resource_pattern(resource, policy.resource_pattern):
                # 检查用户角色
                if user.roles & policy.roles:  # 集合交集
                    # 检查权限
                    if permission in policy.permission_values():
                        # 检查额外条件
                        if self._check_policy_conditions(user, policy.conditions):
                            return True

        return False

    def _match_resource_pattern(self, resource: str, pattern: str) -> bool:
        """匹配资源模式"""
        # 简单的通配符匹配
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            return resource.startswith(prefix)
        else:
            return resource == pattern

    def _check_policy_conditions(self, user: User, conditions: Dict[str, Any]) -> bool:
        """检查策略条件"""
        # 这里可以实现更复杂的条件检查逻辑
        # 目前简单返回True
        return True

    def add_role_to_user(self, user_id: str, role: UserRole) -> bool:
        """为用户添加角色"""
        with self._lock:
            user = self.users.get(user_id)
            if user:
                user.roles.add(role)
                logger.info(f"为用户 {user_id} 添加角色: {role.value}")
                return True
        return False

    def remove_role_from_user(self, user_id: str, role: UserRole) -> bool:
        """从用户移除角色"""
        with self._lock:
            user = self.users.get(user_id)
            if user:
                user.roles.discard(role)
                logger.info(f"从用户 {user_id} 移除角色: {role.value}")
                return True
        return False

    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户信息"""
        with self._lock:
            return self.users.get(user_id)

    def list_users(self) -> Dict[str, Dict[str, Any]]:
        """列出所有用户"""
        with self._lock:
            return {
                user_id: {
                    "username": user.username,
                    "email": user.email,
                    "roles": [role.value for role in user.roles],
                    "is_active": user.is_active,
                    "last_login": user.last_login.isoformat() if user.last_login else None
                }
                for user_id, user in self.users.items()
            }

    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        return self._hash_password(password) == password_hash


class SessionManager:

    """会话管理器"""

    def __init__(self):

        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.session_timeout = timedelta(hours=8)  # 8小时会话超时

    def create_session(self, user_id: str, ip_address: str = "",


                       user_agent: str = "") -> str:
        """创建会话"""
        import uuid
        session_id = str(uuid.uuid4())

        with self._lock:
            self.sessions[session_id] = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "ip_address": ip_address,
                "user_agent": user_agent,
                "is_active": True
            }

        logger.info(f"为用户 {user_id} 创建会话: {session_id}")
        return session_id

    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """验证会话"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session or not session["is_active"]:
                return None

            # 检查会话是否过期
            if datetime.now() - session["last_activity"] > self.session_timeout:
                session["is_active"] = False
                return None

            # 更新最后活动时间
            session["last_activity"] = datetime.now()
            return session

    def destroy_session(self, session_id: str):
        """销毁会话"""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]["is_active"] = False
                logger.info(f"销毁会话: {session_id}")

    def cleanup_expired_sessions(self):
        """清理过期会话"""
        with self._lock:
            expired_sessions = []
            for session_id, session in self.sessions.items():
                if (not session["is_active"]
                        or datetime.now() - session["last_activity"] > self.session_timeout):
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del self.sessions[session_id]

            if expired_sessions:
                logger.info(f"清理过期会话: {len(expired_sessions)} 个")


class AccessControlSystem:

    """访问控制系统主类"""

    def __init__(self):

        self.rbac_manager = RBACManager()
        self.session_manager = SessionManager()

        # 基础设施集成
        self._infrastructure_adapter = None
        if INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            try:
                self._infrastructure_adapter = get_trading_layer_adapter()
            except Exception as e:
                logger.warning(f"基础设施集成初始化失败: {e}")

        # 创建默认管理员用户
        self._create_default_admin()

        logger.info("访问控制系统初始化完成")

    def _create_default_admin(self):
        """创建默认管理员用户"""
        self.rbac_manager.create_user(
            user_id="admin",
            username="Administrator",
            email="admin@rqa2025.com",
            roles={UserRole.ADMIN},
            password=os.getenv("PASSWORD", "")
        )

    def authenticate(self, user_id: str, password: str) -> Optional[str]:
        """用户认证并创建会话"""
        if self.rbac_manager.authenticate_user(user_id, password):
            user = self.rbac_manager.get_user(user_id)
            if user:
                session_id = self.session_manager.create_session(
                    user_id=user_id,
                    ip_address="",  # 在实际使用中应该从请求中获取
                    user_agent=""
                )
                return session_id
        return None

    def authorize(self, session_id: str, resource: str, action: str) -> bool:
        """授权检查"""
        session = self.session_manager.validate_session(session_id)
        if not session:
            return False

        user_id = session["user_id"]
        # 组合resource和action为完整的权限字符串
        permission = f"{resource}:{action}"
        return self.rbac_manager.check_permission(user_id, resource, permission)

    def create_user(self, user_id: str, username: str, email: str,


                    roles: List[str], password: str) -> bool:
        """创建用户"""
        role_set = set()
        for role_str in roles:
            try:
                role_set.add(UserRole(role_str))
            except ValueError:
                logger.error(f"无效的角色: {role_str}")
                return False

        return self.rbac_manager.create_user(user_id, username, email, role_set, password)

    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        user = self.rbac_manager.get_user(user_id)
        if user:
            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        return None

    def update_user_roles(self, user_id: str, roles: List[str]) -> bool:
        """更新用户角色"""
        user = self.rbac_manager.get_user(user_id)
        if not user:
            return False

        # 移除所有现有角色
        for role in list(user.roles):
            self.rbac_manager.remove_role_from_user(user_id, role)

        # 添加新角色
        for role_str in roles:
            try:
                role = UserRole(role_str)
                self.rbac_manager.add_role_to_user(user_id, role)
            except ValueError:
                logger.error(f"无效的角色: {role_str}")
                return False

        return True

    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """验证会话"""
        return self.session_manager.validate_session(session_id)

    def logout(self, session_id: str):
        """用户登出"""
        self.session_manager.destroy_session(session_id)

    def list_users(self) -> Dict[str, Dict[str, Any]]:
        """列出所有用户"""
        return self.rbac_manager.list_users()

    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """检查权限（直接调用，不需要会话）"""
        permission = f"{resource}:{action}"
        return self.rbac_manager.check_permission(user_id, resource, permission)

    def cleanup_sessions(self):
        """清理过期会话"""
        self.session_manager.cleanup_expired_sessions()

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = {
            'component': 'AccessControlSystem',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'users_count': len(self.rbac_manager.users),
            'active_sessions': len([
                s for s in self.session_manager.sessions.values() if s["is_active"]
            ]),
            'warnings': [],
            'critical_issues': []
        }

        # 检查是否有活跃的管理员用户
        admin_users = [
            user_id for user_id, user in self.rbac_manager.users.items()
            if UserRole.ADMIN in user.roles and user.is_active
        ]
        if not admin_users:
            health_info['critical_issues'].append("没有活跃的管理员用户")

        # 检查会话状态
        active_sessions = len([
            s for s in self.session_manager.sessions.values() if s["is_active"]
        ])
        if active_sessions > 1000:  # 太多活跃会话
            health_info['warnings'].append(f"活跃会话数量过多: {active_sessions}")

        # 总体状态评估
        if health_info['critical_issues']:
            health_info['status'] = 'critical'
        elif health_info['warnings']:
            health_info['status'] = 'warning'

        return health_info


# 全局访问控制系统实例
_access_control_system = None
_access_control_system_lock = threading.Lock()


def get_access_control_system() -> AccessControlSystem:
    """获取全局访问控制系统实例"""
    global _access_control_system

    if _access_control_system is None:
        with _access_control_system_lock:
            if _access_control_system is None:
                _access_control_system = AccessControlSystem()

    return _access_control_system


# 便捷函数

def authenticate_user(user_id: str, password: str) -> Optional[str]:
    """用户认证"""
    system = get_access_control_system()
    return system.authenticate(user_id, password)


def check_user_permission(session_id: str, resource: str, action: str) -> bool:
    """检查用户权限"""
    system = get_access_control_system()
    return system.authorize(session_id, resource, action)


def create_system_user(user_id: str, username: str, email: str, roles: List[str], password: str) -> bool:
    """创建系统用户"""
    system = get_access_control_system()
    return system.create_user(user_id, username, email, roles, password)
