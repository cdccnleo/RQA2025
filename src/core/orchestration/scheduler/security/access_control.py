"""
访问控制模块

提供调度器的权限验证和访问控制功能
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """权限枚举"""
    # 任务权限
    TASK_SUBMIT = "task:submit"
    TASK_CANCEL = "task:cancel"
    TASK_PAUSE = "task:pause"
    TASK_RESUME = "task:resume"
    TASK_VIEW = "task:view"
    TASK_RETRY = "task:retry"

    # 定时任务权限
    JOB_CREATE = "job:create"
    JOB_DELETE = "job:delete"
    JOB_UPDATE = "job:update"
    JOB_VIEW = "job:view"

    # 调度器权限
    SCHEDULER_START = "scheduler:start"
    SCHEDULER_STOP = "scheduler:stop"
    SCHEDULER_CONFIG = "scheduler:config"
    SCHEDULER_VIEW = "scheduler:view"

    # 告警权限
    ALERT_CONFIG = "alert:config"
    ALERT_VIEW = "alert:view"

    # 系统权限
    ADMIN = "admin:*"


class Role(Enum):
    """角色枚举"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SERVICE = "service"


# 角色权限映射
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # 管理员拥有所有权限

    Role.OPERATOR: {
        Permission.TASK_SUBMIT,
        Permission.TASK_CANCEL,
        Permission.TASK_PAUSE,
        Permission.TASK_RESUME,
        Permission.TASK_VIEW,
        Permission.TASK_RETRY,
        Permission.JOB_CREATE,
        Permission.JOB_DELETE,
        Permission.JOB_UPDATE,
        Permission.JOB_VIEW,
        Permission.SCHEDULER_START,
        Permission.SCHEDULER_STOP,
        Permission.SCHEDULER_VIEW,
        Permission.ALERT_CONFIG,
        Permission.ALERT_VIEW,
    },

    Role.VIEWER: {
        Permission.TASK_VIEW,
        Permission.JOB_VIEW,
        Permission.SCHEDULER_VIEW,
        Permission.ALERT_VIEW,
    },

    Role.SERVICE: {
        Permission.TASK_SUBMIT,
        Permission.TASK_VIEW,
        Permission.JOB_VIEW,
        Permission.SCHEDULER_VIEW,
    },
}


@dataclass
class User:
    """用户"""
    id: str
    name: str
    role: Role
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True

    def __post_init__(self):
        # 合并角色权限和个人权限
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        self.permissions = self.permissions | role_perms

    def has_permission(self, permission: Permission) -> bool:
        """检查是否有指定权限"""
        return (
            permission in self.permissions or
            Permission.ADMIN in self.permissions
        )

    def has_any_permission(self, permissions: Set[Permission]) -> bool:
        """检查是否有任一权限"""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: Set[Permission]) -> bool:
        """检查是否有所有权限"""
        return all(self.has_permission(p) for p in permissions)


@dataclass
class APIKey:
    """API密钥"""
    key_id: str
    key_hash: str
    name: str
    role: Role
    permissions: Set[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 1000  # 每分钟请求数限制

    def is_valid(self) -> bool:
        """检查API密钥是否有效"""
        if not self.is_active:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


class AccessControl:
    """
    访问控制器

    提供调度器的权限验证功能：
    - 基于角色的访问控制（RBAC）
    - API密钥管理
    - 权限验证装饰器
    - 审计日志

    使用场景：
    - API端点权限控制
    - 任务操作权限验证
    - 管理员操作审计
    """

    def __init__(self):
        """初始化访问控制器"""
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self._api_key_hashes: Dict[str, str] = {}  # key_hash -> key_id
        self._lock = threading.RLock()

        # 审计日志
        self._audit_log: List[Dict[str, Any]] = []
        self._max_audit_log_size = 10000

        # 速率限制
        self._rate_limits: Dict[str, List[datetime]] = {}  # key_id -> 请求时间列表

        # 创建默认管理员
        self._create_default_admin()

    def _create_default_admin(self):
        """创建默认管理员用户"""
        admin = User(
            id="admin",
            name="Administrator",
            role=Role.ADMIN
        )
        self._users[admin.id] = admin

    def create_user(
        self,
        user_id: str,
        name: str,
        role: Role,
        extra_permissions: Optional[Set[Permission]] = None
    ) -> User:
        """
        创建用户

        Args:
            user_id: 用户ID
            name: 用户名
            role: 角色
            extra_permissions: 额外权限

        Returns:
            User: 创建的用户
        """
        with self._lock:
            if user_id in self._users:
                raise ValueError(f"用户 {user_id} 已存在")

            user = User(
                id=user_id,
                name=name,
                role=role,
                permissions=extra_permissions or set()
            )
            self._users[user_id] = user

            self._log_audit("create_user", user_id, {"role": role.value})

            return user

    def get_user(self, user_id: str) -> Optional[User]:
        """
        获取用户

        Args:
            user_id: 用户ID

        Returns:
            Optional[User]: 用户
        """
        return self._users.get(user_id)

    def delete_user(self, user_id: str) -> bool:
        """
        删除用户

        Args:
            user_id: 用户ID

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if user_id not in self._users:
                return False

            del self._users[user_id]
            self._log_audit("delete_user", user_id)
            return True

    def create_api_key(
        self,
        name: str,
        role: Role,
        expires_days: Optional[int] = None,
        extra_permissions: Optional[Set[Permission]] = None
    ) -> tuple:
        """
        创建API密钥

        Args:
            name: 密钥名称
            role: 角色
            expires_days: 过期天数
            extra_permissions: 额外权限

        Returns:
            tuple: (key_id, 原始密钥)
        """
        with self._lock:
            # 生成密钥
            raw_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            key_id = f"ak_{secrets.token_hex(8)}"

            # 计算过期时间
            expires_at = None
            if expires_days:
                expires_at = datetime.now() + timedelta(days=expires_days)

            # 合并权限
            permissions = ROLE_PERMISSIONS.get(role, set()) | (extra_permissions or set())

            api_key = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                name=name,
                role=role,
                permissions=permissions,
                created_at=datetime.now(),
                expires_at=expires_at
            )

            self._api_keys[key_id] = api_key
            self._api_key_hashes[key_hash] = key_id

            self._log_audit("create_api_key", key_id, {"name": name, "role": role.value})

            return key_id, raw_key

    def revoke_api_key(self, key_id: str) -> bool:
        """
        撤销API密钥

        Args:
            key_id: 密钥ID

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if key_id not in self._api_keys:
                return False

            api_key = self._api_keys[key_id]
            api_key.is_active = False

            if api_key.key_hash in self._api_key_hashes:
                del self._api_key_hashes[api_key.key_hash]

            self._log_audit("revoke_api_key", key_id)
            return True

    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        验证API密钥

        Args:
            raw_key: 原始密钥

        Returns:
            Optional[APIKey]: 验证通过的API密钥
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._lock:
            key_id = self._api_key_hashes.get(key_hash)
            if not key_id:
                return None

            api_key = self._api_keys.get(key_id)
            if not api_key or not api_key.is_valid():
                return None

            # 检查速率限制
            if not self._check_rate_limit(key_id, api_key.rate_limit):
                logger.warning(f"API密钥 {key_id} 超出速率限制")
                return None

            # 更新最后使用时间
            api_key.last_used = datetime.now()

            return api_key

    def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        """
        检查速率限制

        Args:
            key_id: 密钥ID
            limit: 限制数

        Returns:
            bool: 是否通过
        """
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        if key_id not in self._rate_limits:
            self._rate_limits[key_id] = []

        # 清理过期记录
        self._rate_limits[key_id] = [
            t for t in self._rate_limits[key_id]
            if t > one_minute_ago
        ]

        # 检查是否超出限制
        if len(self._rate_limits[key_id]) >= limit:
            return False

        # 记录本次请求
        self._rate_limits[key_id].append(now)
        return True

    def check_permission(
        self,
        user_or_key: Any,
        permission: Permission
    ) -> bool:
        """
        检查权限

        Args:
            user_or_key: 用户或API密钥
            permission: 权限

        Returns:
            bool: 是否有权限
        """
        if isinstance(user_or_key, User):
            return user_or_key.has_permission(permission)

        if isinstance(user_or_key, APIKey):
            return permission in user_or_key.permissions or Permission.ADMIN in user_or_key.permissions

        return False

    def require_permission(self, permission: Permission):
        """
        权限验证装饰器

        Args:
            permission: 需要的权限

        Returns:
            decorator: 装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # 从参数中获取用户或API密钥
                user_or_key = kwargs.get('user') or kwargs.get('api_key')

                if not user_or_key:
                    raise PermissionError("缺少身份验证信息")

                if not self.check_permission(user_or_key, permission):
                    self._log_audit(
                        "permission_denied",
                        str(user_or_key),
                        {"permission": permission.value}
                    )
                    raise PermissionError(f"需要权限: {permission.value}")

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def _log_audit(self, action: str, subject: str, details: Optional[Dict] = None):
        """
        记录审计日志

        Args:
            action: 操作
            subject: 对象
            details: 详情
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "subject": subject,
            "details": details or {}
        }

        self._audit_log.append(entry)

        # 限制日志大小
        if len(self._audit_log) > self._max_audit_log_size:
            self._audit_log = self._audit_log[-self._max_audit_log_size:]

    def get_audit_log(
        self,
        limit: int = 100,
        offset: int = 0,
        action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取审计日志

        Args:
            limit: 数量限制
            offset: 偏移量
            action: 操作类型过滤

        Returns:
            List[Dict[str, Any]]: 审计日志
        """
        logs = self._audit_log

        if action:
            logs = [log for log in logs if log["action"] == action]

        return list(reversed(logs))[offset:offset + limit]

    def get_api_keys(self) -> List[Dict[str, Any]]:
        """
        获取所有API密钥

        Returns:
            List[Dict[str, Any]]: API密钥列表（不包含敏感信息）
        """
        return [
            {
                "key_id": key.key_id,
                "name": key.name,
                "role": key.role.value,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "is_active": key.is_active,
                "rate_limit": key.rate_limit
            }
            for key in self._api_keys.values()
        ]

    def get_users(self) -> List[Dict[str, Any]]:
        """
        获取所有用户

        Returns:
            List[Dict[str, Any]]: 用户列表
        """
        return [
            {
                "id": user.id,
                "name": user.name,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_active": user.is_active
            }
            for user in self._users.values()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "users_count": len(self._users),
            "api_keys_count": len(self._api_keys),
            "active_api_keys": sum(1 for k in self._api_keys.values() if k.is_valid()),
            "audit_log_size": len(self._audit_log),
            "roles": {role.value: len([u for u in self._users.values() if u.role == role]) for role in Role}
        }
