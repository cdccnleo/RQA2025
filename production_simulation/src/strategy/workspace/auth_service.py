#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
认证和授权服务
Authentication and Authorization Service

提供用户认证、角色管理和权限控制功能。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import jwt
import hashlib
import secrets

from strategy.core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


@dataclass
class User:

    """用户"""
    user_id: str
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None
    permissions: List[str] = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()
        if self.permissions is None:
            self.permissions = []


@dataclass
class Role:

    """角色"""
    role_id: str
    role_name: str
    description: str
    permissions: List[str]
    created_at: datetime = None
    is_system_role: bool = False

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Permission:

    """权限"""
    permission_id: str
    permission_name: str
    resource: str
    action: str
    description: str
    created_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AuthToken:

    """认证令牌"""
    token: str
    user_id: str
    issued_at: datetime
    expires_at: datetime
    is_revoked: bool = False


class AuthService:

    """
    认证服务
    Authentication Service

    处理用户认证、令牌管理和会话管理。
    """

    def __init__(self, secret_key: str = None, token_expiry_hours: int = 24):
        """
        初始化认证服务

        Args:
            secret_key: JWT密钥
            token_expiry_hours: 令牌过期时间（小时）
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry_hours = token_expiry_hours

        # 用户存储
        self.users: Dict[str, User] = {}

        # 令牌存储
        self.tokens: Dict[str, AuthToken] = {}

        # 会话存储
        self.sessions: Dict[str, Dict[str, Any]] = {}

        self.adapter_factory = get_unified_adapter_factory()

        logger.info("认证服务初始化完成")

    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        用户认证

        Args:
            username: 用户名
            password: 密码

        Returns:
            Optional[str]: 认证成功返回令牌，失败返回None
        """
        try:
            # 查找用户
            user = None
            for u in self.users.values():
                if u.username == username and u.is_active:
                    user = u
                    break

            if not user:
                logger.warning(f"用户认证失败: 用户不存在 - {username}")
                return None

            # 验证密码
            if not self._verify_password(password, user.password_hash):
                logger.warning(f"用户认证失败: 密码错误 - {username}")
                return None

            # 生成令牌
            token = self._generate_token(user.user_id)

            # 更新最后登录时间
            user.last_login = datetime.now()

            logger.info(f"用户认证成功: {username}")
            return token

        except Exception as e:
            logger.error(f"用户认证异常: {e}")
            return None

    async def register_user(self, username: str, email: str, password: str,
                            full_name: str = "", role: str = "user") -> Optional[str]:
        """
        用户注册

        Args:
            username: 用户名
            email: 邮箱
            password: 密码
            full_name: 全名
            role: 角色

        Returns:
            Optional[str]: 注册成功返回用户ID，失败返回None
        """
        try:
            # 检查用户名是否已存在
            if any(u.username == username for u in self.users.values()):
                logger.warning(f"用户注册失败: 用户名已存在 - {username}")
                return None

            # 检查邮箱是否已存在
            if any(u.email == email for u in self.users.values()):
                logger.warning(f"用户注册失败: 邮箱已存在 - {email}")
                return None

            # 创建用户
            user_id = f"user_{datetime.now().strftime('%Y % m % d_ % H % M % S')}_{secrets.token_hex(4)}"
            password_hash = self._hash_password(password)

            user = User(
                user_id=user_id,
                username=username,
                email=email,
                full_name=full_name,
                role=role,
                password_hash=password_hash,
                permissions=self._get_role_permissions(role)
            )

            self.users[user_id] = user

            logger.info(f"用户注册成功: {username} ({user_id})")
            return user_id

        except Exception as e:
            logger.error(f"用户注册异常: {e}")
            return None

    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证令牌

        Args:
            token: JWT令牌

        Returns:
            Optional[Dict[str, Any]]: 验证成功返回用户信息，失败返回None
        """
        try:
            # 解码JWT令牌
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            user_id = payload.get("user_id")
            exp = payload.get("exp")

            if not user_id or not exp:
                return None

            # 检查令牌是否过期
            if datetime.fromtimestamp(exp) < datetime.now():
                return None

            # 检查用户是否存在
            if user_id not in self.users:
                return None

            user = self.users[user_id]
            if not user.is_active:
                return None

            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "permissions": user.permissions
            }

        except jwt.ExpiredSignatureError:
            logger.warning("令牌已过期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("无效的令牌")
            return None
        except Exception as e:
            logger.error(f"令牌验证异常: {e}")
            return None

    async def revoke_token(self, token: str) -> bool:
        """
        撤销令牌

        Args:
            token: JWT令牌

        Returns:
            bool: 撤销是否成功
        """
        try:
            # 这里可以实现令牌黑名单机制
            # 暂时只记录日志
            logger.info(f"令牌已撤销: {token[:20]}...")
            return True

        except Exception as e:
            logger.error(f"撤销令牌异常: {e}")
            return False

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户资料

        Args:
            user_id: 用户ID

        Returns:
            Optional[Dict[str, Any]]: 用户资料
        """
        try:
            if user_id not in self.users:
                return None

            user = self.users[user_id]
            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "permissions": user.permissions
            }

        except Exception as e:
            logger.error(f"获取用户资料异常: {e}")
            return None

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新用户资料

        Args:
            user_id: 用户ID
            updates: 更新内容

        Returns:
            bool: 更新是否成功
        """
        try:
            if user_id not in self.users:
                return False

            user = self.users[user_id]

            # 更新允许的字段
            allowed_fields = ["full_name", "email"]
            for field in allowed_fields:
                if field in updates:
                    setattr(user, field, updates[field])

            logger.info(f"用户资料更新成功: {user_id}")
            return True

        except Exception as e:
            logger.error(f"更新用户资料异常: {e}")
            return False

    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """
        修改密码

        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码

        Returns:
            bool: 修改是否成功
        """
        try:
            if user_id not in self.users:
                return False

            user = self.users[user_id]

            # 验证旧密码
            if not self._verify_password(old_password, user.password_hash):
                logger.warning(f"密码修改失败: 旧密码错误 - {user_id}")
                return False

            # 更新密码
            user.password_hash = self._hash_password(new_password)

            logger.info(f"密码修改成功: {user_id}")
            return True

        except Exception as e:
            logger.error(f"修改密码异常: {e}")
            return False

    def _generate_token(self, user_id: str) -> str:
        """
        生成JWT令牌

        Args:
            user_id: 用户ID

        Returns:
            str: JWT令牌
        """
        payload = {
            "user_id": user_id,
            "iat": datetime.now(),
            "exp": datetime.now() + timedelta(hours=self.token_expiry_hours)
        }

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token

    def _hash_password(self, password: str) -> str:
        """
        密码哈希

        Args:
            password: 明文密码

        Returns:
            str: 密码哈希
        """
        # 使用SHA - 256 + 盐值进行哈希
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
        return f"{salt}:{hashed}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        验证密码

        Args:
            password: 明文密码
            password_hash: 密码哈希

        Returns:
            bool: 密码是否正确
        """
        try:
            if not password_hash or ":" not in password_hash:
                return False

            salt, hashed = password_hash.split(":", 1)
            expected_hash = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
            return hashed == expected_hash

        except Exception:
            return False

    def _get_role_permissions(self, role: str) -> List[str]:
        """
        获取角色权限

        Args:
            role: 角色名称

        Returns:
            List[str]: 权限列表
        """
        # 简化的角色权限映射
        role_permissions = {
            "admin": [
                "strategy:*",
                "backtest:*",
                "optimization:*",
                "monitoring:*",
                "debug:*",
                "user:*"
            ],
            "developer": [
                "strategy:read",
                "strategy:write",
                "backtest:*",
                "optimization:*",
                "monitoring:read",
                "debug:*"
            ],
            "analyst": [
                "strategy:read",
                "backtest:read",
                "optimization:read",
                "monitoring:read"
            ],
            "user": [
                "strategy:read",
                "backtest:read",
                "monitoring:read"
            ]
        }

        return role_permissions.get(role, [])


class AuthorizationService:

    """
    授权服务
    Authorization Service

    处理权限检查和访问控制。
    """

    def __init__(self):
        """初始化授权服务"""
        # 权限定义
        self.permissions = self._load_permissions()

        # 角色定义
        self.roles = self._load_roles()

        logger.info("授权服务初始化完成")

    def _load_permissions(self) -> Dict[str, Permission]:
        """
        加载权限定义

        Returns:
            Dict[str, Permission]: 权限字典
        """
        permissions = {}

        # 策略权限
        permissions["strategy:read"] = Permission(
            permission_id="strategy:read",
            permission_name="策略读取",
            resource="strategy",
            action="read",
            description="读取策略信息"
        )

        permissions["strategy:write"] = Permission(
            permission_id="strategy:write",
            permission_name="策略写入",
            resource="strategy",
            action="write",
            description="创建和修改策略"
        )

        permissions["strategy:delete"] = Permission(
            permission_id="strategy:delete",
            permission_name="策略删除",
            resource="strategy",
            action="delete",
            description="删除策略"
        )

        # 回测权限
        permissions["backtest:read"] = Permission(
            permission_id="backtest:read",
            permission_name="回测读取",
            resource="backtest",
            action="read",
            description="读取回测结果"
        )

        permissions["backtest:write"] = Permission(
            permission_id="backtest:write",
            permission_name="回测执行",
            resource="backtest",
            action="write",
            description="执行回测任务"
        )

        # 其他权限类似定义...
        # 这里简化为几个关键权限

        return permissions

    def _load_roles(self) -> Dict[str, Role]:
        """
        加载角色定义

        Returns:
            Dict[str, Role]: 角色字典
        """
        roles = {}

        # 管理员角色
        roles["admin"] = Role(
            role_id="admin",
            role_name="管理员",
            description="系统管理员，具有所有权限",
            permissions=["strategy:*", "backtest:*", "optimization:*",
                         "monitoring:*", "debug:*", "user:*"],
            is_system_role=True
        )

        # 开发者角色
        roles["developer"] = Role(
            role_id="developer",
            role_name="开发者",
            description="策略开发者，可以创建和管理策略",
            permissions=["strategy:*", "backtest:*",
                         "optimization:*", "monitoring:read", "debug:*"],
            is_system_role=True
        )

        # 分析师角色
        roles["analyst"] = Role(
            role_id="analyst",
            role_name="分析师",
            description="数据分析师，可以查看分析结果",
            permissions=["strategy:read", "backtest:read", "optimization:read", "monitoring:read"],
            is_system_role=True
        )

        # 普通用户角色
        roles["user"] = Role(
            role_id="user",
            role_name="普通用户",
            description="普通用户，基本查看权限",
            permissions=["strategy:read", "backtest:read", "monitoring:read"],
            is_system_role=True
        )

        return roles

    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """
        检查权限

        Args:
            user_permissions: 用户权限列表
            required_permission: 需要的权限

        Returns:
            bool: 是否有权限
        """
        # 检查通配符权限
        for permission in user_permissions:
            if permission == "*" or permission == "*:*" or required_permission.startswith(permission.split(":")[0] + ":"):
                if permission.endswith("*") or permission == required_permission:
                    return True

        return required_permission in user_permissions

    def get_user_permissions(self, role: str) -> List[str]:
        """
        获取用户权限

        Args:
            role: 用户角色

        Returns:
            List[str]: 权限列表
        """
        if role not in self.roles:
            return []

        return self.roles[role].permissions.copy()

    def get_available_permissions(self) -> List[Dict[str, Any]]:
        """
        获取可用权限

        Returns:
            List[Dict[str, Any]]: 权限列表
        """
        return [
            {
                "id": perm.permission_id,
                "name": perm.permission_name,
                "resource": perm.resource,
                "action": perm.action,
                "description": perm.description
            }
            for perm in self.permissions.values()
        ]

    def get_available_roles(self) -> List[Dict[str, Any]]:
        """
        获取可用角色

        Returns:
            List[Dict[str, Any]]: 角色列表
        """
        return [
            {
                "id": role.role_id,
                "name": role.role_name,
                "description": role.description,
                "permissions": role.permissions,
                "is_system_role": role.is_system_role
            }
            for role in self.roles.values()
        ]


class SessionManager:

    """
    会话管理器
    Session Manager

    管理用户会话和状态。
    """

    def __init__(self, session_timeout_minutes: int = 30):
        """
        初始化会话管理器

        Args:
            session_timeout_minutes: 会话超时时间（分钟）
        """
        self.session_timeout_minutes = session_timeout_minutes
        self.sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("会话管理器初始化完成")

    def create_session(self, user_id: str, user_info: Dict[str, Any]) -> str:
        """
        创建会话

        Args:
            user_id: 用户ID
            user_info: 用户信息

        Returns:
            str: 会话ID
        """
        session_id = secrets.token_hex(32)

        session = {
            "session_id": session_id,
            "user_id": user_id,
            "user_info": user_info,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "is_active": True
        }

        self.sessions[session_id] = session

        logger.info(f"会话创建成功: {session_id} (用户: {user_id})")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            Optional[Dict[str, Any]]: 会话信息
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # 检查会话是否过期
        if not self._is_session_valid(session):
            self.destroy_session(session_id)
            return None

        # 更新最后活动时间
        session["last_activity"] = datetime.now()

        return session.copy()

    def destroy_session(self, session_id: str) -> bool:
        """
        销毁会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 销毁是否成功
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"会话销毁成功: {session_id}")
            return True

        return False

    def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话

        Returns:
            int: 清理的会话数量
        """
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if not self._is_session_valid(session):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")

        return len(expired_sessions)

    def _is_session_valid(self, session: Dict[str, Any]) -> bool:
        """
        检查会话是否有效

        Args:
            session: 会话信息

        Returns:
            bool: 是否有效
        """
        if not session.get("is_active", False):
            return False

        last_activity = session.get("last_activity")
        if not last_activity:
            return False

        timeout = timedelta(minutes=self.session_timeout_minutes)
        if datetime.now() - last_activity > timeout:
            return False

        return True


# 导出
__all__ = [
    'AuthService',
    'AuthorizationService',
    'SessionManager',
    'User',
    'Role',
    'Permission',
    'AuthToken'
]
