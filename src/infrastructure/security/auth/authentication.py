import os
#!/usr/bin/env python3
"""
多因素认证服务模块

提供完整的用户认证、会话管理和权限控制功能
    创建时间: 2024年12月
"""

import logging
import hashlib
import hmac
import secrets
import time
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import jwt
from enum import Enum

logger = logging.getLogger(__name__)


class AuthMethod(Enum):

    """认证方法枚举"""
    PASSWORD = os.getenv("PASSWORD", "")
    TOTP = "totp"  # Time - based One - Time Password
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = os.getenv("BIOMETRIC", "")
    HARDWARE_TOKEN = "hardware_token"


class UserRole(Enum):

    """用户角色枚举"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"


class AuthStatus(Enum):

    """认证状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    EXPIRED = "expired"
    LOCKED = "locked"
    PENDING = "pending"


@dataclass
class User:

    """用户数据类"""
    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    failed_attempts: int = 0
    locked_until: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AuthSession:

    """认证会话数据类"""
    session_id: str
    user_id: str
    token: str
    expires_at: datetime
    created_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True


@dataclass
class AuthResult:

    """认证结果数据类"""
    status: AuthStatus
    user: Optional[User] = None
    session: Optional[AuthSession] = None
    token: Optional[str] = None
    message: str = ""
    factors_completed: List[str] = None

    def __post_init__(self):

        if self.factors_completed is None:
            self.factors_completed = []


class IAuthenticator(ABC):

    """认证器接口"""

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """执行认证"""

    @abstractmethod
    def setup(self, user_id: str, config: Dict[str, Any]) -> bool:
        """设置认证方法"""

    @abstractmethod
    def verify(self, user_id: str, token: str) -> bool:
        """验证令牌"""


class PasswordAuthenticator(IAuthenticator):

    """密码认证器"""

    def __init__(self, user_store: Dict[str, str]):

        self.user_store = user_store  # user_id -> hashed_password

    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """执行密码认证"""
        user_id = credentials.get('user_id')
        password = credentials.get('password')

        if not user_id or not password:
            return AuthResult(
                status=AuthStatus.FAILED,
                message="用户名或密码不能为空"
            )

        stored_hash = self.user_store.get(user_id)
        if not stored_hash:
            return AuthResult(
                status=AuthStatus.FAILED,
                message="用户不存在"
            )

        if self._verify_password(password, stored_hash):
            return AuthResult(
                status=AuthStatus.SUCCESS,
                factors_completed=["password"],
                message="密码认证成功"
            )
        else:
            return AuthResult(
                status=AuthStatus.FAILED,
                message="密码错误"
            )

    def setup(self, user_id: str, config: Dict[str, Any]) -> bool:
        """设置密码"""
        password = config.get('password')
        if not password:
            return False

        hashed = self._hash_password(password)
        self.user_store[user_id] = hashed
        return True

    def verify(self, user_id: str, token: str) -> bool:
        """验证密码（不适用）"""
        return False

    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = secrets.token_hex(16)
        # 存储格式: salt:hash
        hash_value = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        return f"{salt}:{hash_value}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """验证密码"""
        try:
            # 解析存储的salt:hash格式
            if ':' not in stored_hash:
                # 兼容旧格式（纯SHA256）
                return hashlib.sha256(password.encode()).hexdigest() == stored_hash

            salt, hash_value = stored_hash.split(':', 1)

            # 使用相同的salt重新计算哈希
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt.encode(),
                100000
            ).hex()

            return computed_hash == hash_value
        except Exception:
            return False


class TOTPAuthenticator(IAuthenticator):

    """TOTP认证器"""

    def __init__(self, secret_store: Dict[str, str]):

        self.secret_store = secret_store  # user_id -> secret

    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """执行TOTP认证"""
        user_id = credentials.get('user_id')
        totp_code = credentials.get('totp_code')

        if not user_id or not totp_code:
            return AuthResult(
                status=AuthStatus.FAILED,
                message="用户ID或TOTP代码不能为空"
            )

        secret = self.secret_store.get(user_id)
        if not secret:
            return AuthResult(
                status=AuthStatus.FAILED,
                message="TOTP未设置"
            )

        if self._verify_totp(totp_code, secret):
            return AuthResult(
                status=AuthStatus.SUCCESS,
                factors_completed=["totp"],
                message="TOTP认证成功"
            )
        else:
            return AuthResult(
                status=AuthStatus.FAILED,
                message="TOTP代码错误"
            )

    def setup(self, user_id: str, config: Dict[str, Any]) -> bool:
        """设置TOTP密钥"""
        secret = secrets.token_hex(20)
        self.secret_store[user_id] = secret
        return True

    def verify(self, user_id: str, token: str) -> bool:
        """验证TOTP令牌"""
        secret = self.secret_store.get(user_id)
        if not secret:
            return False
        return self._verify_totp(token, secret)

    def _verify_totp(self, code: str, secret: str) -> bool:
        """验证TOTP代码"""
        # 简化的TOTP验证，实际应使用标准TOTP算法
        current_time = int(time.time() // 30)
        for i in range(-1, 2):  # 允许前后30秒的误差
            expected = self._generate_totp(secret, current_time + i)
            if expected == code:
                return True
        return False

    def _generate_totp(self, secret: str, time_step: int) -> str:
        """生成TOTP代码"""
        message = time_step.to_bytes(8, 'big')
        hmac_hash = hmac.new(secret.encode(), message, hashlib.sha1).digest()
        offset = hmac_hash[-1] & 0x0f
        code = ((hmac_hash[offset] & 0x7f) << 24
                | (hmac_hash[offset + 1] & 0xff) << 16
                | (hmac_hash[offset + 2] & 0xff) << 8
                | (hmac_hash[offset + 3] & 0xff))
        return str(code % 1000000).zfill(6)


class MultiFactorAuthenticationService:

    """多因素认证服务"""

    def __init__(self, jwt_secret: str = None):

        self.authenticators: Dict[AuthMethod, IAuthenticator] = {}
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, AuthSession] = {}
        self.jwt_secret = jwt_secret or secrets.token_hex(32)
        self.logger = logging.getLogger(__name__)

        # 初始化默认认证器
        self._init_default_authenticators()

    def _init_default_authenticators(self):
        """初始化默认认证器"""
        # 密码认证器
        password_store = {}  # 实际应从数据库加载
        self.authenticators[AuthMethod.PASSWORD] = PasswordAuthenticator(password_store)

        # TOTP认证器
        totp_store = {}  # 实际应从数据库加载
        self.authenticators[AuthMethod.TOTP] = TOTPAuthenticator(totp_store)

    def register_authenticator(self, method: AuthMethod, authenticator: IAuthenticator):
        """注册认证器"""
        self.authenticators[method] = authenticator

    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.VIEWER) -> Optional[str]:
        """创建用户"""
        # 参数验证
        self._validate_user_creation_params(username, email, password, role)

        # 检查用户唯一性
        self._check_user_uniqueness(username, email)

        # 规范化角色
        normalized_role = self._normalize_user_role(role)

        # 生成用户ID并创建用户
        return self._create_user_account(username, email, password, normalized_role)

    def _validate_user_creation_params(self, username: str, email: str, password: str, role: UserRole) -> None:
        """验证用户创建参数"""
        self._validate_username(username)
        self._validate_email(email)
        self._validate_password(password)
        self._validate_role(role)

    def _validate_username(self, username: str) -> None:
        """验证用户名"""
        if not username or not username.strip():
            raise ValueError("用户名不能为空")

        if len(username) > 50:
            raise ValueError("用户名长度不能超过50字符")

        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            raise ValueError("用户名只能包含字母、数字和下划线")

    def _validate_email(self, email: str) -> None:
        """验证邮箱"""
        if not email or not email.strip():
            raise ValueError("邮箱不能为空")

        # 邮箱格式验证
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("邮箱格式无效")

    def _validate_password(self, password: str) -> None:
        """验证密码"""
        if not password or not password.strip():
            raise ValueError("密码不能为空")

        if len(password) > 128:
            raise ValueError("密码长度不能超过128字符")

        # 密码强度检查
        if len(password) < 8:
            raise ValueError("密码强度不足")

        # 检查常见弱密码
        weak_passwords = ["123", "abc", "password", "qwerty", "123456", "admin", "root"]
        if password.lower() in weak_passwords:
            raise ValueError("密码强度不足")

    def _validate_role(self, role: UserRole) -> None:
        """验证角色"""
        if role is not None and role not in [UserRole.ADMIN, UserRole.TRADER, UserRole.ANALYST, UserRole.VIEWER]:
            raise ValueError("无效的用户角色")

    def _check_user_uniqueness(self, username: str, email: str) -> None:
        """检查用户唯一性"""
        for existing_user in self.users.values():
            if existing_user.username == username:
                raise ValueError("用户名已存在")
            if existing_user.email == email:
                raise ValueError("邮箱已被使用")

    def _normalize_user_role(self, role: UserRole) -> UserRole:
        """规范化用户角色"""
        return role if role is not None else UserRole.VIEWER

    def _create_user_account(self, username: str, email: str, password: str, role: UserRole) -> Optional[str]:
        """创建用户账户"""
        user_id = secrets.token_hex(16)

        # 设置密码
        password_auth = self.authenticators.get(AuthMethod.PASSWORD)
        if password_auth and password_auth.setup(user_id, {'password': password}):
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                role=role
            )
            self.users[user_id] = user
            self.logger.info(f"用户创建成功: {username} ({user_id})")
            return user_id

        return None

    def authenticate_user(self, username: str, credentials: Dict[str, Any], required_factors: List[AuthMethod] = None) -> AuthResult:
        """认证用户"""
        if required_factors is None:
            required_factors = [AuthMethod.PASSWORD]

        # 查找用户
        user = None
        for u in self.users.values():
            if u.username == username or u.email == username:
                user = u
                break

        if not user:
            return AuthResult(
                status=AuthStatus.FAILED,
                message="用户不存在"
            )

        # 检查用户状态
        if not user.is_active:
            return AuthResult(
                status=AuthStatus.LOCKED,
                message="用户已被禁用"
            )

        # 执行多因素认证
        completed_factors = []
        for factor in required_factors:
            authenticator = self.authenticators.get(factor)
            if not authenticator:
                continue

            # 合并用户ID到凭据
            auth_credentials = {**credentials, 'user_id': user.user_id}

            result = authenticator.authenticate(auth_credentials)
            if result.status == AuthStatus.SUCCESS:
                completed_factors.extend(result.factors_completed)
            else:
                return AuthResult(
                    status=AuthStatus.FAILED,
                    message=f"{factor.value}认证失败: {result.message}"
                )

        # 认证成功，创建会话
        session = self._create_session(user.user_id, "127.0.0.1", "User - Agent")
        token = self._generate_jwt_token(user, session)

        # 更新用户最后登录时间
        user.last_login = datetime.now()

        return AuthResult(
            status=AuthStatus.SUCCESS,
            user=user,
            session=session,
            token=token,
            factors_completed=completed_factors,
            message="多因素认证成功"
        )

    def verify_token(self, token: str) -> Optional[User]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])

            user_id = payload.get('user_id')
            session_id = payload.get('session_id')

            # 验证会话
            session = self.sessions.get(session_id)
            if not session or session.user_id != user_id or not session.is_active:
                return None

            # 检查过期
            if datetime.now() > session.expires_at:
                session.is_active = False
                return None

            return self.users.get(user_id)

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def setup_mfa(self, user_id: str, method: AuthMethod, config: Dict[str, Any]) -> bool:
        """设置多因素认证"""
        authenticator = self.authenticators.get(method)
        if not authenticator:
            return False

        return authenticator.setup(user_id, config)

    def get_totp_secret(self, user_id: str) -> Optional[str]:
        """获取用户的TOTP密钥"""
        totp_auth = self.authenticators.get(AuthMethod.TOTP)
        if totp_auth and hasattr(totp_auth, 'secret_store'):
            return totp_auth.secret_store.get(user_id)
        return None

    def generate_current_totp(self, user_id: str) -> Optional[str]:
        """生成用户当前的TOTP代码"""
        secret = self.get_totp_secret(user_id)
        if not secret:
            return None
        current_time = int(time.time() // 30)
        totp_auth = self.authenticators.get(AuthMethod.TOTP)
        if totp_auth and hasattr(totp_auth, '_generate_totp'):
            return totp_auth._generate_totp(secret, current_time)
        return None

    def logout(self, token: str) -> bool:
        """用户登出"""
        if not token:
            return False

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            session_id = payload.get('session_id')

            if session_id and session_id in self.sessions:
                # 从会话字典中完全移除会话
                del self.sessions[session_id]
                return True

        except Exception as e:
            # 记录异常但不抛出
            self.logger.warning(f"登出时发生异常: {e}")

        return False

    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> AuthSession:
        """创建认证会话"""
        session_id = secrets.token_hex(16)
        now = datetime.now()

        session = AuthSession(
            session_id=session_id,
            user_id=user_id,
            token="",  # 稍后设置
            expires_at=now + timedelta(hours=8),  # 8小时过期
            created_at=now,
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.sessions[session_id] = session
        return session

    def _generate_jwt_token(self, user: User, session: AuthSession) -> str:
        """生成JWT令牌"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role.value,
            'session_id': session.session_id,
            'exp': int(session.expires_at.timestamp()),
            'iat': int(session.created_at.timestamp())
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        session.token = token
        return token


class AuthorizationService:

    """授权服务"""

    def __init__(self, auth_service: MultiFactorAuthenticationService):

        self.auth_service = auth_service
        self.permissions: Dict[UserRole, List[str]] = self._init_permissions()

    def _init_permissions(self) -> Dict[UserRole, List[str]]:
        """初始化权限配置"""
        return {
            UserRole.ADMIN: [
                "data:view", "data:create", "data:update", "data:delete",
                "user:manage", "system:configure", "report:generate",
                "trading:all", "risk:manage", "audit:view"
            ],
            UserRole.TRADER: [
                "data:view", "data:create", "trading:execute", "trading:view", "position:manage",
                "order:manage", "risk:view"
            ],
            UserRole.ANALYST: [
                "data:view", "report:view", "report:generate", "analysis:run",
                "analysis:create", "model:view", "backtest:run"
            ],
            UserRole.VIEWER: [
                "data:view", "report:view", "dashboard:view"
            ]
        }

    def check_permission(self, token: str, permission: str) -> bool:
        """检查权限"""
        user = self.auth_service.verify_token(token)
        if not user:
            return False

        role_permissions = self.permissions.get(user.role, [])
        return permission in role_permissions

    def get_user_permissions(self, token: str) -> List[str]:
        """获取用户权限"""
        user = self.auth_service.verify_token(token)
        if not user:
            return []

        return self.permissions.get(user.role, [])


# 使用示例
if __name__ == "__main__":
    # 初始化认证服务
    auth_service = MultiFactorAuthenticationService()

    # 创建用户（密码应从环境变量获取）
    demo_password = os.getenv("DEMO_PASSWORD", "CHANGE_ME_IN_PRODUCTION")
    user_id = auth_service.create_user(
        username="trader001",
        email=os.getenv("EMAIL", ""),
        password=demo_password,
        role=UserRole.TRADER
    )

    if user_id:
        print(f"用户创建成功: {user_id}")

        # 设置TOTP
        auth_service.setup_mfa(user_id, AuthMethod.TOTP, {})

        # 执行多因素认证（密码应从环境变量获取）
        result = auth_service.authenticate_user(
            "trader001",
            {
                "password": demo_password,
                "totp_code": "123456"  # 实际应从TOTP应用获取
            },
            required_factors=[AuthMethod.PASSWORD, AuthMethod.TOTP]
        )

        print(f"认证结果: {result.status.value} - {result.message}")

        if result.status == AuthStatus.SUCCESS:
            print(f"JWT令牌: {result.token}")

            # 验证令牌
            user = auth_service.verify_token(result.token)
            if user:
                print(f"令牌验证成功: {user.username}")

                # 授权服务
                authz_service = AuthorizationService(auth_service)

                # 检查权限
                has_permission = authz_service.check_permission(result.token, "trading:execute")
                print(f"交易执行权限: {has_permission}")

    print("多因素认证服务演示完成")
