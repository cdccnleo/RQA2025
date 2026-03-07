#!/usr/bin/env python3
"""
RQA2025 JWT认证系统
提供安全的令牌认证机制
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional
import secrets
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UserCredentials:
    """用户凭据数据类"""
    username: str
    password_hash: str
    roles: list
    is_active: bool = True
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    last_login: Optional[datetime] = None


@dataclass
class TokenPair:
    """令牌对"""
    access_token: str
    refresh_token: str
    expires_in: int


class JWTAuthticator:
    """JWT认证器"""

    def __init__(self,
                 secret_key: str = None,
                 algorithm: str = "HS256",
                 access_token_expire_minutes: int = 30,
                 refresh_token_expire_days: int = 7,
                 max_failed_attempts: int = 5,
                 lockout_duration_minutes: int = 15):
        """
        初始化JWT认证器

        Args:
            secret_key: JWT密钥
            algorithm: 加密算法
            access_token_expire_minutes: 访问令牌过期时间(分钟)
            refresh_token_expire_days: 刷新令牌过期时间(天)
            max_failed_attempts: 最大失败尝试次数
            lockout_duration_minutes: 锁定持续时间(分钟)
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_minutes = lockout_duration_minutes

        # 用户存储 (生产环境中应该使用数据库)
        self._users: Dict[str, UserCredentials] = {}

        # 黑名单令牌存储
        self._token_blacklist: set = set()

        logger.info("JWT认证器初始化完成")

    def _generate_secret_key(self) -> str:
        """生成随机密钥"""
        return secrets.token_hex(32)

    def register_user(self, username: str, password: str, roles: list = None) -> bool:
        """
        注册新用户

        Args:
            username: 用户名
            password: 密码
            roles: 用户角色列表

        Returns:
            注册是否成功
        """
        if username in self._users:
            logger.warning(f"用户 {username} 已存在")
            return False

        # 密码哈希
        password_hash = self._hash_password(password)

        # 创建用户凭据
        user = UserCredentials(
            username=username,
            password_hash=password_hash,
            roles=roles or ["user"],
            is_active=True
        )

        self._users[username] = user
        logger.info(f"用户 {username} 注册成功")
        return True

    def authenticate(self, username: str, password: str) -> Optional[TokenPair]:
        """
        用户认证

        Args:
            username: 用户名
            password: 密码

        Returns:
            令牌对或None
        """
        user = self._users.get(username)
        if not user:
            logger.warning(f"用户 {username} 不存在")
            return None

        # 检查账户状态
        if not user.is_active:
            logger.warning(f"用户 {username} 账户已禁用")
            return None

        # 检查账户锁定
        if user.locked_until and datetime.now() < user.locked_until:
            logger.warning(f"用户 {username} 账户已锁定")
            return None

        # 验证密码
        if not self._verify_password(password, user.password_hash):
            user.failed_attempts += 1

            # 检查是否需要锁定账户
            if user.failed_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.now() + timedelta(minutes=self.lockout_duration_minutes)
                logger.warning(f"用户 {username} 账户因多次失败尝试而锁定")

            logger.warning(f"用户 {username} 密码验证失败")
            return None

        # 认证成功，重置失败计数
        user.failed_attempts = 0
        user.last_login = datetime.now()

        # 生成令牌
        token_pair = self._generate_token_pair(username, user.roles)
        logger.info(f"用户 {username} 认证成功")
        return token_pair

    def refresh_token(self, refresh_token: str) -> Optional[TokenPair]:
        """
        刷新访问令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            新的令牌对或None
        """
        try:
            # 验证刷新令牌
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])

            # 检查令牌类型
            if payload.get("type") != "refresh":
                logger.warning("无效的刷新令牌类型")
                return None

            username = payload.get("sub")
            roles = payload.get("roles", [])

            # 检查用户是否仍然有效
            user = self._users.get(username)
            if not user or not user.is_active:
                logger.warning(f"用户 {username} 不再有效")
                return None

            # 生成新的令牌对
            token_pair = self._generate_token_pair(username, roles)
            logger.info(f"用户 {username} 令牌刷新成功")
            return token_pair

        except jwt.ExpiredSignatureError:
            logger.warning("刷新令牌已过期")
        except jwt.InvalidTokenError:
            logger.warning("无效的刷新令牌")

        return None

    def validate_token(self, token: str) -> Optional[Dict]:
        """
        验证访问令牌

        Args:
            token: 访问令牌

        Returns:
            令牌载荷或None
        """
        try:
            # 检查是否在黑名单中
            if token in self._token_blacklist:
                logger.warning("令牌已被吊销")
                return None

            # 验证令牌
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # 检查令牌类型
            if payload.get("type") != "access":
                logger.warning("无效的访问令牌类型")
                return None

            # 检查用户是否仍然有效
            username = payload.get("sub")
            user = self._users.get(username)
            if not user or not user.is_active:
                logger.warning(f"用户 {username} 不再有效")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("访问令牌已过期")
        except jwt.InvalidTokenError:
            logger.warning("无效的访问令牌")

        return None

    def revoke_token(self, token: str) -> bool:
        """
        吊销令牌

        Args:
            token: 要吊销的令牌

        Returns:
            吊销是否成功
        """
        try:
            # 验证令牌有效性
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")

            # 添加到黑名单
            self._token_blacklist.add(token)
            logger.info(f"用户 {username} 的令牌已吊销")
            return True

        except jwt.InvalidTokenError:
            logger.warning("尝试吊销无效令牌")
            return False

    def get_user_info(self, username: str) -> Optional[Dict]:
        """
        获取用户信息

        Args:
            username: 用户名

        Returns:
            用户信息字典或None
        """
        user = self._users.get(username)
        if not user:
            return None

        return {
            "username": user.username,
            "roles": user.roles,
            "is_active": user.is_active,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "failed_attempts": user.failed_attempts
        }

    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def _generate_token_pair(self, username: str, roles: list) -> TokenPair:
        """生成令牌对"""
        now = datetime.utcnow()

        # 访问令牌
        access_payload = {
            "sub": username,
            "roles": roles,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=self.access_token_expire_minutes)
        }

        # 刷新令牌
        refresh_payload = {
            "sub": username,
            "roles": roles,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(days=self.refresh_token_expire_days)
        }

        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60
        )

    def cleanup_expired_tokens(self):
        """清理过期的黑名单令牌"""
        # 在生产环境中，这个方法应该定期清理过期的令牌
        # 这里简化实现


# 全局认证器实例
authenticator = JWTAuthticator()


def init_auth_system():
    """初始化认证系统"""
    # 创建默认管理员用户
    authenticator.register_user(
        username="admin",
        password="Admin@123",
        roles=["admin", "user"]
    )

    # 创建普通用户
    authenticator.register_user(
        username="trader",
        password="Trader@123",
        roles=["user", "trader"]
    )

    logger.info("认证系统初始化完成")


if __name__ == "__main__":
    # 初始化认证系统
    init_auth_system()

    # 测试认证
    print("🔐 测试JWT认证系统")
    print("=" * 50)

    # 测试用户注册
    print("✅ 用户注册测试")
    print(f"管理员用户: admin / Admin@123")
    print(f"交易用户: trader / Trader@123")

    # 测试认证
    print("\n🔑 测试认证")
    token_pair = authenticator.authenticate("admin", "Admin@123")
    if token_pair:
        print("✅ 管理员认证成功")
        print(f"访问令牌: {token_pair.access_token[:50]}...")
        print(f"刷新令牌: {token_pair.refresh_token[:50]}...")
        print(f"过期时间: {token_pair.expires_in}秒")

        # 测试令牌验证
        print("\n🔍 测试令牌验证")
        payload = authenticator.validate_token(token_pair.access_token)
        if payload:
            print("✅ 令牌验证成功")
            print(f"用户: {payload['sub']}")
            print(f"角色: {payload['roles']}")
        else:
            print("❌ 令牌验证失败")
    else:
        print("❌ 认证失败")

    print("\n🎉 JWT认证系统测试完成")
