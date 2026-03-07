#!/usr/bin/env python3
"""
多因素认证服务单元测试

测试MultiFactorAuthenticationService及其组件的完整功能
    创建时间: 2024年12月
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import jwt
import hashlib
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.infrastructure.security.auth.authentication import (
    MultiFactorAuthenticationService,
    PasswordAuthenticator,
    TOTPAuthenticator,
    AuthorizationService,
    AuthMethod,
    AuthStatus,
    UserRole,
    User,
    AuthSession,
    AuthResult,
    IAuthenticator
)


class TestMultiFactorAuthenticationService:
    """多因素认证服务测试类"""

    def setup_method(self):
        """测试前准备"""
        self.auth_service = MultiFactorAuthenticationService("test_jwt_secret")

    def test_init_default_authenticators(self):
        """测试默认认证器初始化"""
        assert AuthMethod.PASSWORD in self.auth_service.authenticators
        assert AuthMethod.TOTP in self.auth_service.authenticators
        assert isinstance(self.auth_service.authenticators[AuthMethod.PASSWORD], PasswordAuthenticator)
        assert isinstance(self.auth_service.authenticators[AuthMethod.TOTP], TOTPAuthenticator)

    def test_register_authenticator(self):
        """测试注册认证器"""
        mock_auth = MagicMock(spec=IAuthenticator)
        self.auth_service.register_authenticator(AuthMethod.SMS, mock_auth)
        assert self.auth_service.authenticators[AuthMethod.SMS] is mock_auth

    def test_create_user_success(self):
        """测试用户创建成功"""
        user_id = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!",
            role=UserRole.TRADER
        )

        assert user_id is not None
        assert user_id in self.auth_service.users
        user = self.auth_service.users[user_id]
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.TRADER
        assert user.is_active is True

    def test_create_user_validation_errors(self):
        """测试用户创建参数验证"""
        # 空用户名
        with pytest.raises(ValueError, match="用户名不能为空"):
            self.auth_service.create_user("", "test@example.com", "password", UserRole.VIEWER)

        # 无效邮箱
        with pytest.raises(ValueError, match="邮箱格式无效"):
            self.auth_service.create_user("testuser", "invalid-email", "password", UserRole.VIEWER)

        # 弱密码
        with pytest.raises(ValueError, match="密码强度不足"):
            self.auth_service.create_user("testuser", "test@example.com", "123", UserRole.VIEWER)

        # 用户名已存在
        self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.VIEWER)
        with pytest.raises(ValueError, match="用户名已存在"):
            self.auth_service.create_user("testuser", "test2@example.com", "ValidPass123!", UserRole.VIEWER)

    def test_create_user_duplicate_email(self):
        """测试邮箱重复"""
        self.auth_service.create_user("user1", "test@example.com", "ValidPass123!", UserRole.VIEWER)
        with pytest.raises(ValueError, match="邮箱已被使用"):
            self.auth_service.create_user("user2", "test@example.com", "ValidPass123!", UserRole.VIEWER)

    def test_create_user_invalid_role(self):
        """测试无效角色"""
        with pytest.raises(ValueError, match="无效的用户角色"):
            self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", "invalid_role")

    def test_create_user_default_role(self):
        """测试默认角色"""
        user_id = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!"
        )
        user = self.auth_service.users[user_id]
        assert user.role == UserRole.VIEWER

    def test_authenticate_user_success_password_only(self):
        """测试密码认证成功"""
        # 创建用户
        user_id = self.auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="ValidPass123!",
            role=UserRole.TRADER
        )

        # 认证
        result = self.auth_service.authenticate_user(
            "testuser",
            {"password": "ValidPass123!"},
            [AuthMethod.PASSWORD]
        )

        assert result.status == AuthStatus.SUCCESS
        assert result.user is not None
        assert result.session is not None
        assert result.token is not None
        assert "password" in result.factors_completed

    def test_authenticate_user_wrong_password(self):
        """测试密码错误"""
        self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)

        result = self.auth_service.authenticate_user(
            "testuser",
            {"password": "wrong_password"},
            [AuthMethod.PASSWORD]
        )

        assert result.status == AuthStatus.FAILED
        assert "密码错误" in result.message

    def test_authenticate_user_user_not_found(self):
        """测试用户不存在"""
        result = self.auth_service.authenticate_user(
            "nonexistent",
            {"password": "password"},
            [AuthMethod.PASSWORD]
        )

        assert result.status == AuthStatus.FAILED
        assert "用户不存在" in result.message

    def test_authenticate_user_inactive(self):
        """测试用户被禁用"""
        user_id = self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)
        self.auth_service.users[user_id].is_active = False

        result = self.auth_service.authenticate_user(
            "testuser",
            {"password": "ValidPass123!"},
            [AuthMethod.PASSWORD]
        )

        assert result.status == AuthStatus.LOCKED
        assert "用户已被禁用" in result.message

    def test_authenticate_user_mfa_success(self):
        """测试多因素认证成功"""
        # 创建用户
        user_id = self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)

        # 设置TOTP
        self.auth_service.setup_mfa(user_id, AuthMethod.TOTP, {})

        # 获取当前TOTP代码
        current_totp = self.auth_service.generate_current_totp(user_id)

        # 多因素认证
        result = self.auth_service.authenticate_user(
            "testuser",
            {
                "password": "ValidPass123!",
                "totp_code": current_totp
            },
            [AuthMethod.PASSWORD, AuthMethod.TOTP]
        )

        assert result.status == AuthStatus.SUCCESS
        assert len(result.factors_completed) == 2
        assert "password" in result.factors_completed
        assert "totp" in result.factors_completed

    def test_authenticate_user_mfa_partial_failure(self):
        """测试多因素认证部分失败"""
        user_id = self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)

        result = self.auth_service.authenticate_user(
            "testuser",
            {"password": "ValidPass123!", "totp_code": "invalid"},
            [AuthMethod.PASSWORD, AuthMethod.TOTP]
        )

        assert result.status == AuthStatus.FAILED
        assert "totp认证失败" in result.message

    def test_verify_token_valid(self):
        """测试有效令牌验证"""
        user_id = self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)

        # 创建会话并生成令牌
        session = self.auth_service._create_session(user_id, "127.0.0.1", "Test Agent")
        token = self.auth_service._generate_jwt_token(self.auth_service.users[user_id], session)

        # 验证令牌
        verified_user = self.auth_service.verify_token(token)
        assert verified_user is not None
        assert verified_user.user_id == user_id

    def test_verify_token_expired(self):
        """测试过期令牌验证"""
        user_id = self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)

        # 创建过期会话
        session = AuthSession(
            session_id="test_session",
            user_id=user_id,
            token="",
            expires_at=datetime.now() - timedelta(hours=1),  # 已过期
            created_at=datetime.now() - timedelta(hours=2),
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        self.auth_service.sessions[session.session_id] = session

        # 生成过期令牌
        token = self.auth_service._generate_jwt_token(self.auth_service.users[user_id], session)

        # 验证应失败
        verified_user = self.auth_service.verify_token(token)
        assert verified_user is None

    def test_verify_token_invalid(self):
        """测试无效令牌验证"""
        verified_user = self.auth_service.verify_token("invalid_token")
        assert verified_user is None

    def test_setup_mfa_totp(self):
        """测试TOTP MFA设置"""
        user_id = self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)

        success = self.auth_service.setup_mfa(user_id, AuthMethod.TOTP, {})
        assert success is True

        secret = self.auth_service.get_totp_secret(user_id)
        assert secret is not None
        assert len(secret) == 40  # 20字节的十六进制

    def test_get_totp_secret_no_setup(self):
        """测试获取未设置的TOTP密钥"""
        secret = self.auth_service.get_totp_secret("nonexistent")
        assert secret is None

    def test_generate_current_totp_no_secret(self):
        """测试生成不存在用户的TOTP"""
        code = self.auth_service.generate_current_totp("nonexistent")
        assert code is None

    def test_logout_success(self):
        """测试成功登出"""
        user_id = self.auth_service.create_user("testuser", "test@example.com", "ValidPass123!", UserRole.TRADER)
        session = self.auth_service._create_session(user_id, "127.0.0.1", "Test Agent")
        token = self.auth_service._generate_jwt_token(self.auth_service.users[user_id], session)

        success = self.auth_service.logout(token)
        assert success is True
        assert session.session_id not in self.auth_service.sessions

    def test_logout_invalid_token(self):
        """测试无效令牌登出"""
        success = self.auth_service.logout("invalid_token")
        assert success is False

    def test_create_session(self):
        """测试会话创建"""
        user_id = "test_user_id"
        session = self.auth_service._create_session(user_id, "192.168.1.1", "Chrome/90")

        assert session.user_id == user_id
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Chrome/90"
        assert session.is_active is True
        assert session.session_id in self.auth_service.sessions
        # 检查过期时间（8小时）
        expected_expiry = session.created_at + timedelta(hours=8)
        assert abs((session.expires_at - expected_expiry).total_seconds()) < 1

    def test_generate_jwt_token(self):
        """测试JWT令牌生成"""
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            role=UserRole.ADMIN
        )
        session = AuthSession(
            session_id="test_session",
            user_id="test_user",
            token="",
            expires_at=datetime.now() + timedelta(hours=1),
            created_at=datetime.now(),
            ip_address="127.0.0.1",
            user_agent="Test"
        )

        token = self.auth_service._generate_jwt_token(user, session)

        # 解码并验证payload
        payload = jwt.decode(token, "test_jwt_secret", algorithms=['HS256'])
        assert payload['user_id'] == user.user_id
        assert payload['username'] == user.username
        assert payload['role'] == user.role.value
        assert payload['session_id'] == session.session_id
        assert 'exp' in payload
        assert 'iat' in payload

        # 检查session.token是否被设置
        assert session.token == token


class TestPasswordAuthenticator:
    """密码认证器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.user_store = {}
        self.auth = PasswordAuthenticator(self.user_store)

    def test_authenticate_success(self):
        """测试密码认证成功"""
        # 设置密码
        self.auth.setup("user1", {"password": "testpass123"})

        # 认证
        result = self.auth.authenticate({
            "user_id": "user1",
            "password": "testpass123"
        })

        assert result.status == AuthStatus.SUCCESS
        assert "password" in result.factors_completed

    def test_authenticate_wrong_password(self):
        """测试密码错误"""
        self.auth.setup("user1", {"password": "testpass123"})

        result = self.auth.authenticate({
            "user_id": "user1",
            "password": "wrongpass"
        })

        assert result.status == AuthStatus.FAILED
        assert "密码错误" in result.message

    def test_authenticate_user_not_found(self):
        """测试用户不存在"""
        result = self.auth.authenticate({
            "user_id": "nonexistent",
            "password": "password"
        })

        assert result.status == AuthStatus.FAILED
        assert "用户不存在" in result.message

    def test_authenticate_missing_credentials(self):
        """测试缺少凭据"""
        result = self.auth.authenticate({})
        assert result.status == AuthStatus.FAILED
        assert "用户名或密码不能为空" in result.message

    def test_setup_password(self):
        """测试密码设置"""
        success = self.auth.setup("user1", {"password": "newpass123"})
        assert success is True
        assert "user1" in self.user_store

    def test_setup_missing_password(self):
        """测试缺少密码设置"""
        success = self.auth.setup("user1", {})
        assert success is False

    def test_verify_password_legacy_format(self):
        """测试旧格式密码验证"""
        # 直接设置旧格式哈希
        self.user_store["user1"] = hashlib.sha256("testpass".encode()).hexdigest()

        result = self.auth.authenticate({
            "user_id": "user1",
            "password": "testpass"
        })

        assert result.status == AuthStatus.SUCCESS

    def test_hash_password_format(self):
        """测试密码哈希格式"""
        hashed = self.auth._hash_password("testpass")
        assert ":" in hashed
        salt, hash_value = hashed.split(":", 1)
        assert len(salt) == 32  # 16字节的十六进制
        assert len(hash_value) == 64  # SHA256的十六进制长度

    def test_verify_method_not_implemented(self):
        """测试verify方法未实现"""
        result = self.auth.verify("user1", "token")
        assert result is False


class TestTOTPAuthenticator:
    """TOTP认证器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.secret_store = {}
        self.auth = TOTPAuthenticator(self.secret_store)

    def test_setup_totp(self):
        """测试TOTP设置"""
        success = self.auth.setup("user1", {})
        assert success is True
        assert "user1" in self.secret_store
        assert len(self.secret_store["user1"]) == 40  # 20字节十六进制

    def test_authenticate_totp_success(self):
        """测试TOTP认证成功"""
        # 设置密钥
        self.auth.setup("user1", {})

        # 获取当前时间窗口的TOTP
        current_time = int(time.time() // 30)
        totp_code = self.auth._generate_totp(self.secret_store["user1"], current_time)

        result = self.auth.authenticate({
            "user_id": "user1",
            "totp_code": totp_code
        })

        assert result.status == AuthStatus.SUCCESS
        assert "totp" in result.factors_completed

    def test_authenticate_totp_wrong_code(self):
        """测试TOTP错误代码"""
        self.auth.setup("user1", {})

        result = self.auth.authenticate({
            "user_id": "user1",
            "totp_code": "000000"
        })

        assert result.status == AuthStatus.FAILED
        assert "TOTP代码错误" in result.message

    def test_authenticate_totp_no_secret(self):
        """测试TOTP未设置"""
        result = self.auth.authenticate({
            "user_id": "user1",
            "totp_code": "123456"
        })

        assert result.status == AuthStatus.FAILED
        assert "TOTP未设置" in result.message

    def test_authenticate_totp_missing_code(self):
        """测试缺少TOTP代码"""
        result = self.auth.authenticate({"user_id": "user1"})
        assert result.status == AuthStatus.FAILED
        assert "用户ID或TOTP代码不能为空" in result.message

    def test_verify_totp_time_window(self):
        """测试TOTP时间窗口"""
        self.auth.setup("user1", {})
        secret = self.secret_store["user1"]

        current_time = int(time.time() // 30)

        # 测试当前时间、前一时间、后一时间
        for offset in [-1, 0, 1]:
            code = self.auth._generate_totp(secret, current_time + offset)
            assert self.auth._verify_totp(code, secret)

        # 测试超出窗口的时间
        code = self.auth._generate_totp(secret, current_time + 2)
        assert not self.auth._verify_totp(code, secret)

    def test_generate_totp_format(self):
        """测试TOTP代码格式"""
        secret = "a" * 40  # 40字符十六进制
        code = self.auth._generate_totp(secret, 123456)
        assert isinstance(code, str)
        assert len(code) == 6
        assert code.isdigit()

    def test_verify_totp_method(self):
        """测试verify方法"""
        self.auth.setup("user1", {})
        secret = self.secret_store["user1"]

        current_time = int(time.time() // 30)
        code = self.auth._generate_totp(secret, current_time)

        assert self.auth.verify("user1", code)
        assert not self.auth.verify("user1", "000000")


class TestAuthorizationService:
    """授权服务测试类"""

    def setup_method(self):
        """测试前准备"""
        self.auth_service = MultiFactorAuthenticationService()
        self.authz_service = AuthorizationService(self.auth_service)

    def test_check_permission_valid_token(self):
        """测试有效令牌权限检查"""
        # 创建用户并认证
        user_id = self.auth_service.create_user("admin", "admin@example.com", "AdminPass123!", UserRole.ADMIN)
        result = self.auth_service.authenticate_user("admin", {"password": "AdminPass123!"}, [AuthMethod.PASSWORD])

        # 检查管理员权限
        has_permission = self.authz_service.check_permission(result.token, "user:manage")
        assert has_permission is True

        # 检查不存在的权限
        has_permission = self.authz_service.check_permission(result.token, "nonexistent:permission")
        assert has_permission is False

    def test_check_permission_invalid_token(self):
        """测试无效令牌权限检查"""
        has_permission = self.authz_service.check_permission("invalid_token", "user:manage")
        assert has_permission is False

    def test_get_user_permissions_admin(self):
        """测试获取管理员权限"""
        user_id = self.auth_service.create_user("admin", "admin@example.com", "AdminPass123!", UserRole.ADMIN)
        result = self.auth_service.authenticate_user("admin", {"password": "AdminPass123!"}, [AuthMethod.PASSWORD])

        permissions = self.authz_service.get_user_permissions(result.token)
        expected_permissions = [
            "data:view", "data:create", "data:update", "data:delete",
            "user:manage", "system:configure", "report:generate",
            "trading:all", "risk:manage", "audit:view"
        ]
        assert set(permissions) == set(expected_permissions)

    def test_get_user_permissions_trader(self):
        """测试获取交易员权限"""
        user_id = self.auth_service.create_user("trader", "trader@example.com", "TraderPass123!", UserRole.TRADER)
        result = self.auth_service.authenticate_user("trader", {"password": "TraderPass123!"}, [AuthMethod.PASSWORD])

        permissions = self.authz_service.get_user_permissions(result.token)
        assert "trading:execute" in permissions
        assert "user:manage" not in permissions

    def test_get_user_permissions_analyst(self):
        """测试获取分析师权限"""
        user_id = self.auth_service.create_user("analyst", "analyst@example.com", "AnalystPass123!", UserRole.ANALYST)
        result = self.auth_service.authenticate_user("analyst", {"password": "AnalystPass123!"}, [AuthMethod.PASSWORD])

        permissions = self.authz_service.get_user_permissions(result.token)
        assert "analysis:run" in permissions
        assert "trading:execute" not in permissions

    def test_get_user_permissions_viewer(self):
        """测试获取查看者权限"""
        user_id = self.auth_service.create_user("viewer", "viewer@example.com", "ViewerPass123!", UserRole.VIEWER)
        result = self.auth_service.authenticate_user("viewer", {"password": "ViewerPass123!"}, [AuthMethod.PASSWORD])

        permissions = self.authz_service.get_user_permissions(result.token)
        assert "data:view" in permissions
        assert len(permissions) == 3  # data:view, report:view, dashboard:view

    def test_get_user_permissions_invalid_token(self):
        """测试无效令牌获取权限"""
        permissions = self.authz_service.get_user_permissions("invalid_token")
        assert permissions == []


class TestDataClasses:
    """数据类测试"""

    def test_user_creation(self):
        """测试User创建"""
        user = User(
            user_id="test_id",
            username="testuser",
            email="test@example.com",
            role=UserRole.TRADER
        )
        assert user.user_id == "test_id"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.TRADER
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)

    def test_auth_session_creation(self):
        """测试AuthSession创建"""
        session = AuthSession(
            session_id="session_id",
            user_id="user_id",
            token="jwt_token",
            expires_at=datetime.now() + timedelta(hours=1),
            created_at=datetime.now(),
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        assert session.session_id == "session_id"
        assert session.user_id == "user_id"
        assert session.token == "jwt_token"
        assert session.ip_address == "127.0.0.1"
        assert session.user_agent == "Test Agent"
        assert session.is_active is True

    def test_auth_result_creation(self):
        """测试AuthResult创建"""
        result = AuthResult(
            status=AuthStatus.SUCCESS,
            message="Success",
            factors_completed=["password", "totp"]
        )
        assert result.status == AuthStatus.SUCCESS
        assert result.message == "Success"
        assert result.factors_completed == ["password", "totp"]
        assert result.user is None
        assert result.session is None
        assert result.token is None


class TestEnums:
    """枚举类测试"""

    def test_auth_method_values(self):
        """测试认证方法枚举"""
        assert AuthMethod.PASSWORD.value == ""
        assert AuthMethod.TOTP.value == "totp"
        assert AuthMethod.SMS.value == "sms"
        assert AuthMethod.EMAIL.value == "email"
        assert AuthMethod.BIOMETRIC.value == ""
        assert AuthMethod.HARDWARE_TOKEN.value == "hardware_token"

    def test_user_role_values(self):
        """测试用户角色枚举"""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.TRADER.value == "trader"
        assert UserRole.ANALYST.value == "analyst"
        assert UserRole.VIEWER.value == "viewer"

    def test_auth_status_values(self):
        """测试认证状态枚举"""
        assert AuthStatus.SUCCESS.value == "success"
        assert AuthStatus.FAILED.value == "failed"
        assert AuthStatus.EXPIRED.value == "expired"
        assert AuthStatus.LOCKED.value == "locked"
        assert AuthStatus.PENDING.value == "pending"


if __name__ == "__main__":
    pytest.main([__file__])
