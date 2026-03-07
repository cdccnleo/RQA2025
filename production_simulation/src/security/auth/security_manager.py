#!/usr/bin/env python3
"""
RQA2025 安全认证管理器
整合JWT、RBAC、MFA的安全认证系统
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import time
import logging

from .jwt_auth import authenticator
from .rbac import rbac_manager, Permission
from .mfa import mfa_provider, MFAType

logger = logging.getLogger(__name__)


@dataclass
class AuthResult:
    """认证结果"""
    success: bool
    message: str
    token_pair: Optional[Any] = None
    user_info: Optional[Dict[str, Any]] = None
    require_mfa: bool = False
    mfa_setup_required: bool = False


@dataclass
class PermissionCheckResult:
    """权限检查结果"""
    has_permission: bool
    user_permissions: List[str]
    required_permissions: List[str]


class SecurityManager:
    """安全认证管理器"""

    def __init__(self):
        self.jwt_auth = authenticator
        self.rbac_manager = rbac_manager
        self.mfa_provider = mfa_provider

        logger.info("安全认证管理器初始化完成")

    def register_user(self, username: str, password: str,
                      roles: List[str] = None,
                      require_mfa: bool = False) -> bool:
        """
        注册新用户

        Args:
            username: 用户名
            password: 密码
            roles: 用户角色列表
            require_mfa: 是否需要MFA

        Returns:
            注册是否成功
        """
        # 注册到JWT认证系统
        if not self.jwt_auth.register_user(username, password, roles or ["user"]):
            return False

        # 分配角色到RBAC系统
        for role in roles or ["user"]:
            if not self.rbac_manager.assign_role_to_user(username, role):
                logger.warning(f"为用户 {username} 分配角色 {role} 失败")

        # 如果需要MFA，稍后会提示设置
        if require_mfa:
            logger.info(f"用户 {username} 需要设置MFA")

        logger.info(f"用户 {username} 注册完成")
        return True

    def authenticate_user(self, username: str, password: str,
                          mfa_code: Optional[str] = None) -> AuthResult:
        """
        用户认证

        Args:
            username: 用户名
            password: 密码
            mfa_code: MFA代码 (可选)

        Returns:
            认证结果
        """
        result = AuthResult(success=False, message="")

        # 检查MFA设置
        mfa_status = self.mfa_provider.get_mfa_status(username)
        mfa_enabled = mfa_status is not None

        # 如果启用了MFA但没有提供MFA代码，先进行基础认证
        if mfa_enabled and not mfa_code:
            # 验证基础凭据
            temp_token = self.jwt_auth.authenticate(username, password)
            if temp_token:
                result.require_mfa = True
                result.message = "需要MFA验证"
                result.user_info = self.jwt_auth.get_user_info(username)
            else:
                result.message = "用户名或密码错误"
            return result

        # 完整认证流程
        if mfa_enabled and mfa_code:
            # MFA认证
            mfa_valid = self.mfa_provider.verify_mfa(username, mfa_code)
            if not mfa_valid:
                result.message = "MFA代码错误"
                return result

        # JWT认证
        token_pair = self.jwt_auth.authenticate(username, password)
        if not token_pair:
            result.message = "用户名或密码错误"
            return result

        # 获取用户信息
        user_info = self.jwt_auth.get_user_info(username)
        if not user_info:
            result.message = "无法获取用户信息"
            return result

        result.success = True
        result.message = "认证成功"
        result.token_pair = token_pair
        result.user_info = user_info

        logger.info(f"用户 {username} 认证成功")
        return result

    def refresh_user_token(self, refresh_token: str) -> Optional[Any]:
        """
        刷新用户令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            新的令牌对或None
        """
        return self.jwt_auth.refresh_token(refresh_token)

    def validate_user_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        验证用户令牌

        Args:
            access_token: 访问令牌

        Returns:
            令牌载荷或None
        """
        return self.jwt_auth.validate_token(access_token)

    def revoke_user_token(self, token: str) -> bool:
        """
        吊销用户令牌

        Args:
            token: 要吊销的令牌

        Returns:
            吊销是否成功
        """
        return self.jwt_auth.revoke_token(token)

    def check_user_permission(self, username: str,
                              permission: Permission) -> PermissionCheckResult:
        """
        检查用户权限

        Args:
            username: 用户名
            permission: 权限

        Returns:
            权限检查结果
        """
        has_permission = self.rbac_manager.check_permission(username, permission)

        user_permissions = [
            p.value for p in self.rbac_manager.get_user_permissions(username)
        ]

        return PermissionCheckResult(
            has_permission=has_permission,
            user_permissions=user_permissions,
            required_permissions=[permission.value]
        )

    def check_user_permissions(self, username: str,
                               permissions: List[Permission],
                               require_all: bool = True) -> PermissionCheckResult:
        """
        检查用户多个权限

        Args:
            username: 用户名
            permissions: 权限列表
            require_all: 是否需要全部权限

        Returns:
            权限检查结果
        """
        if require_all:
            has_permission = self.rbac_manager.check_all_permissions(username, permissions)
        else:
            has_permission = self.rbac_manager.check_any_permission(username, permissions)

        user_permissions = [
            p.value for p in self.rbac_manager.get_user_permissions(username)
        ]

        required_permissions = [p.value for p in permissions]

        return PermissionCheckResult(
            has_permission=has_permission,
            user_permissions=user_permissions,
            required_permissions=required_permissions
        )

    def setup_user_mfa(self, username: str,
                       mfa_type: MFAType = MFAType.TOTP) -> Dict[str, Any]:
        """
        设置用户MFA

        Args:
            username: 用户名
            mfa_type: MFA类型

        Returns:
            MFA设置信息
        """
        if mfa_type == MFAType.TOTP:
            return self.mfa_provider.setup_totp(username)
        elif mfa_type in [MFAType.SMS, MFAType.EMAIL]:
            # 这里需要用户提供联系方式
            raise ValueError("SMS/Email MFA需要提供联系方式")
        else:
            raise ValueError(f"不支持的MFA类型: {mfa_type}")

    def send_mfa_code(self, username: str, contact: str,
                      mfa_type: MFAType) -> bool:
        """
        发送MFA验证码

        Args:
            username: 用户名
            contact: 联系方式
            mfa_type: MFA类型

        Returns:
            发送是否成功
        """
        return self.mfa_provider.send_verification_code(username, contact, mfa_type)

    def disable_user_mfa(self, username: str) -> bool:
        """
        禁用用户MFA

        Args:
            username: 用户名

        Returns:
            禁用是否成功
        """
        return self.mfa_provider.disable_mfa(username)

    def get_user_security_info(self, username: str) -> Dict[str, Any]:
        """
        获取用户安全信息

        Args:
            username: 用户名

        Returns:
            用户安全信息
        """
        user_info = self.jwt_auth.get_user_info(username)
        mfa_status = self.mfa_provider.get_mfa_status(username)
        user_roles = self.rbac_manager.get_user_roles(username)
        user_permissions = [
            p.value for p in self.rbac_manager.get_user_permissions(username)
        ]

        return {
            "username": username,
            "user_info": user_info,
            "mfa_enabled": mfa_status is not None,
            "mfa_type": mfa_status.get("type") if mfa_status else None,
            "mfa_last_used": mfa_status.get("last_used") if mfa_status else None,
            "roles": user_roles,
            "permissions": user_permissions,
            "last_login": user_info.get("last_login") if user_info else None
        }

    def assign_role_to_user(self, username: str, role_name: str) -> bool:
        """
        为用户分配角色

        Args:
            username: 用户名
            role_name: 角色名

        Returns:
            分配是否成功
        """
        return self.rbac_manager.assign_role_to_user(username, role_name)

    def revoke_role_from_user(self, username: str, role_name: str) -> bool:
        """
        从用户撤销角色

        Args:
            username: 用户名
            role_name: 角色名

        Returns:
            撤销是否成功
        """
        return self.rbac_manager.revoke_role_from_user(username, role_name)

    def list_users(self) -> List[Dict[str, Any]]:
        """
        列出所有用户

        Returns:
            用户列表
        """
        users = []
        for user_info in self.rbac_manager.list_users():
            username = user_info["username"]
            security_info = self.get_user_security_info(username)
            users.append(security_info)
        return users

    def get_system_security_stats(self) -> Dict[str, Any]:
        """
        获取系统安全统计

        Returns:
            安全统计信息
        """
        users = self.rbac_manager.list_users()
        total_users = len(users)
        mfa_enabled_users = sum(
            1 for user in users
            if self.mfa_provider.get_mfa_status(user["username"]) is not None
        )

        return {
            "total_users": total_users,
            "mfa_enabled_users": mfa_enabled_users,
            "mfa_adoption_rate": mfa_enabled_users / total_users if total_users > 0 else 0,
            "active_roles": len(self.rbac_manager.roles),
            "total_permissions": len(Permission),
            "system_start_time": time.time()  # 简化为当前时间
        }


# 全局安全管理器实例
security_manager = SecurityManager()


def init_security_system():
    """初始化安全系统"""
    # 初始化各个子系统
    from .jwt_auth import init_auth_system
    from .rbac import init_rbac_system
    from .mfa import init_mfa_system

    init_auth_system()
    init_rbac_system()
    init_mfa_system()

    # 创建默认管理员用户
    security_manager.register_user(
        username="admin",
        password="Admin@123456",
        roles=["super_admin"],
        require_mfa=True
    )

    # 创建示例用户
    security_manager.register_user(
        username="trader",
        password="Trader@123456",
        roles=["trader"],
        require_mfa=False
    )

    security_manager.register_user(
        username="analyst",
        password="Analyst@123456",
        roles=["analyst"],
        require_mfa=False
    )

    logger.info("安全认证系统初始化完成")


if __name__ == "__main__":
    # 初始化安全系统
    init_security_system()

    # 测试综合安全认证
    print("🔐 测试RQA2025综合安全认证系统")
    print("=" * 60)

    # 1. 测试用户注册
    print("📝 测试用户注册")
    success = security_manager.register_user(
        "test_user", "Test@123456", ["user"], False
    )
    print(f"用户注册: {'✅ 成功' if success else '❌ 失败'}")

    # 2. 测试认证
    print("\n🔑 测试用户认证")
    auth_result = security_manager.authenticate_user("admin", "Admin@123456")
    print(f"管理员认证: {'✅ 成功' if auth_result.success else '❌ 失败'}")
    print(f"消息: {auth_result.message}")

    if auth_result.success and auth_result.token_pair:
        print(f"访问令牌: {auth_result.token_pair.access_token[:30]}...")
        print(f"刷新令牌: {auth_result.token_pair.refresh_token[:30]}...")

        # 3. 测试令牌验证
        print("\n🔍 测试令牌验证")
        token_payload = security_manager.validate_user_token(auth_result.token_pair.access_token)
        if token_payload:
            print("✅ 令牌验证成功")
            print(f"用户: {token_payload['sub']}")
            print(f"角色: {token_payload['roles']}")
        else:
            print("❌ 令牌验证失败")

        # 4. 测试权限检查
        print("\n👥 测试权限检查")
        perm_result = security_manager.check_user_permission("admin", Permission.SYSTEM_ADMIN)
        print(f"管理员权限检查: {'✅ 有权限' if perm_result.has_permission else '❌ 无权限'}")
        print(f"用户权限数量: {len(perm_result.user_permissions)}")

        perm_result2 = security_manager.check_user_permission("trader", Permission.SYSTEM_ADMIN)
        print(f"交易员权限检查: {'✅ 有权限' if perm_result2.has_permission else '❌ 无权限'}")

    # 5. 测试MFA设置
    print("\n📱 测试MFA设置")
    try:
        mfa_setup = security_manager.setup_user_mfa("admin", MFAType.TOTP)
        print("✅ MFA设置成功")
        print(f"TOTP密钥: {mfa_setup['secret']}")
        print(f"二维码URL: {mfa_setup['qr_code_url']}")

        # 测试MFA认证
        from .mfa import TOTPGenerator
        totp_code = TOTPGenerator.generate_totp(mfa_setup['secret'])
        auth_with_mfa = security_manager.authenticate_user("admin", "Admin@123456", totp_code)
        print(f"MFA认证: {'✅ 成功' if auth_with_mfa.success else '❌ 失败'}")

    except Exception as e:
        print(f"❌ MFA测试出错: {e}")

    # 6. 系统安全统计
    print("\n📊 系统安全统计")
    stats = security_manager.get_system_security_stats()
    print(f"总用户数: {stats['total_users']}")
    print(f"MFA启用用户: {stats['mfa_enabled_users']}")
    print(".1%")
    print(f"活跃角色数: {stats['active_roles']}")
    print(f"权限总数: {stats['total_permissions']}")

    print("\n🎉 RQA2025综合安全认证系统测试完成！")
    print("=" * 60)
