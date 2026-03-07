#!/usr/bin/env python3
"""
RQA2025 多因子认证系统
提供TOTP和SMS等多种MFA方式
"""

import hmac
import hashlib
import base64
import secrets
import time
import struct
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MFAType(Enum):
    """多因子认证类型"""
    TOTP = "totp"  # 时间-based OTP
    HOTP = "hotp"  # HMAC-based OTP
    SMS = "sms"   # 短信验证码
    EMAIL = "email"  # 邮件验证码


@dataclass
class MFASecret:
    """MFA密钥信息"""
    secret: str
    type: MFAType
    created_at: float
    last_used: Optional[float] = None
    backup_codes: list = None

    def __post_init__(self):
        if self.backup_codes is None:
            self.backup_codes = []


class TOTPGenerator:
    """TOTP生成器"""

    @staticmethod
    def generate_secret(length: int = 32) -> str:
        """生成随机密钥"""
        return base64.b32encode(secrets.token_bytes(length)).decode('utf-8')

    @staticmethod
    def generate_totp(secret: str, time_step: int = 30,
                      digits: int = 6) -> str:
        """
        生成TOTP码

        Args:
            secret: 密钥
            time_step: 时间步长(秒)
            digits: 数字位数

        Returns:
            TOTP码
        """
        # 解码密钥
        key = base64.b32decode(secret.upper())

        # 计算时间窗口
        time_window = int(time.time() // time_step)

        # 转换为字节
        time_bytes = struct.pack('>Q', time_window)

        # HMAC-SHA1
        hmac_hash = hmac.new(key, time_bytes, hashlib.sha1).digest()

        # 动态截断
        offset = hmac_hash[-1] & 0x0F
        code = struct.unpack('>I', hmac_hash[offset:offset + 4])[0] & 0x7FFFFFFF

        # 生成指定位数的代码
        return str(code % (10 ** digits)).zfill(digits)

    @staticmethod
    def verify_totp(secret: str, code: str, time_step: int = 30,
                    window: int = 1, digits: int = 6) -> bool:
        """
        验证TOTP码

        Args:
            secret: 密钥
            code: 要验证的代码
            time_step: 时间步长
            window: 验证窗口(前后各多少个时间步长)
            digits: 数字位数

        Returns:
            验证是否成功
        """
        try:
            code_int = int(code)
        except ValueError:
            return False

        current_time = int(time.time() // time_step)

        # 检查时间窗口内的代码
        for i in range(-window, window + 1):
            check_time = current_time + i
            expected_code = TOTPGenerator.generate_totp(
                secret, time_step, digits)

            if int(expected_code) == code_int:
                return True

        return False

    @staticmethod
    def get_totp_uri(secret: str, account_name: str,
                     issuer: str = "RQA2025") -> str:
        """
        生成TOTP URI (用于生成二维码)

        Args:
            secret: 密钥
            account_name: 账户名
            issuer: 发行者

        Returns:
            TOTP URI
        """
        return f"otpauth://totp/{issuer}:{account_name}?secret={secret}&issuer={issuer}"


class MFAProvider:
    """MFA提供商"""

    def __init__(self):
        # 用户MFA配置存储 (生产环境中应该使用数据库)
        self._user_mfa: Dict[str, MFASecret] = {}

        # 验证码存储 (带过期时间)
        self._verification_codes: Dict[str, Dict[str, Any]] = {}

        logger.info("MFA提供商初始化完成")

    def setup_totp(self, username: str) -> Dict[str, str]:
        """
        设置TOTP认证

        Args:
            username: 用户名

        Returns:
            包含密钥和URI的字典
        """
        if username in self._user_mfa:
            raise ValueError(f"用户 {username} 已经设置了MFA")

        secret = TOTPGenerator.generate_secret()
        uri = TOTPGenerator.get_totp_uri(secret, username)

        mfa_secret = MFASecret(
            secret=secret,
            type=MFAType.TOTP,
            created_at=time.time(),
            backup_codes=self._generate_backup_codes()
        )

        self._user_mfa[username] = mfa_secret

        logger.info(f"用户 {username} TOTP设置完成")

        return {
            "secret": secret,
            "uri": uri,
            "qr_code_url": f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={uri}"
        }

    def verify_mfa(self, username: str, code: str,
                   mfa_type: MFAType = MFAType.TOTP) -> bool:
        """
        验证MFA代码

        Args:
            username: 用户名
            code: MFA代码
            mfa_type: MFA类型

        Returns:
            验证是否成功
        """
        mfa_secret = self._user_mfa.get(username)
        if not mfa_secret:
            logger.warning(f"用户 {username} 未设置MFA")
            return False

        if mfa_secret.type != mfa_type:
            logger.warning(f"用户 {username} MFA类型不匹配")
            return False

        success = False

        if mfa_type == MFAType.TOTP:
            success = TOTPGenerator.verify_totp(mfa_secret.secret, code)
        elif mfa_type in [MFAType.SMS, MFAType.EMAIL]:
            success = self._verify_verification_code(username, code, mfa_type)
        else:
            logger.error(f"不支持的MFA类型: {mfa_type}")
            return False

        if success:
            mfa_secret.last_used = time.time()
            logger.info(f"用户 {username} MFA验证成功")

        return success

    def send_verification_code(self, username: str, contact: str,
                               mfa_type: MFAType) -> bool:
        """
        发送验证码 (模拟实现)

        Args:
            username: 用户名
            contact: 联系方式(手机号或邮箱)
            mfa_type: MFA类型

        Returns:
            发送是否成功
        """
        if mfa_type not in [MFAType.SMS, MFAType.EMAIL]:
            raise ValueError(f"不支持的验证码类型: {mfa_type}")

        # 生成6位验证码
        code = str(secrets.randbelow(1000000)).zfill(6)

        # 存储验证码 (5分钟过期)
        self._verification_codes[username] = {
            "code": code,
            "type": mfa_type,
            "contact": contact,
            "created_at": time.time(),
            "expires_at": time.time() + 300  # 5分钟
        }

        # 模拟发送 (生产环境中应该调用真实的SMS/Email服务)
        if mfa_type == MFAType.SMS:
            print(f"📱 [模拟] 短信验证码已发送到 {contact}: {code}")
        elif mfa_type == MFAType.EMAIL:
            print(f"📧 [模拟] 邮件验证码已发送到 {contact}: {code}")

        logger.info(f"验证码已发送给用户 {username}")
        return True

    def use_backup_code(self, username: str, code: str) -> bool:
        """
        使用备用代码

        Args:
            username: 用户名
            code: 备用代码

        Returns:
            使用是否成功
        """
        mfa_secret = self._user_mfa.get(username)
        if not mfa_secret:
            return False

        if code in mfa_secret.backup_codes:
            mfa_secret.backup_codes.remove(code)
            mfa_secret.last_used = time.time()
            logger.info(f"用户 {username} 使用备用代码成功")
            return True

        logger.warning(f"用户 {username} 备用代码无效")
        return False

    def disable_mfa(self, username: str) -> bool:
        """
        禁用用户MFA

        Args:
            username: 用户名

        Returns:
            禁用是否成功
        """
        if username in self._user_mfa:
            del self._user_mfa[username]
            logger.info(f"用户 {username} MFA已禁用")
            return True

        return False

    def get_mfa_status(self, username: str) -> Optional[Dict[str, Any]]:
        """
        获取用户MFA状态

        Args:
            username: 用户名

        Returns:
            MFA状态信息或None
        """
        mfa_secret = self._user_mfa.get(username)
        if not mfa_secret:
            return None

        return {
            "enabled": True,
            "type": mfa_secret.type.value,
            "created_at": mfa_secret.created_at,
            "last_used": mfa_secret.last_used,
            "backup_codes_count": len(mfa_secret.backup_codes)
        }

    def _verify_verification_code(self, username: str, code: str,
                                  mfa_type: MFAType) -> bool:
        """验证验证码"""
        verification = self._verification_codes.get(username)
        if not verification:
            return False

        # 检查过期
        if time.time() > verification["expires_at"]:
            del self._verification_codes[username]
            return False

        # 检查类型匹配
        if verification["type"] != mfa_type:
            return False

        # 检查验证码
        if verification["code"] == code:
            del self._verification_codes[username]  # 使用后删除
            return True

        return False

    def _generate_backup_codes(self, count: int = 10) -> list:
        """生成备用代码"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8位十六进制
            codes.append(code)
        return codes

    def cleanup_expired_codes(self):
        """清理过期的验证码"""
        current_time = time.time()
        expired_users = [
            username for username, data in self._verification_codes.items()
            if current_time > data["expires_at"]
        ]

        for username in expired_users:
            del self._verification_codes[username]

        if expired_users:
            logger.info(f"清理了 {len(expired_users)} 个过期验证码")


# 全局MFA提供商实例
mfa_provider = MFAProvider()


def init_mfa_system():
    """初始化MFA系统"""
    logger.info("多因子认证系统初始化完成")


# 认证流程辅助函数
def authenticate_with_mfa(username: str, password: str,
                          mfa_code: str = None) -> Dict[str, Any]:
    """
    带MFA的完整认证流程

    Args:
        username: 用户名
        password: 密码
        mfa_code: MFA代码 (可选)

    Returns:
        认证结果字典
    """
    from .jwt_auth import authenticator

    result = {
        "success": False,
        "message": "",
        "require_mfa": False,
        "token_pair": None
    }

    # 首先进行基础认证
    token_pair = authenticator.authenticate(username, password)
    if not token_pair:
        result["message"] = "用户名或密码错误"
        return result

    # 检查是否启用了MFA
    mfa_status = mfa_provider.get_mfa_status(username)
    if not mfa_status:
        # 未启用MFA，直接认证成功
        result["success"] = True
        result["token_pair"] = token_pair
        result["message"] = "认证成功"
        return result

    # 需要MFA验证
    result["require_mfa"] = True

    if not mfa_code:
        result["message"] = "需要MFA验证"
        return result

    # 验证MFA代码
    if mfa_provider.verify_mfa(username, mfa_code):
        result["success"] = True
        result["token_pair"] = token_pair
        result["message"] = "认证成功"
    else:
        result["message"] = "MFA代码错误"

    return result


if __name__ == "__main__":
    # 初始化MFA系统
    init_mfa_system()

    # 测试MFA系统
    print("🔐 测试多因子认证系统")
    print("=" * 50)

    username = "admin"

    # 1. 设置TOTP
    print("📱 设置TOTP认证")
    try:
        setup_result = mfa_provider.setup_totp(username)
        print("✅ TOTP设置成功")
        print(f"密钥: {setup_result['secret']}")
        print(f"二维码URL: {setup_result['qr_code_url']}")

        # 2. 生成TOTP码
        print("\n🔢 生成TOTP码")
        secret = setup_result['secret']
        totp_code = TOTPGenerator.generate_totp(secret)
        print(f"当前TOTP码: {totp_code}")

        # 3. 验证TOTP码
        print("\n🔍 验证TOTP码")
        is_valid = mfa_provider.verify_mfa(username, totp_code)
        print(f"TOTP验证结果: {'✅ 成功' if is_valid else '❌ 失败'}")

        # 4. 测试备用代码
        print("\n🔧 测试备用代码")
        mfa_secret = mfa_provider._user_mfa[username]
        if mfa_secret.backup_codes:
            backup_code = mfa_secret.backup_codes[0]
            backup_valid = mfa_provider.use_backup_code(username, backup_code)
            print(f"备用代码验证: {'✅ 成功' if backup_valid else '❌ 失败'}")

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")

    # 5. 测试SMS验证
    print("\n📲 测试SMS验证")
    try:
        mfa_provider.send_verification_code(username, "+1234567890", MFAType.SMS)

        # 模拟用户输入验证码
        verification = mfa_provider._verification_codes.get(username)
        if verification:
            test_code = verification["code"]
            sms_valid = mfa_provider.verify_mfa(username, test_code, MFAType.SMS)
            print(f"SMS验证结果: {'✅ 成功' if sms_valid else '❌ 失败'}")

    except Exception as e:
        print(f"❌ SMS测试出错: {e}")

    print("\n🎉 多因子认证系统测试完成")
