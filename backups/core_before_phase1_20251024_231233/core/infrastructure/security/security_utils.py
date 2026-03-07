"""
RQA2025 Security Utils

安全工具类，提供各种安全相关的工具函数
"""

from typing import Dict, Optional
import hashlib
import hmac
import secrets
import base64
import re
import logging

logger = logging.getLogger(__name__)


class SecurityUtils:

    """安全工具类"""

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """哈希密码"""
        if salt is None:
            salt = SecurityUtils.generate_salt()

        # 使用PBKDF2进行密码哈希
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{base64.b64encode(hash_obj).decode()}"

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            salt, hash_value = hashed_password.split(':', 1)
            return SecurityUtils.hash_password(password, salt) == hashed_password
        except (ValueError, IndexError):
            return False

    @staticmethod
    def generate_salt(length: int = 32) -> str:
        """生成盐值"""
        return secrets.token_hex(length // 2)

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """生成安全令牌"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def sanitize_html(input_text: str) -> str:
        """清理HTML输入，防止XSS攻击"""
        if not input_text:
            return ""

        # 移除危险的HTML标签和属性
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # script标签
            r'<iframe[^>]*>.*?</iframe>',  # iframe标签
            r'<object[^>]*>.*?</object>',  # object标签
            r'<embed[^>]*>.*?</embed>',    # embed标签
            r'javascript:',                # javascript协议
            r'on\w+\s*=',                  # 事件处理器
            r'<[^>]*>',                    # 其他HTML标签
        ]

        sanitized = input_text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)

        return sanitized.strip()

    @staticmethod
    def sanitize_sql(input_text: str) -> str:
        """清理SQL输入，防止SQL注入"""
        if not input_text:
            return ""

        # 移除危险的SQL关键字和字符
        dangerous_patterns = [
            r';\s*',           # 分号
            r'--',             # SQL注释
            r'/\*.*?\*/',      # 多行注释
            r'xp_',            # 扩展存储过程
            r'sp_',            # 系统存储过程
            r'exec\s+',        # 执行命令
            r'union\s+',       # UNION操作
            r'drop\s+',        # DROP操作
            r'insert\s+',      # INSERT操作
            r'update\s+',      # UPDATE操作
            r'delete\s+',      # DELETE操作
            r'select\s+',      # SELECT操作
        ]

        sanitized = input_text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        if not email:
            return False

        pattern = r'^[a - zA - Z0 - 9._%+-]+@[a - zA - Z0 - 9.-]+\.[a - zA - Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """验证IP地址格式"""
        if not ip:
            return False

        # IPv4验证
        ipv4_pattern = r'^((25[0 - 5]|2[0 - 4][0 - 9]|[01]?[0 - 9][0 - 9]?)\.){3}(25[0 - 5]|2[0 - 4][0 - 9]|[01]?[0 - 9][0 - 9]?)$'
        if re.match(ipv4_pattern, ip):
            return True

        # IPv6验证（简化版）
        ipv6_pattern = r'^([0 - 9a - fA - F]{1,4}:){7}[0 - 9a - fA - F]{1,4}$'
        return re.match(ipv6_pattern, ip) is not None

    @staticmethod
    def generate_hmac_signature(data: str, secret_key: str) -> str:
        """生成HMAC签名"""
        if not data or not secret_key:
            return ""

        signature = hmac.new(
            secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).digest()

        return base64.b64encode(signature).decode()

    @staticmethod
    def verify_hmac_signature(data: str, signature: str, secret_key: str) -> bool:
        """验证HMAC签名"""
        if not data or not signature or not secret_key:
            return False

        expected_signature = SecurityUtils.generate_hmac_signature(data, secret_key)
        return hmac.compare_digest(signature, expected_signature)

    @staticmethod
    def encrypt_data(data: str, key: str) -> str:
        """简单的数据加密"""
        if not data or not key:
            return ""

        try:
            # 使用简单的XOR加密（仅用于演示，生产环境应使用更强的加密）
            encrypted = []
            key_length = len(key)

            for i, char in enumerate(data):
                key_char = key[i % key_length]
                encrypted_char = chr(ord(char) ^ ord(key_char))
                encrypted.append(encrypted_char)

            return base64.b64encode(''.join(encrypted).encode()).decode()
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            return ""

    @staticmethod
    def decrypt_data(encrypted_data: str, key: str) -> str:
        """简单的数据解密"""
        if not encrypted_data or not key:
            return ""

        try:
            # 解码base64
            decoded_data = base64.b64decode(encrypted_data).decode()

            # XOR解密
            decrypted = []
            key_length = len(key)

            for i, char in enumerate(decoded_data):
                key_char = key[i % key_length]
                decrypted_char = chr(ord(char) ^ ord(key_char))
                decrypted.append(decrypted_char)

            return ''.join(decrypted)
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            return ""

    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """获取安全HTTP头"""
        return {
            'X - Content - Type - Options': 'nosniff',
            'X - Frame - Options': 'DENY',
            'X - XSS - Protection': '1; mode=block',
            'Strict - Transport - Security': 'max - age=31536000; includeSubDomains',
            'Content - Security - Policy': "default - src 'self'",
            'Referrer - Policy': 'strict - origin - when - cross - origin'
        }

    @staticmethod
    def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
        """掩码敏感数据"""
        if not data or len(data) <= visible_chars:
            return data

        masked_length = len(data) - visible_chars
        mask = '*' * masked_length

        return f"{mask}{data[-visible_chars:]}"
