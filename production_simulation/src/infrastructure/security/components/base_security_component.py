"""
基础安全模块
提供统一的安全接口和基础实现
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import hashlib
import hmac
import secrets
import base64
from datetime import datetime, timedelta
from enum import Enum
from cryptography.fernet import Fernet


# 类型别名定义（前向引用）
# ISecurity 将指向 ISecurityComponent


class SecurityLevel(Enum):

    """安全级别枚举"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ISecurityComponent(ABC):

    """
    Security组件接口

    # Security功能的核心抽象接口

    # # 功能特性
    - 提供Security功能的标准接口定义
    - 支持扩展和定制化实现
    - 保证功能的一致性和可靠性
    """

# 接口定义
# 该接口定义了Security组件的基本契约
# - 核心功能方法定义
# - 错误处理规范
# - 生命周期管理
# - 配置参数要求

    # 实现要求
    # 实现类需要满足以下要求
    # 1. 实现所有抽象方法
    # 2. 处理异常情况
    # 3. 提供必要的配置选项
    # 4. 保证线程安全（如果适用）

    # 使用示例
    # component = ConcreteSecurityComponent(config)

    # 使用组件功能
    # try:
    #     result = component.execute_operation()
    #     print(f"操作结果: {result}")
    # except ComponentError as e:
    #     print(f"组件错误: {e}")

    # 注意事项
    # - 实现类必须保证异常安全
    # - 资源使用需要正确清理
    # - 配置参数需要验证
    # - 日志记录需要完善

    # 相关组件
    # - 依赖: 基础配置组件
    # - 协作: 监控和日志组件
    # - 扩展: 具体实现类

    @abstractmethod
    def encrypt(self, data: str) -> str:
        """Encrypt data"""

    @abstractmethod
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""

    @abstractmethod
    def hash(self, data: str) -> str:
        """Hash data"""

    @abstractmethod
    def verify_hash(self, data: str, hash_value: str) -> bool:
        """Verify hash"""

    @abstractmethod
    def generate_token(self, data: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate token"""

    @abstractmethod
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token"""

    @abstractmethod
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input"""


class BaseSecurityComponent(ISecurityComponent):

    """基础安全实现"""

    def __init__(self, secret_key: Optional[str] = None):

        self.secret_key = secret_key or self._generate_secret_key()
        self.security_level = SecurityLevel.MEDIUM

    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        try:
            # 简单的base64编码（实际应用中应使用更强的加密）
            encoded = base64.b64encode(data.encode()).decode()
            return f"encrypted:{encoded}"
        except Exception:
            return data

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            if encrypted_data.startswith("encrypted:"):
                encoded = encrypted_data[10:]
                decoded = base64.b64decode(encoded.encode()).decode()
                return decoded
            return encrypted_data
        except Exception:
            return encrypted_data

    def hash(self, data: str) -> str:
        """Hash data"""
        try:
            return hashlib.sha256(data.encode()).hexdigest()
        except Exception:
            return ""

    def verify_hash(self, data: str, hash_value: str) -> bool:
        """Verify hash"""
        try:
            return self.hash(data) == hash_value
        except Exception:
            return False

    def generate_token(self, data: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate token"""
        try:
            # 添加过期时间
            data['exp'] = datetime.now().timestamp() + expires_in
            data['iat'] = datetime.now().timestamp()

            # 简单的令牌生成（实际应用中应使用JWT）
            token_data = f"{data.get('user_id', '')}:{data.get('exp', 0)}"
            signature = hmac.new(
                self.secret_key.encode(),
                token_data.encode(),
                hashlib.sha256
            ).hexdigest()

            return f"{token_data}.{signature}"
        except Exception:
            return ""

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token"""
        try:
            if '.' not in token:
                return None

            token_data, signature = token.rsplit('.', 1)

            # 验证签名
            expected_signature = hmac.new(
                self.secret_key.encode(),
                token_data.encode(),
                hashlib.sha256
            ).hexdigest()

            if signature != expected_signature:
                return None

            # 解析令牌数据
            user_id, exp = token_data.split(':', 1)

            # 检查过期时间
            if datetime.now().timestamp() > float(exp):
                return None

            return {
                'user_id': user_id,
                'exp': float(exp)
            }

        except Exception:
            return None

    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input"""
        try:
            # 基本的输入清理
            sanitized = input_data.strip()

            # 移除潜在的SQL注入字符
            dangerous_chars = ["'", '"', ';', '--', '/*', '*/']
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')

            # 移除潜在的XSS字符
            xss_chars = ['<', '>', 'script', 'javascript']
            for char in xss_chars:
                sanitized = sanitized.replace(char, '')

            return sanitized
        except Exception:
            return ""

    def _generate_secret_key(self) -> str:
        """生成密钥"""
        return secrets.token_hex(32)

    def set_security_level(self, level: SecurityLevel) -> None:
        """设置安全级别"""
        self.security_level = level

    def get_security_level(self) -> SecurityLevel:
        """获取安全级别"""
        return self.security_level

    def validate_password(self, password: str) -> Dict[str, Any]:
        """验证密码强度"""
        result = {
            'is_valid': True,
            'score': 0,
            'issues': []
        }

        if len(password) < 8:
            result['is_valid'] = False
            result['issues'].append("密码长度至少8位")
        else:
            result['score'] += 1

        if not any(c.isupper() for c in password):
            result['issues'].append("需要包含大写字母")
        else:
            result['score'] += 1

        if not any(c.islower() for c in password):
            result['issues'].append("需要包含小写字母")
        else:
            result['score'] += 1

        if not any(c.isdigit() for c in password):
            result['issues'].append("需要包含数字")
        else:
            result['score'] += 1

        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            result['issues'].append("需要包含特殊字符")
        else:
            result['score'] += 1

        if result['score'] < 3:
            result['is_valid'] = False

        return result

    def generate_secure_password(self, length: int = 12) -> str:
        """生成安全密码"""
        import string

        # 确保包含所有类型的字符
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"

        # 每种类型至少一个字符
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(symbols)
        ]

        # 填充剩余长度
        all_chars = lowercase + uppercase + digits + symbols
        password.extend(secrets.choice(all_chars) for _ in range(length - 4))

        # 打乱顺序
        password_list = list(password)
        secrets.SystemRandom().shuffle(password_list)

        return ''.join(password_list)


class AdvancedSecurity(BaseSecurityComponent):

    """高级安全实现"""

    def __init__(self, secret_key: Optional[str] = None):

        super().__init__(secret_key)
        self.security_level = SecurityLevel.HIGH
        self._rate_limit = {}
        self._blacklist = set()

    def encrypt(self, data: str) -> str:
        """高级加密"""
        try:
            # 使用更强的加密算法
            from cryptography.fernet import Fernet

            # 生成密钥
            key = Fernet.generate_key()
            cipher = Fernet(key)

            # 加密
            encrypted = cipher.encrypt(data.encode())
            return f"advanced:{base64.b64encode(key).decode()}:{base64.b64encode(encrypted).decode()}"
        except ImportError:
            # 如果没有cryptography库，回退到基础实现
            return super().encrypt(data)
        except Exception:
            return data

    def decrypt(self, encrypted_data: str) -> str:
        """高级解密"""
        try:
            if encrypted_data.startswith("advanced:"):

                parts = encrypted_data.split(":", 2)
                if len(parts) == 3:
                    key = base64.b64decode(parts[1])
                    encrypted = base64.b64decode(parts[2])

                    cipher = Fernet(key)
                    decrypted = cipher.decrypt(encrypted)
                    return decrypted.decode()

            return super().decrypt(encrypted_data)
        except ImportError:
            return super().decrypt(encrypted_data)
        except Exception:
            return encrypted_data

    def check_rate_limit(self, identifier: str, max_attempts: int = 5, window: int = 300) -> bool:
        """检查速率限制"""
        now = datetime.now()

        if identifier not in self._rate_limit:
            self._rate_limit[identifier] = []

        # 清理过期的尝试
        self._rate_limit[identifier] = [
            attempt for attempt in self._rate_limit[identifier]
            if now - attempt < timedelta(seconds=window)
        ]

        # 检查是否超过限制
        if len(self._rate_limit[identifier]) >= max_attempts:
            return False

        # 记录当前尝试
        self._rate_limit[identifier].append(now)
        return True

    def add_to_blacklist(self, identifier: str) -> None:
        """添加到黑名单"""
        self._blacklist.add(identifier)

    def is_blacklisted(self, identifier: str) -> bool:
        """检查是否在黑名单中"""
        return identifier in self._blacklist

    def remove_from_blacklist(self, identifier: str) -> None:
        """从黑名单移除"""
        self._blacklist.discard(identifier)


# 默认安全实现
ISecurity = BaseSecurityComponent
