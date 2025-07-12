"""Security utilities for configuration management including data sanitization and encryption."""
import os
from functools import wraps
from typing import Any, Dict, Callable, Optional
from datetime import datetime, timedelta
import hashlib
import re
import time
import logging
import threading
from typing import Dict, Any, Optional
from enum import Enum

from Crypto.Util.Padding import unpad
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from gmssl import sm4  # 使用gmssl库的SM4实现
from numpy import pad

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityConfig:
    """安全配置参数类，用于集中管理安全相关配置"""
    
    def __init__(self):
        # 加密配置
        self.encryption_algorithm = 'AES'  # AES|SM4
        self.key_rotation_days = 30
        
        # 签名配置
        self.signature_algorithm = 'HMAC-SHA256'
        
        # 敏感数据处理配置
        self.sensitive_keys = {'password', 'secret', 'token', 'key', 'credential'}
        self.api_key_display_chars = 8  # 显示API密钥前N个字符
        
        # 输入验证配置
        self.enable_sql_injection_check = True
        self.enable_xss_check = True
        self.enable_path_traversal_check = True
    
    def validate(self) -> bool:
        """验证配置参数是否有效"""
        if self.encryption_algorithm not in {'AES', 'SM4'}:
            return False
        if self.key_rotation_days < 1:
            return False
        if self.api_key_display_chars < 0:
            return False
        return True
    
    def update_from_dict(self, config_dict: dict) -> None:
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class KeyManager:
    """密钥管理服务，支持动态轮换"""
    
    def __init__(self):
        self._keys = {}  # {key_id: {'current': key, 'previous': key, 'expiry': datetime}}
        self._rotation_interval = timedelta(days=30)
        
    def generate_key(self, algorithm: str = 'AES') -> bytes:
        """生成新密钥"""
        if algorithm == 'SM4':
            return os.urandom(16)
        return os.urandom(32)
        
    def rotate_key(self, key_id: str) -> None:
        """轮换密钥"""
        if key_id in self._keys:
            self._keys[key_id]['previous'] = self._keys[key_id]['current']
        self._keys[key_id] = {
            'current': self.generate_key(),
            'expiry': datetime.now() + self._rotation_interval
        }
        
    def get_key(self, key_id: str, allow_previous: bool = False) -> bytes:
        """获取当前有效密钥"""
        if key_id not in self._keys or datetime.now() > self._keys[key_id]['expiry']:
            self.rotate_key(key_id)
        return self._keys[key_id]['current']

class EncryptionService:
    """加密服务，支持AES和国密SM4算法"""
    
    def __init__(self):
        self._key_manager = KeyManager()
        
    def encrypt(self, data: str, key_id: str = 'default', algorithm: str = 'AES') -> bytes:
        """加密数据"""
        key = self._key_manager.get_key(key_id)
        if algorithm == 'SM4':
            cipher = sm4.CryptSM4()
            cipher.set_key(key, sm4.SM4_ENCRYPT)
            iv = os.urandom(16)
            cipher.set_iv(iv)
            return iv + cipher.crypt_cbc(pad(data.encode()))
        else:
            iv = os.urandom(16)
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data.encode()) + padder.finalize()
            return iv + encryptor.update(padded_data) + encryptor.finalize()
            
    def decrypt(self, encrypted_data: bytes, key_id: str = 'default', algorithm: str = 'AES') -> str:
        """解密数据"""
        key = self._key_manager.get_key(key_id, allow_previous=True)
        if algorithm == 'SM4':
            iv = encrypted_data[:16]
            cipher = sm4.CryptSM4()
            cipher.set_key(key, sm4.SM4_DECRYPT)
            cipher.set_iv(iv)
            return unpad(cipher.crypt_cbc(encrypted_data[16:])).decode()
        else:
            iv = encrypted_data[:16]
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            unpadder = padding.PKCS7(128).unpadder()
            padded_data = decryptor.update(encrypted_data[16:]) + decryptor.finalize()
            unpadded_data = unpadder.update(padded_data) + unpadder.finalize()
            return unpadded_data.decode()

class DataSanitizer:
    """Utility class for sanitizing sensitive data in configuration."""

    # Default sensitive keys
    SENSITIVE_KEYS = {'password', 'secret', 'token', 'key', 'credential'}

    @classmethod
    def sanitize(cls, data: Any, context: str = '') -> Any:
        """Recursively sanitize sensitive data."""
        if isinstance(data, dict):
            return {
                k: '***' if any(s in k.lower() for s in cls.SENSITIVE_KEYS)
                   else cls.sanitize(v, k)
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return type(data)(cls.sanitize(item) for item in data)
        elif isinstance(data, str) and any(
            s in context.lower() for s in cls.SENSITIVE_KEYS
        ):
            return '***'
        return data

    @classmethod
    def sensitive_updates(cls, func: Callable) -> Callable:
        """Decorator to sanitize configuration updates."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize input arguments
            args = tuple(cls.sanitize(arg) for arg in args)
            kwargs = {k: cls.sanitize(v, k) for k, v in kwargs.items()}

            # Execute original function
            result = func(*args, **kwargs)

            # Sanitize output if needed
            return cls.sanitize(result)
        return wrapper

    @classmethod
    def add_sensitive_key(cls, key: str) -> None:
        """Register additional sensitive key patterns.
        
        Example:
            >>> DataSanitizer.add_sensitive_key('api_token')
            >>> DataSanitizer.sanitize({'api_token': 'secret'})
            {'api_token': '***'}
        """
        cls.SENSITIVE_KEYS.add(key.lower())

class SecurityManager:
    """统一安全服务，整合脱敏、加密和签名功能"""
    
    def __init__(self):
        self.sanitizer = DataSanitizer
        self.encryptor = EncryptionService()
        self._sign_key = os.urandom(32)  # HMAC-SHA256签名密钥
        
    def sign_config(self, config: dict) -> dict:
        """对配置进行数字签名
        
        Args:
            config: 要签名的配置字典
            
        Returns:
            包含签名信息的配置字典
        """
        import hmac
        import hashlib
        import json
        
        # 序列化配置并计算签名
        serialized = json.dumps(config, sort_keys=True).encode()
        signature = hmac.new(
            self._sign_key,
            serialized,
            hashlib.sha256
        ).hexdigest()
        
        return {
            'config': config,
            'signature': signature,
            'algorithm': 'HMAC-SHA256'
        }
        
    def verify_signature(self, signed_config: dict) -> bool:
        """验证配置签名
        
        Args:
            signed_config: 包含签名的配置字典
            
        Returns:
            bool: 签名是否有效
        """
        import hmac
        import hashlib
        import json
        
        if 'config' not in signed_config or 'signature' not in signed_config:
            return False
            
        # 重新计算签名
        serialized = json.dumps(signed_config['config'], sort_keys=True).encode()
        expected_signature = hmac.new(
            self._sign_key,
            serialized,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(
            expected_signature,
            signed_config['signature']
        )
        
    def protect_config(self, config: dict) -> dict:
        """保护敏感配置：加密值并脱敏显示"""
        protected = {}
        for key, value in config.items():
            if any(s in key.lower() for s in DataSanitizer.SENSITIVE_KEYS):
                protected[key] = {
                    'encrypted': True,
                    'value': self.encryptor.encrypt(str(value)),
                    'display': '***'
                }
            else:
                protected[key] = value
        return protected
        
    def reveal_config(self, protected_config: dict) -> dict:
        """解密受保护的配置"""
        revealed = {}
        for key, value in protected_config.items():
            if isinstance(value, dict) and value.get('encrypted'):
                revealed[key] = self.encryptor.decrypt(value['value'])
            else:
                revealed[key] = value
        return revealed

    def filter_sensitive_data(self, config: dict) -> dict:
        """过滤配置中的敏感数据

        处理规则:
        - 对于普通敏感字段(password/secret等): 替换为'***'
        - 对于api_keys字典: 保留前8个字符(包括下划线)，其余替换为'****'
        - 其他数据保持不变
        """
        if not isinstance(config, dict):
            return config

        filtered = {}
        for key, value in config.items():
            if isinstance(value, dict):
                if key == 'api_keys':
                    # 特殊处理api_keys字典，保留前8个字符(包括下划线)
                    filtered[key] = {
                        k: (v[:8] if len(v) > 8 else v) + '****' if isinstance(v, str) else '****'
                        for k, v in value.items()
                    }
                else:
                    filtered[key] = self.filter_sensitive_data(value)
            else:
                # 处理普通字段
                if any(s in key.lower() for s in ['password', 'secret', 'token', 'key']):
                    filtered[key] = '***'
                elif 'api_key' in key.lower() and isinstance(value, str):
                    # 保留前8个字符(包括下划线)
                    filtered[key] = value[:8] + '****'
                else:
                    filtered[key] = value
        return filtered

    def detect_malicious_input(self, data: dict) -> bool:
        """检测配置中的恶意输入

        Args:
            data: 要检查的输入数据

        Returns:
            bool: 如果检测到恶意输入返回True，否则返回False
        """
        sql_injection_patterns = [
            r';.*DROP\s+TABLE',
            r';.*DELETE\s+FROM',
            r';.*INSERT\s+INTO',
            r';.*UPDATE\s+\w+\s+SET',
            r';.*--',
            r';.*/\*.*\*/'
        ]

        xss_patterns = [
            r'<script.*?>.*?</script>',
            r'onerror\s*=',
            r'onload\s*='
        ]

        # 添加路径遍历检测模式
        path_traversal_patterns = [
            r'\.\./\.\./',
            r'\.\.\\\.\.\\',
            r'etc/passwd',
            r'etc\\passwd',
            r'\.\.%2f\.\.%2f',  # URL编码的路径遍历
            r'\.\.%5c\.\.%5c'  # URL编码的反斜杠路径遍历
        ]

        def _check(value):
            if isinstance(value, str):
                # 检查SQL注入
                for pattern in sql_injection_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return True
                # 检查XSS
                for pattern in xss_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return True
                # 检查路径遍历
                for pattern in path_traversal_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return True
            elif isinstance(value, dict):
                return any(_check(v) for v in value.values())
            elif isinstance(value, list):
                return any(_check(v) for v in value)
            return False

        return _check(data)

class SecurityService:
    """安全服务 - 单例模式实现"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化安全服务"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._secret_key = None
        self._audit_log = []
        self._audit_lock = threading.Lock()
        
    def _sign_data(self, data: bytes) -> bytes:
        """签名数据"""
        if not self._secret_key:
            raise ValueError("Secret key not set")
        return hmac.new(self._secret_key, data, hashlib.sha256).digest()
    
    def _verify_signature(self, data: bytes, signature: bytes) -> bool:
        """验证签名"""
        expected = self._sign_data(data)
        return hmac.compare_digest(expected, signature)
    
    def audit(self, action: str, details: Dict[str, Any]) -> None:
        """记录审计日志"""
        import logging
        
        with self._audit_lock:
            audit_entry = {
                'action': action,
                'details': details,
                'timestamp': time.time()
            }
            self._audit_log.append(audit_entry)
            
            # 限制日志大小
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]
            
            # 记录到日志系统
            logger = logging.getLogger(__name__)
            logger.info(f"Security audit: {action} - {details}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置安全性"""
        # 检查敏感配置项
        sensitive_keys = ['password', 'secret', 'key', 'token']
        for key in sensitive_keys:
            if key in str(config).lower():
                self.audit('config_validation_failed', {
                    'reason': 'sensitive_key_found',
                    'key': key
                })
                return False
        return True
    
    def check_access(self, resource: str, user: str) -> bool:
        """检查访问权限"""
        # 简单实现，实际应该有完整的权限系统
        self.audit('access_check', {
            'resource': resource,
            'user': user,
            'result': True
        })
        return True

def get_default_security_service() -> SecurityManager:
    """获取默认安全服务实例"""
    return SecurityManager()

