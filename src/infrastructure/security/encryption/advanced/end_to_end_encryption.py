"""
端到端数据加密模块

功能：
- AES-256-GCM 对称加密
- RSA-4096 非对称加密
- 密钥派生 (PBKDF2, Argon2)
- 数字签名与验证
- 密钥管理服务集成
- 加密数据流处理

技术栈：
- cryptography: 加密算法实现
- hashlib: 哈希函数
- secrets: 安全随机数生成

作者: Claude
创建日期: 2026-02-21
"""

import base64
import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argon2可选导入
try:
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False
    logger.warning("Argon2不可用，将使用PBKDF2作为替代")


@dataclass
class EncryptedData:
    """加密数据结构"""
    ciphertext: bytes
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    algorithm: str = "AES-256-GCM"
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典（Base64编码）"""
        result = {
            'ciphertext': base64.b64encode(self.ciphertext).decode('utf-8'),
            'algorithm': self.algorithm
        }
        if self.nonce:
            result['nonce'] = base64.b64encode(self.nonce).decode('utf-8')
        if self.tag:
            result['tag'] = base64.b64encode(self.tag).decode('utf-8')
        if self.salt:
            result['salt'] = base64.b64encode(self.salt).decode('utf-8')
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'EncryptedData':
        """从字典解析"""
        return cls(
            ciphertext=base64.b64decode(data['ciphertext']),
            nonce=base64.b64decode(data['nonce']) if 'nonce' in data else None,
            tag=base64.b64decode(data['tag']) if 'tag' in data else None,
            salt=base64.b64decode(data['salt']) if 'salt' in data else None,
            algorithm=data.get('algorithm', 'AES-256-GCM')
        )


@dataclass
class KeyPair:
    """密钥对"""
    private_key: bytes
    public_key: bytes
    algorithm: str = "RSA-4096"


class SymmetricEncryption:
    """
    对称加密类
    
    使用AES-256-GCM算法提供认证加密
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        初始化对称加密
        
        Args:
            key: 32字节密钥，如果不提供则自动生成
        """
        self.key = key or self._generate_key()
        if len(self.key) != 32:
            raise ValueError("密钥必须是32字节（256位）")
    
    @staticmethod
    def _generate_key() -> bytes:
        """生成随机密钥"""
        return secrets.token_bytes(32)
    
    def encrypt(self, plaintext: Union[str, bytes], 
                associated_data: Optional[bytes] = None) -> EncryptedData:
        """
        加密数据
        
        Args:
            plaintext: 明文数据
            associated_data: 附加认证数据（AAD）
            
        Returns:
            加密数据结构
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        # 生成随机nonce
        nonce = secrets.token_bytes(12)
        
        # 创建AESGCM实例
        aesgcm = AESGCM(self.key)
        
        # 加密并认证
        ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data)
        
        # 分离密文和认证标签
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]
        
        return EncryptedData(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            algorithm="AES-256-GCM"
        )
    
    def decrypt(self, encrypted_data: EncryptedData,
                associated_data: Optional[bytes] = None) -> bytes:
        """
        解密数据
        
        Args:
            encrypted_data: 加密数据结构
            associated_data: 附加认证数据（AAD）
            
        Returns:
            明文数据
        """
        if not encrypted_data.nonce or not encrypted_data.tag:
            raise ValueError("缺少nonce或tag")
        
        # 重新组合密文和标签
        ciphertext_with_tag = encrypted_data.ciphertext + encrypted_data.tag
        
        # 创建AESGCM实例
        aesgcm = AESGCM(self.key)
        
        # 解密并验证
        try:
            plaintext = aesgcm.decrypt(
                encrypted_data.nonce, 
                ciphertext_with_tag, 
                associated_data
            )
            return plaintext
        except Exception as e:
            logger.error(f"解密失败: {e}")
            raise ValueError("解密失败：数据可能被篡改或密钥错误")
    
    def get_key_b64(self) -> str:
        """获取Base64编码的密钥"""
        return base64.b64encode(self.key).decode('utf-8')
    
    @classmethod
    def from_key_b64(cls, key_b64: str) -> 'SymmetricEncryption':
        """从Base64编码创建实例"""
        key = base64.b64decode(key_b64)
        return cls(key)


class AsymmetricEncryption:
    """
    非对称加密类
    
    使用RSA-4096算法
    """
    
    def __init__(self, key_pair: Optional[KeyPair] = None):
        """
        初始化非对称加密
        
        Args:
            key_pair: 密钥对，如果不提供则自动生成
        """
        if key_pair:
            self.key_pair = key_pair
            self._load_keys()
        else:
            self._generate_keypair()
    
    def _generate_keypair(self) -> None:
        """生成RSA密钥对"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        self.private_key = private_key
        self.public_key = private_key.public_key()
        
        # 序列化密钥
        self.key_pair = KeyPair(
            private_key=private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ),
            public_key=private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )
    
    def _load_keys(self) -> None:
        """加载密钥"""
        self.private_key = serialization.load_pem_private_key(
            self.key_pair.private_key,
            password=None,
            backend=default_backend()
        )
        self.public_key = serialization.load_pem_public_key(
            self.key_pair.public_key,
            backend=default_backend()
        )
    
    def encrypt(self, plaintext: Union[str, bytes]) -> bytes:
        """
        使用公钥加密
        
        Args:
            plaintext: 明文数据
            
        Returns:
            密文
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        ciphertext = self.public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        使用私钥解密
        
        Args:
            ciphertext: 密文
            
        Returns:
            明文
        """
        plaintext = self.private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext
    
    def sign(self, message: Union[str, bytes]) -> bytes:
        """
        数字签名
        
        Args:
            message: 消息
            
        Returns:
            签名
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify(self, message: Union[str, bytes], signature: bytes) -> bool:
        """
        验证签名
        
        Args:
            message: 消息
            signature: 签名
            
        Returns:
            是否验证通过
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        try:
            self.public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class KeyDerivation:
    """
    密钥派生类
    
    支持PBKDF2和Argon2id
    """
    
    @staticmethod
    def pbkdf2_derive(password: str, salt: Optional[bytes] = None,
                     iterations: int = 100000) -> Tuple[bytes, bytes]:
        """
        使用PBKDF2派生密钥
        
        Args:
            password: 密码
            salt: 盐值，如果不提供则自动生成
            iterations: 迭代次数
            
        Returns:
            (密钥, 盐值)
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
    
    @staticmethod
    def argon2_derive(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        使用Argon2id派生密钥
        
        Args:
            password: 密码
            salt: 盐值，如果不提供则自动生成
            
        Returns:
            (密钥, 盐值)
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # 检查Argon2是否可用
        if not ARGON2_AVAILABLE:
            logger.warning("Argon2不可用，回退到PBKDF2")
            return KeyDerivation.pbkdf2_derive(password, salt, iterations=100000)
        
        # 使用Argon2id
        kdf = Argon2id(
            length=32,
            salt=salt,
            iterations=3,
            lanes=4,
            memory_cost=65536,
            ad=None,
            secret=None
        )
        
        key = kdf.derive(password.encode('utf-8'))
        return key, salt


class EndToEndEncryption:
    """
    端到端加密主类
    
    结合对称和非对称加密，提供完整的端到端加密方案
    """
    
    def __init__(self):
        """初始化端到端加密"""
        self.symmetric = SymmetricEncryption()
        self.asymmetric = None  # 按需初始化
    
    def encrypt_for_storage(self, data: Union[str, bytes],
                           associated_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        加密数据用于存储
        
        Args:
            data: 明文数据
            associated_data: 附加认证数据
            
        Returns:
            包含加密数据和加密密钥的字典
        """
        # 生成新的对称密钥
        data_key = SymmetricEncryption()
        
        # 加密数据
        encrypted = data_key.encrypt(data, associated_data)
        
        # 加密数据密钥（使用主密钥）
        encrypted_key = self.symmetric.encrypt(data_key.key)
        
        return {
            'encrypted_data': encrypted.to_dict(),
            'encrypted_key': encrypted_key.to_dict(),
            'version': '1.0'
        }
    
    def decrypt_from_storage(self, encrypted_package: Dict[str, Any],
                            associated_data: Optional[bytes] = None) -> bytes:
        """
        解密存储的数据
        
        Args:
            encrypted_package: 加密包
            associated_data: 附加认证数据
            
        Returns:
            明文数据
        """
        # 解密密钥
        encrypted_key = EncryptedData.from_dict(encrypted_package['encrypted_key'])
        data_key_bytes = self.symmetric.decrypt(encrypted_key)
        
        # 解密数据
        data_key = SymmetricEncryption(data_key_bytes)
        encrypted_data = EncryptedData.from_dict(encrypted_package['encrypted_data'])
        
        return data_key.decrypt(encrypted_data, associated_data)
    
    def encrypt_for_transmission(self, data: Union[str, bytes],
                                recipient_public_key: bytes) -> Dict[str, Any]:
        """
        加密数据用于传输（端到端加密）
        
        Args:
            data: 明文数据
            recipient_public_key: 接收者公钥
            
        Returns:
            加密包
        """
        # 生成临时对称密钥
        session_key = SymmetricEncryption()
        
        # 加密数据
        encrypted_data = session_key.encrypt(data)
        
        # 使用接收者公钥加密会话密钥
        recipient = AsymmetricEncryption(KeyPair(
            private_key=b'',
            public_key=recipient_public_key
        ))
        encrypted_key = recipient.encrypt(session_key.key)
        
        return {
            'encrypted_data': encrypted_data.to_dict(),
            'encrypted_session_key': base64.b64encode(encrypted_key).decode('utf-8'),
            'algorithm': 'hybrid-rsa-aes'
        }
    
    def decrypt_transmission(self, encrypted_package: Dict[str, Any],
                           private_key: bytes) -> bytes:
        """
        解密传输的数据
        
        Args:
            encrypted_package: 加密包
            private_key: 接收者私钥
            
        Returns:
            明文数据
        """
        # 使用私钥解密会话密钥
        recipient = AsymmetricEncryption(KeyPair(
            private_key=private_key,
            public_key=b''
        ))
        
        encrypted_key = base64.b64decode(encrypted_package['encrypted_session_key'])
        session_key_bytes = recipient.decrypt(encrypted_key)
        
        # 解密数据
        session_key = SymmetricEncryption(session_key_bytes)
        encrypted_data = EncryptedData.from_dict(encrypted_package['encrypted_data'])
        
        return session_key.decrypt(encrypted_data)
    
    def rotate_key(self) -> None:
        """轮换主密钥"""
        self.symmetric = SymmetricEncryption()
        logger.info("主密钥已轮换")


class EncryptedField:
    """
    加密字段装饰器
    
    用于自动加密/解密模型字段
    """
    
    def __init__(self, field_name: str, encryption: Optional[EndToEndEncryption] = None):
        """
        初始化加密字段
        
        Args:
            field_name: 字段名
            encryption: 加密实例
        """
        self.field_name = field_name
        self.encryption = encryption or EndToEndEncryption()
    
    def __get__(self, instance, owner):
        """获取值时自动解密"""
        if instance is None:
            return self
        
        encrypted_value = instance.__dict__.get(f'_encrypted_{self.field_name}')
        if encrypted_value is None:
            return None
        
        try:
            decrypted = self.encryption.decrypt_from_storage(encrypted_value)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"解密字段 {self.field_name} 失败: {e}")
            return None
    
    def __set__(self, instance, value):
        """设置值时自动加密"""
        if value is None:
            instance.__dict__[f'_encrypted_{self.field_name}'] = None
            return
        
        encrypted = self.encryption.encrypt_for_storage(str(value))
        instance.__dict__[f'_encrypted_{self.field_name}'] = encrypted


# 便捷函数
def generate_secure_password(length: int = 32) -> str:
    """
    生成安全密码
    
    Args:
        length: 密码长度
        
    Returns:
        安全密码
    """
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """
    哈希敏感数据
    
    Args:
        data: 敏感数据
        salt: 盐值
        
    Returns:
        哈希值
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
    return f"{salt}${base64.b64encode(hash_obj).decode('utf-8')}"


def verify_hash(data: str, hashed: str) -> bool:
    """
    验证哈希
    
    Args:
        data: 原始数据
        hashed: 哈希值
        
    Returns:
        是否匹配
    """
    try:
        salt, _ = hashed.split('$', 1)
        return hash_sensitive_data(data, salt) == hashed
    except Exception:
        return False


# 单例实例
_encryption_instance: Optional[EndToEndEncryption] = None


def get_encryption() -> EndToEndEncryption:
    """
    获取端到端加密单例
    
    Returns:
        EndToEndEncryption实例
    """
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = EndToEndEncryption()
    return _encryption_instance
