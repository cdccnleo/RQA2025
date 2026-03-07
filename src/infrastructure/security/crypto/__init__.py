"""
加密模块
提供数据加密、密钥管理和安全算法支持
"""

from .algorithms import (
    EncryptionAlgorithm,
    AESEncryption,
    RSAEncryption,
    HashAlgorithm,
    HMACAlgorithm,
    Base64Encoding,
    EncryptionAlgorithmFactory
)

from .key_management import (
    KeyMetadata,
    KeyStore,
    KeyGenerator,
    KeyRotationPolicy,
    KeyManager
)

__all__ = [
    # 算法组件
    'EncryptionAlgorithm',
    'AESEncryption',
    'RSAEncryption',
    'HashAlgorithm',
    'HMACAlgorithm',
    'Base64Encoding',
    'EncryptionAlgorithmFactory',

    # 密钥管理组件
    'KeyMetadata',
    'KeyStore',
    'KeyGenerator',
    'KeyRotationPolicy',
    'KeyManager'
]
