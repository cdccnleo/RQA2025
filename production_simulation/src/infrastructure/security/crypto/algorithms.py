"""
加密算法组件
提供各种加密算法的实现
"""

import base64
import hashlib
import hmac
from typing import Optional, Union
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os


class EncryptionAlgorithm:
    """加密算法基类"""

    def encrypt(self, data: bytes, key: bytes) -> bytes:
        """加密数据"""
        raise NotImplementedError

    def decrypt(self, data: bytes, key: bytes) -> bytes:
        """解密数据"""
        raise NotImplementedError


class AESEncryption(EncryptionAlgorithm):
    """AES加密算法"""

    def __init__(self, mode: str = "CBC", key_size: int = 256):
        self.mode = mode
        self.key_size = key_size

    def encrypt(self, data: bytes, key: bytes) -> bytes:
        """AES加密"""
        if self.mode == "CBC":
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            # 在明文前附加完整性校验（SHA256）
            digest = hashlib.sha256(data).digest()
            payload = digest + data
            # PKCS7 padding
            block_size = 16
            padding_length = block_size - (len(payload) % block_size)
            if padding_length == 0:
                padding_length = block_size
            padded_data = payload + bytes([padding_length]) * padding_length
            encrypted = encryptor.update(padded_data) + encryptor.finalize()
            return iv + encrypted
        else:
            raise ValueError(f"Unsupported AES mode: {self.mode}")

    def decrypt(self, data: bytes, key: bytes) -> bytes:
        """AES解密"""
        if self.mode == "CBC":
            iv = data[:16]
            encrypted_data = data[16:]
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
            # Remove PKCS7 padding
            padding_length = decrypted_padded[-1]
            if padding_length > 16 or padding_length == 0:
                raise ValueError("Invalid padding")
            for i in range(1, padding_length + 1):
                if decrypted_padded[-i] != padding_length:
                    raise ValueError("Invalid padding")
            payload = decrypted_padded[:-padding_length]
            if len(payload) < hashlib.sha256().digest_size:
                raise ValueError("Invalid ciphertext payload")
            digest = payload[:hashlib.sha256().digest_size]
            plaintext = payload[hashlib.sha256().digest_size:]
            if hashlib.sha256(plaintext).digest() != digest:
                raise ValueError("Integrity check failed")
            return plaintext
        else:
            raise ValueError(f"Unsupported AES mode: {self.mode}")


class RSAEncryption:
    """RSA加密算法"""

    def __init__(self, key_size: int = 2048):
        self.key_size = key_size

    def generate_keypair(self):
        """生成RSA密钥对"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Serialize keys to PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem, public_pem

    def encrypt(self, data: bytes, public_key_pem: bytes) -> bytes:
        """RSA加密"""
        # Load public key from PEM
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )

        encrypted = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted

    def decrypt(self, data: bytes, private_key_pem: bytes) -> bytes:
        """RSA解密"""
        # Load private key from PEM
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )

        decrypted = private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted


class HashAlgorithm:
    """哈希算法"""

    @staticmethod
    def sha256(data: Union[str, bytes]) -> str:
        """SHA256哈希"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def sha512(data: Union[str, bytes]) -> str:
        """SHA512哈希"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha512(data).hexdigest()

    @staticmethod
    def pbkdf2(password: str, salt: bytes, iterations: int = 100000) -> bytes:
        """PBKDF2密钥派生"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(password.encode())


class HMACAlgorithm:
    """HMAC算法"""

    @staticmethod
    def generate_hmac(key: bytes, message: Union[str, bytes], algorithm='sha256') -> str:
        """生成HMAC"""
        if isinstance(message, str):
            message = message.encode('utf-8')

        if algorithm == 'sha256':
            hmac_obj = hmac.new(key, message, hashlib.sha256)
        elif algorithm == 'sha512':
            hmac_obj = hmac.new(key, message, hashlib.sha512)
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")

        return hmac_obj.hexdigest()

    @staticmethod
    def verify_hmac(key: bytes, message: Union[str, bytes], hmac_value: str, algorithm='sha256') -> bool:
        """验证HMAC"""
        expected_hmac = HMACAlgorithm.generate_hmac(key, message, algorithm)
        return hmac.compare_digest(expected_hmac, hmac_value)


class Base64Encoding:
    """Base64编解码"""

    @staticmethod
    def encode(data: bytes) -> str:
        """Base64编码"""
        return base64.b64encode(data).decode('utf-8')

    @staticmethod
    def decode(data: str) -> bytes:
        """Base64解码"""
        return base64.b64decode(data)


class EncryptionAlgorithmFactory:
    """加密算法工厂"""

    @staticmethod
    def create_aes(key_size: int = 256, mode: str = "CBC") -> AESEncryption:
        """创建AES加密器"""
        return AESEncryption(mode=mode, key_size=key_size)

    @staticmethod
    def create_rsa(key_size: int = 2048) -> RSAEncryption:
        """创建RSA加密器"""
        return RSAEncryption(key_size=key_size)

    @staticmethod
    def get_hash_algorithm() -> HashAlgorithm:
        """获取哈希算法"""
        return HashAlgorithm()

    @staticmethod
    def get_hmac_algorithm() -> HMACAlgorithm:
        """获取HMAC算法"""
        return HMACAlgorithm()

    @staticmethod
    def get_base64_encoding() -> Base64Encoding:
        """获取Base64编解码器"""
        return Base64Encoding()