"""
测试加密算法组件
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.security.crypto.algorithms import (
    AESEncryption,
    RSAEncryption,
    HashAlgorithm,
    HMACAlgorithm,
    Base64Encoding,
    EncryptionAlgorithmFactory
)


class TestAESEncryption:
    """测试AES加密"""

    def test_aes_encrypt_decrypt_cbc(self):
        """测试AES CBC模式的加密解密"""
        aes = AESEncryption(mode="CBC", key_size=256)
        key = b'0' * 32  # 256-bit key
        data = b"Hello, World!"

        encrypted = aes.encrypt(data, key)
        decrypted = aes.decrypt(encrypted, key)

        assert decrypted == data
        assert encrypted != data

    def test_aes_different_keys_produce_different_results(self):
        """测试不同密钥产生不同结果"""
        aes = AESEncryption()
        key1 = b'0' * 32
        key2 = b'1' * 32
        data = b"Test data"

        encrypted1 = aes.encrypt(data, key1)
        encrypted2 = aes.encrypt(data, key2)

        assert encrypted1 != encrypted2

        # Should not be able to decrypt with wrong key
        with pytest.raises(Exception):
            aes.decrypt(encrypted1, key2)


class TestRSAEncryption:
    """测试RSA加密"""

    def test_rsa_generate_keypair(self):
        """测试RSA密钥对生成"""
        rsa_enc = RSAEncryption(key_size=2048)
        private_pem, public_pem = rsa_enc.generate_keypair()

        assert private_pem is not None
        assert public_pem is not None
        assert b"PRIVATE KEY" in private_pem
        assert b"PUBLIC KEY" in public_pem

    def test_rsa_encrypt_decrypt(self):
        """测试RSA加密解密"""
        rsa_enc = RSAEncryption()
        private_key, public_key = rsa_enc.generate_keypair()

        data = b"Hello, RSA World!"

        encrypted = rsa_enc.encrypt(data, public_key)
        decrypted = rsa_enc.decrypt(encrypted, private_key)

        assert decrypted == data
        assert encrypted != data


class TestHashAlgorithm:
    """测试哈希算法"""

    def test_sha256_consistent(self):
        """测试SHA256一致性"""
        data = "Hello, World!"
        hash1 = HashAlgorithm.sha256(data)
        hash2 = HashAlgorithm.sha256(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 character hex string

    def test_sha256_different_inputs(self):
        """测试SHA256对不同输入产生不同结果"""
        hash1 = HashAlgorithm.sha256("data1")
        hash2 = HashAlgorithm.sha256("data2")

        assert hash1 != hash2

    def test_sha512(self):
        """测试SHA512"""
        data = "Test data"
        hash_value = HashAlgorithm.sha512(data)

        assert len(hash_value) == 128  # SHA512 produces 128 character hex string
        assert isinstance(hash_value, str)

    def test_pbkdf2(self):
        """测试PBKDF2密钥派生"""
        password = "password123"
        salt = b"salt123"

        key1 = HashAlgorithm.pbkdf2(password, salt)
        key2 = HashAlgorithm.pbkdf2(password, salt)

        assert key1 == key2  # Same inputs should produce same key
        assert len(key1) == 32  # Default length is 32 bytes


class TestHMACAlgorithm:
    """测试HMAC算法"""

    def test_hmac_sha256(self):
        """测试HMAC SHA256"""
        key = b"secret_key"
        message = "Hello, HMAC!"

        hmac_value = HMACAlgorithm.generate_hmac(key, message, 'sha256')

        assert len(hmac_value) == 64  # SHA256 HMAC hex length
        assert HMACAlgorithm.verify_hmac(key, message, hmac_value, 'sha256')

    def test_hmac_verification_failure(self):
        """测试HMAC验证失败"""
        key = b"secret_key"
        message = "Hello, HMAC!"
        wrong_hmac = "wrong_hmac_value"

        assert not HMACAlgorithm.verify_hmac(key, message, wrong_hmac, 'sha256')

    def test_hmac_sha512(self):
        """测试HMAC SHA512"""
        key = b"secret_key"
        message = "Hello, HMAC!"

        hmac_value = HMACAlgorithm.generate_hmac(key, message, 'sha512')

        assert len(hmac_value) == 128  # SHA512 HMAC hex length
        assert HMACAlgorithm.verify_hmac(key, message, hmac_value, 'sha512')


class TestBase64Encoding:
    """测试Base64编解码"""

    def test_base64_encode_decode(self):
        """测试Base64编解码"""
        data = b"Hello, Base64 World!"

        encoded = Base64Encoding.encode(data)
        decoded = Base64Encoding.decode(encoded)

        assert decoded == data
        assert isinstance(encoded, str)
        assert isinstance(decoded, bytes)


class TestEncryptionAlgorithmFactory:
    """测试加密算法工厂"""

    def test_create_aes(self):
        """测试创建AES加密器"""
        aes = EncryptionAlgorithmFactory.create_aes(key_size=256, mode="CBC")
        assert isinstance(aes, AESEncryption)
        assert aes.key_size == 256
        assert aes.mode == "CBC"

    def test_create_rsa(self):
        """测试创建RSA加密器"""
        rsa_enc = EncryptionAlgorithmFactory.create_rsa(key_size=2048)
        assert isinstance(rsa_enc, RSAEncryption)
        assert rsa_enc.key_size == 2048

    def test_get_hash_algorithm(self):
        """测试获取哈希算法"""
        hash_algo = EncryptionAlgorithmFactory.get_hash_algorithm()
        assert isinstance(hash_algo, HashAlgorithm)

    def test_get_hmac_algorithm(self):
        """测试获取HMAC算法"""
        hmac_algo = EncryptionAlgorithmFactory.get_hmac_algorithm()
        assert isinstance(hmac_algo, HMACAlgorithm)

    def test_get_base64_encoding(self):
        """测试获取Base64编解码器"""
        b64 = EncryptionAlgorithmFactory.get_base64_encoding()
        assert isinstance(b64, Base64Encoding)
