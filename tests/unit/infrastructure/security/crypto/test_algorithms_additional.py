import base64
import os

import pytest

from src.infrastructure.security.crypto.algorithms import (
    AESEncryption,
    HMACAlgorithm,
    Base64Encoding,
)


@pytest.fixture
def aes_key():
    return os.urandom(32)  # 256-bit key


def test_aes_encrypt_decrypt_padding_error(aes_key):
    aes = AESEncryption(mode="CBC", key_size=256)
    plaintext = b"16-bytes-block!!"  # 16 bytes so padding will add a full block
    encrypted = aes.encrypt(plaintext, aes_key)

    with pytest.raises(ValueError):
        aes.decrypt(encrypted[:-1], aes_key)


def test_hmac_algorithm_supports_sha512():
    key = os.urandom(32)
    payload = "sensitive-data"
    signature = HMACAlgorithm.generate_hmac(key, payload, algorithm="sha512")
    assert HMACAlgorithm.verify_hmac(key, payload, signature, algorithm="sha512") is True


def test_hmac_algorithm_rejects_unknown_algo():
    with pytest.raises(ValueError):
        HMACAlgorithm.generate_hmac(b"key", "data", algorithm="md5")


def test_base64_encoding_roundtrip():
    binary = os.urandom(64)
    encoded = Base64Encoding.encode(binary)
    assert Base64Encoding.decode(encoded) == binary


def test_base64_encoding_rejects_invalid_data():
    with pytest.raises(Exception):
        Base64Encoding.decode("@@not-base64@@")
