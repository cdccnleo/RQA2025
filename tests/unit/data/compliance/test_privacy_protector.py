import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import hashlib

import pytest

from src.data.compliance.privacy_protector import PrivacyProtector


@pytest.fixture()
def protector():
    return PrivacyProtector()


@pytest.mark.parametrize(
    "level,expected_suffix",
    [
        ("encrypted", hashlib.sha256("secret".encode()).hexdigest()),
        ("none", "secret"),
    ],
)
def test_protect_handles_encrypted_and_none_levels(protector, level, expected_suffix):
    result = protector.protect("secret", level=level)
    assert result == expected_suffix


def test_protect_defaults_to_standard_when_invalid_level(protector):
    result = protector.protect("john.doe@example.com", level="invalid")
    assert result.startswith("j***@")
    assert result.endswith(".com")


@pytest.mark.parametrize(
    "raw,expected_pattern",
    [
        ("13812345678", ("13", "78")),
        ("john.doe@example.com", ("j***@", ".com")),
        ("1234567890123456", ("1234", "3456")),
        ("IDCARD12345678901", ("ID", "01")),
        ("张三", ("张", "*")),
        ("abcd", ("****", None)),
        ("longaddressXYZ", ("lo", "YZ")),
    ],
)
def test_standard_masking_patterns(protector, raw, expected_pattern):
    masked = protector.protect(raw, level="standard")
    head, tail = expected_pattern
    if tail is None:
        assert masked == head
    else:
        assert masked.startswith(head)
        assert masked.endswith(tail)


def test_protect_non_string_returns_original(protector):
    payload = {"secret": True}
    assert protector.protect(payload) is payload

