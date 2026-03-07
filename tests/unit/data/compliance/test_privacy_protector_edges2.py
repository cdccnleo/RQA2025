"""
边界测试：privacy_protector.py
测试边界情况和异常场景
"""
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


import pytest
import hashlib
from src.data.compliance.privacy_protector import PrivacyProtector


def test_privacy_protector_init():
    """测试 PrivacyProtector（初始化）"""
    protector = PrivacyProtector()
    
    assert protector is not None


def test_privacy_protector_protect_non_string():
    """测试 PrivacyProtector（保护，非字符串）"""
    protector = PrivacyProtector()
    
    result = protector.protect(123)
    
    assert result == 123


def test_privacy_protector_protect_dict():
    """测试 PrivacyProtector（保护，字典）"""
    protector = PrivacyProtector()
    data = {"key": "value"}
    
    result = protector.protect(data)
    
    assert result == data


def test_privacy_protector_protect_list():
    """测试 PrivacyProtector（保护，列表）"""
    protector = PrivacyProtector()
    data = [1, 2, 3]
    
    result = protector.protect(data)
    
    assert result == data


def test_privacy_protector_protect_none_level():
    """测试 PrivacyProtector（保护，None级别）"""
    protector = PrivacyProtector()
    
    result = protector.protect("test@example.com", level=None)
    
    # None级别应该被转换为standard
    assert result != "test@example.com"
    assert "*" in result


def test_privacy_protector_protect_invalid_level():
    """测试 PrivacyProtector（保护，无效级别）"""
    protector = PrivacyProtector()
    
    result = protector.protect("test@example.com", level="invalid")
    
    # 无效级别应该被转换为standard
    assert result != "test@example.com"
    assert "*" in result


def test_privacy_protector_protect_none():
    """测试 PrivacyProtector（保护，none级别）"""
    protector = PrivacyProtector()
    data = "sensitive_data"
    
    result = protector.protect(data, level="none")
    
    assert result == data


def test_privacy_protector_protect_encrypted():
    """测试 PrivacyProtector（保护，encrypted级别）"""
    protector = PrivacyProtector()
    data = "sensitive_data"
    
    result = protector.protect(data, level="encrypted")
    
    assert result != data
    assert isinstance(result, str)
    assert len(result) == 64  # SHA256 hex digest length
    assert result == hashlib.sha256(data.encode()).hexdigest()


def test_privacy_protector_protect_standard():
    """测试 PrivacyProtector（保护，standard级别）"""
    protector = PrivacyProtector()
    data = "sensitive_data"
    
    result = protector.protect(data, level="standard")
    
    assert result != data
    assert "*" in result


def test_privacy_protector_mask_data_empty():
    """测试 PrivacyProtector（脱敏，空字符串）"""
    protector = PrivacyProtector()
    
    result = protector._mask_data("")
    
    assert result == ""


def test_privacy_protector_mask_data_phone():
    """测试 PrivacyProtector（脱敏，手机号）"""
    protector = PrivacyProtector()
    phone = "13812345678"
    
    result = protector._mask_data(phone)
    
    assert result.startswith("13")
    assert result.endswith("78")
    assert "*******" in result
    assert len(result) == len(phone)


def test_privacy_protector_mask_data_phone_invalid():
    """测试 PrivacyProtector（脱敏，无效手机号）"""
    protector = PrivacyProtector()
    phone = "12345678901"  # 不是以1开头的11位数字
    
    result = protector._mask_data(phone)
    
    # 应该使用默认模式
    assert result != phone


def test_privacy_protector_mask_data_email():
    """测试 PrivacyProtector（脱敏，邮箱）"""
    protector = PrivacyProtector()
    email = "test@example.com"
    
    result = protector._mask_data(email)
    
    assert "@" in result
    assert "***" in result
    assert ".com" in result or "com" in result


def test_privacy_protector_mask_data_email_short_username():
    """测试 PrivacyProtector（脱敏，邮箱，短用户名）"""
    protector = PrivacyProtector()
    email = "a@example.com"
    
    result = protector._mask_data(email)
    
    assert "@" in result
    assert "***" in result


def test_privacy_protector_mask_data_id_card():
    """测试 PrivacyProtector（脱敏，身份证号）"""
    protector = PrivacyProtector()
    id_card = "110101199001011234"
    
    result = protector._mask_data(id_card)
    
    assert result.startswith("110101")
    assert result.endswith("1234")
    assert "****" in result


def test_privacy_protector_mask_data_id_card_with_x():
    """测试 PrivacyProtector（脱敏，身份证号，带X）"""
    protector = PrivacyProtector()
    id_card = "11010119900101123X"
    
    result = protector._mask_data(id_card)
    
    assert result.startswith("110101")
    assert result.endswith("123X")
    assert "****" in result


def test_privacy_protector_mask_data_credit_card():
    """测试 PrivacyProtector（脱敏，信用卡号）"""
    protector = PrivacyProtector()
    card = "1234567890123456"
    
    result = protector._mask_data(card)
    
    assert result.startswith("1234")
    assert result.endswith("3456")
    assert "****" in result


def test_privacy_protector_mask_data_bank_account():
    """测试 PrivacyProtector（脱敏，银行账号）"""
    protector = PrivacyProtector()
    account = "1234567890123456789"
    
    result = protector._mask_data(account)
    
    assert result.startswith("1234")
    assert result.endswith(account[-4:])
    assert "****" in result


def test_privacy_protector_mask_data_address():
    """测试 PrivacyProtector（脱敏，地址）"""
    protector = PrivacyProtector()
    address = "北京市朝阳区某某街道123号"
    
    result = protector._mask_data(address)
    
    assert result.startswith("北京")
    assert result.endswith("号")
    assert "****" in result


def test_privacy_protector_mask_data_name():
    """测试 PrivacyProtector（脱敏，姓名）"""
    protector = PrivacyProtector()
    name = "张三"
    
    result = protector._mask_data(name)
    
    assert result == "张*"


def test_privacy_protector_mask_data_short_string():
    """测试 PrivacyProtector（脱敏，短字符串）"""
    protector = PrivacyProtector()
    short = "test"
    
    result = protector._mask_data(short)
    
    assert result == "****"
    assert len(result) == len(short)


def test_privacy_protector_mask_data_long_string():
    """测试 PrivacyProtector（脱敏，长字符串）"""
    protector = PrivacyProtector()
    long_str = "这是一个很长的字符串用于测试脱敏功能"
    
    result = protector._mask_data(long_str)
    
    assert result.startswith("这是")
    assert result.endswith("功能")
    assert "****" in result


def test_privacy_protector_mask_data_medium_string():
    """测试 PrivacyProtector（脱敏，中等长度字符串）"""
    protector = PrivacyProtector()
    medium = "中等长度"
    
    result = protector._mask_data(medium)
    
    # 长度在4-8之间，应该使用默认模式
    assert result != medium
    assert "*" in result


def test_privacy_protector_protect_phone_standard():
    """测试 PrivacyProtector（保护手机号，standard级别）"""
    protector = PrivacyProtector()
    phone = "13812345678"
    
    result = protector.protect(phone, level="standard")
    
    assert result.startswith("13")
    assert result.endswith("78")
    assert "*******" in result


def test_privacy_protector_protect_email_standard():
    """测试 PrivacyProtector（保护邮箱，standard级别）"""
    protector = PrivacyProtector()
    email = "user@example.com"
    
    result = protector.protect(email, level="standard")
    
    assert "@" in result
    assert "***" in result


def test_privacy_protector_protect_email_encrypted():
    """测试 PrivacyProtector（保护邮箱，encrypted级别）"""
    protector = PrivacyProtector()
    email = "user@example.com"
    
    result = protector.protect(email, level="encrypted")
    
    assert result == hashlib.sha256(email.encode()).hexdigest()
    assert len(result) == 64


def test_privacy_protector_protect_id_card_standard():
    """测试 PrivacyProtector（保护身份证号，standard级别）"""
    protector = PrivacyProtector()
    id_card = "110101199001011234"
    
    result = protector.protect(id_card, level="standard")
    
    assert result.startswith("110101")
    assert result.endswith("1234")
    assert "****" in result


def test_privacy_protector_protect_id_card_encrypted():
    """测试 PrivacyProtector（保护身份证号，encrypted级别）"""
    protector = PrivacyProtector()
    id_card = "110101199001011234"
    
    result = protector.protect(id_card, level="encrypted")
    
    assert result == hashlib.sha256(id_card.encode()).hexdigest()
    assert len(result) == 64


def test_privacy_protector_protect_name_standard():
    """测试 PrivacyProtector（保护姓名，standard级别）"""
    protector = PrivacyProtector()
    name = "张三"
    
    result = protector.protect(name, level="standard")
    
    assert result == "张*"


def test_privacy_protector_protect_name_encrypted():
    """测试 PrivacyProtector（保护姓名，encrypted级别）"""
    protector = PrivacyProtector()
    name = "张三"
    
    result = protector.protect(name, level="encrypted")
    
    assert result == hashlib.sha256(name.encode()).hexdigest()
    assert len(result) == 64
