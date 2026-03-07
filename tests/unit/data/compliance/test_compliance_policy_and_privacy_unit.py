#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from src.data.compliance.data_policy_manager import DataPolicyManager
from src.data.compliance.privacy_protector import PrivacyProtector


def test_policy_register_validate_update_delete_and_list():
    mgr = DataPolicyManager()
    # 非法策略
    assert mgr.register_policy(None) is False
    assert mgr.register_policy({"name": "n"}) is False  # 缺少 required_fields
    # 合法策略
    ok = {
        "name": "P1",
        "required_fields": ["id", "email"],
        "enforcement_level": "strict",
        "privacy_level": "standard",
    }
    assert mgr.register_policy(ok) is True
    # 重复注册（同 ID）应失败
    pid = list(mgr.list_policies().keys())[0]
    dup = {"id": pid, "name": "Dup", "required_fields": []}
    assert mgr.register_policy(dup) is False
    # 更新与获取
    assert mgr.update_policy(pid, {"enforcement_level": "moderate"}) is True
    got = mgr.get_policy(pid)
    assert got and got["enforcement_level"] == "moderate"
    # 删除
    assert mgr.delete_policy(pid) is True
    assert mgr.get_policy(pid) is None


@pytest.mark.parametrize(
    "input_str,level,expect_rule",
    [
        ("13800138000", "standard", "mask"),  # 手机号
        ("user@example.com", "standard", "mask"),  # 邮箱
        ("1234567890123456", "standard", "mask"),  # 信用卡/银行号
        ("张三", "standard", "mask"),  # 姓名
        ("abcd", "standard", "mask"),  # 短字符串
        ("longaddressXXXX", "standard", "mask"),  # 地址/长串
        ("plain", "none", "keep"),  # 不处理
    ],
)
def test_privacy_protector_mask_and_levels(input_str, level, expect_rule):
    p = PrivacyProtector()
    out = p.protect(input_str, level=level)
    if expect_rule == "keep":
        assert out == input_str
    else:
        assert out != input_str


def test_privacy_protector_encrypted_mode():
    p = PrivacyProtector()
    s = "secret@example.com"
    h = p.protect(s, level="encrypted")
    assert isinstance(h, str) and len(h) == 64  # sha256 hex 长度


