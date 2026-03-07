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


import types

import pandas as pd
import pytest

from src.data.compliance.compliance_checker import ComplianceChecker
from src.data.compliance.data_policy_manager import DataPolicyManager


@pytest.fixture
def policy_manager():
    return DataPolicyManager()


@pytest.fixture
def checker(policy_manager):
    return ComplianceChecker(policy_manager)


def _register_sample_policy(manager, **overrides):
    policy = {
        "id": "sample_policy",
        "name": "Sample Policy",
        "required_fields": ["user_id", "email", "status"],
        "field_types": {"user_id": "integer", "email": "string", "status": "string"},
        "max_field_lengths": {"status": 10},
        "business_rules": {
            "value_ranges": {"amount": {"min": 0, "max": 1000}},
            "enum_values": {"status": ["active", "inactive"]},
        },
    }
    policy.update(overrides)
    assert manager.register_policy(policy) is True
    return policy["id"]


def test_check_requires_policy_id(checker):
    result = checker.check({"field": "value"})
    assert result["compliance"] is False
    assert any("Policy id is required" in issue for issue in result["issues"])


def test_check_reports_missing_policy(policy_manager, checker):
    result = checker.check({"field": "value"}, policy_id="unknown")
    assert result["compliance"] is False
    assert any("Policy 'unknown' not found" in issue for issue in result["issues"])


def test_check_collects_all_validation_issues(policy_manager):
    policy_id = _register_sample_policy(policy_manager, max_field_lengths={"status": 5})
    checker = ComplianceChecker(policy_manager)

    data = {
        "user_id": "abc",
        "status": "suspended",
        "amount": -5,
    }

    result = checker.check(data, policy_id)
    issues = result["issues"]

    assert result["compliance"] is False
    assert any("缺失字段" in issue for issue in issues)
    assert any("类型错误" in issue for issue in issues)
    assert any("超出限制" in issue for issue in issues)
    assert any("值低于最小值" in issue for issue in issues)
    assert any("取值不在允许列表" in issue for issue in issues)


def test_check_validates_ranges_and_enum(policy_manager, checker):
    policy = {
        "name": "RangePolicy",
        "required_fields": ["level", "status"],
        "field_types": {"level": "integer", "status": "string"},
        "business_rules": {
            "value_ranges": {"level": {"min": 10, "max": 20}},
            "enum_values": {"status": ["ok", "pending"]},
        },
        "id": "range_policy",
    }
    policy_manager.register_policy(policy)

    data = {"level": 5, "status": "invalid"}
    result = checker.check(data, policy_id="range_policy")

    assert result["compliance"] is False
    assert any("值低于最小值" in issue for issue in result["issues"])
    assert any("取值不在允许列表" in issue for issue in result["issues"])


def test_check_with_dataframe_uses_first_non_null_value(policy_manager):
    policy_id = _register_sample_policy(
        policy_manager,
        required_fields=["user_id", "status"],
        field_types={"user_id": "integer", "status": "string"},
        max_field_lengths={},
    )
    checker = ComplianceChecker(policy_manager)

    df = pd.DataFrame(
        {
            "user_id": pd.Series([None, 123], dtype=object),
            "status": pd.Series([None, "active"], dtype=object),
        }
    )
    result = checker.check(df, policy_id)

    assert result["compliance"] is True
    assert not result["issues"]


def test_check_accepts_object_attributes(policy_manager):
    policy = {
        "name": "ObjectPolicy",
        "required_fields": ["user_id"],
        "field_types": {"user_id": "integer"},
        "id": "object_policy",
    }
    policy_manager.register_policy(policy)

    checker = ComplianceChecker(policy_manager)
    payload = types.SimpleNamespace(user_id=123)
    result = checker.check(payload, policy_id="object_policy")

    assert result["compliance"] is True


def test_check_bulk_data_aggregates_results(policy_manager):
    policy_id = _register_sample_policy(policy_manager)
    checker = ComplianceChecker(policy_manager)

    data_list = [
        {"user_id": 1, "email": "a@test.com", "status": "active"},
        {"user_id": "bad", "email": "b@test.com", "status": "blocked"},
    ]

    result = checker.check_bulk_data(data_list, policy_id)

    assert result["total_records"] == 2
    assert result["compliant_records"] == 1
    assert result["non_compliant_records"] == 1
    assert result["compliance_rate"] == pytest.approx(0.5)
    assert result["all_issues"]


def test_check_bulk_data_handles_empty_iterable(policy_manager):
    policy = {
        "name": "BulkPolicy",
        "required_fields": [],
        "id": "bulk_policy",
    }
    policy_manager.register_policy(policy)

    checker = ComplianceChecker(policy_manager)
    result = checker.check_bulk_data([], policy_id="bulk_policy")

    assert result["total_records"] == 0
    assert result["compliance_rate"] == 0.0
    assert result["all_issues"] == []


def test_check_trading_compliance_detects_multiple_issues(policy_manager):
    checker = ComplianceChecker(policy_manager)
    trade_data = {"amount": -10, "trade_type": "invalid", "timestamp": "bad_time"}

    result = checker.check_trading_compliance(trade_data)
    issues = result["issues"]

    assert result["compliance"] is False
    assert any("positive number" in issue for issue in issues)
    assert any("invalid trade type" in issue for issue in issues)
    assert any("invalid ISO timestamp" in issue for issue in issues)


def test_check_trading_compliance_detects_timestamp_error_only(policy_manager):
    checker = ComplianceChecker(policy_manager)
    trade = {"amount": 100, "trade_type": "buy", "timestamp": "invalid"}

    result = checker.check_trading_compliance(trade)
    assert result["compliance"] is False
    assert any("invalid ISO timestamp" in issue for issue in result["issues"])


def test_register_policy_rejects_invalid_enforcement_level():
    manager = DataPolicyManager()
    policy = {
        "name": "Invalid Policy",
        "required_fields": ["user_id"],
        "enforcement_level": "impossible",
    }

    assert manager.register_policy(policy) is False

