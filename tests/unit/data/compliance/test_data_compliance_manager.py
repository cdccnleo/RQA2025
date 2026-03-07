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

from src.data.compliance.data_compliance_manager import DataComplianceManager


class StubPolicyManager:
    def __init__(self, existing=None):
        self.existing = existing or {}
        self.registered = []

    def register_policy(self, policy):
        self.registered.append(policy)
        return True

    def get_policy(self, policy_id):
        return self.existing.get(policy_id)

    def list_policies(self):
        return self.existing


class StubComplianceChecker:
    def __init__(self, result=None, bulk_result=None, trade_result=None):
        self.result = result or {}
        self.bulk_result = bulk_result or {}
        self.trade_result = trade_result or {}
        self.last_check_args = None
        self.last_bulk_args = None

    def check(self, data, policy_id=None):
        self.last_check_args = (data, policy_id)
        return self.result

    def check_bulk_data(self, data_list, policy_id=None):
        self.last_bulk_args = (data_list, policy_id)
        return self.bulk_result

    def check_trading_compliance(self, trade_data):
        return self.trade_result


class StubPrivacyProtector:
    def protect(self, data, level="standard"):
        return {"protected": data, "level": level}


@pytest.fixture()
def manager():
    mgr = DataComplianceManager()
    mgr.policy_manager = StubPolicyManager()
    mgr.compliance_checker = StubComplianceChecker()
    mgr.privacy_protector = StubPrivacyProtector()
    return mgr


def test_register_policy_calls_policy_manager(manager):
    policy = {"id": "sample"}
    assert manager.register_policy(policy) is True
    assert manager.policy_manager.registered == [policy]


def test_generate_compliance_report_with_issues():
    issues = ["缺失字段: email", "类型错误: balance"]
    checker = StubComplianceChecker(
        result={
            "compliance": False,
            "issues": issues,
            "checked_at": "2025-01-01T00:00:00",
            "check_duration_seconds": 0.5,
            "check_type": "enhanced",
        }
    )

    mgr = DataComplianceManager()
    mgr.policy_manager = StubPolicyManager()
    mgr.compliance_checker = checker
    mgr.privacy_protector = StubPrivacyProtector()

    report = mgr.generate_compliance_report({"dummy": True}, policy_id="user_policy")

    assert report["policy_id"] == "user_policy"
    assert report["compliance"] is False
    assert len(report["recommendations"]) >= 2
    assert "缺失字段" in report["issues"][0]
    assert report["check_type"] == "enhanced"


def test_generate_bulk_compliance_report_assesses_severity():
    bulk_result = {
        "total_records": 100,
        "compliant_records": 70,
        "non_compliant_records": 30,
        "compliance_rate": 0.7,
        "all_issues": ["issue1"] * 15,
        "checked_at": "2025-01-02T00:00:00",
    }
    checker = StubComplianceChecker(bulk_result=bulk_result)

    mgr = DataComplianceManager()
    mgr.policy_manager = StubPolicyManager()
    mgr.compliance_checker = checker
    mgr.privacy_protector = StubPrivacyProtector()

    report = mgr.generate_bulk_compliance_report([{"item": 1}], policy_id="bulk_policy")

    assert report["severity_assessment"] == "concerning"
    assert "sample_issues" in report and len(report["sample_issues"]) == 10
    assert any("合规率" in msg for msg in report["recommendations"])


def test_setup_default_policies_registers_missing():
    mgr = DataComplianceManager()
    # first policy missing, second already exists
    existing = {"trade_data_policy": {"id": "trade_data_policy"}}
    mgr.policy_manager = StubPolicyManager(existing=existing)
    mgr.compliance_checker = StubComplianceChecker()
    mgr.privacy_protector = StubPrivacyProtector()

    mgr.setup_default_policies()

    registered_ids = {policy["id"] for policy in mgr.policy_manager.registered}
    assert "user_data_policy" in registered_ids
    assert "trade_data_policy" not in registered_ids  # already existed


def test_audit_compliance_status_generates_summary():
    policies = {
        "p1": {"id": "p1", "category": "user", "active": True},
        "p2": {"id": "p2", "category": "trade", "active": False},
    }
    mgr = DataComplianceManager()
    mgr.policy_manager = StubPolicyManager(existing=policies)
    mgr.compliance_checker = StubComplianceChecker()
    mgr.privacy_protector = StubPrivacyProtector()

    audit = mgr.audit_compliance_status()

    assert audit["total_policies"] == 2
    assert audit["active_policies"] == 1
    assert audit["policies_by_category"]["user"] == 1
    assert audit["policies_by_category"]["trade"] == 1


def test_protect_privacy_uses_privacy_protector(manager):
    result = manager.protect_privacy({"secret": True}, level="high")
    assert result == {"protected": {"secret": True}, "level": "high"}

