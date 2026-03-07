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


from datetime import datetime, timedelta

import pytest

from src.data.governance.enterprise_governance import (
    AuditType,
    ComplianceManager,
    DataPolicyManager,
    EnterpriseDataGovernanceManager,
    EnforcementLevel,
    PolicyType,
    RegulationType,
    RiskLevel,
    SecurityAuditor,
)


def test_data_policy_manager_update_and_deactivate_records_history():
    manager = DataPolicyManager()

    policy = manager.create_policy(
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="初始描述",
        rules=[{"rule_type": "auth", "method": "rbac"}],
        enforcement_level=EnforcementLevel.HIGH,
    )

    history_length = len(manager.policy_history)
    updated = manager.update_policy(policy.policy_id, description="更新后的描述")

    assert updated.description == "更新后的描述"
    assert len(manager.policy_history) == history_length + 1
    assert manager.policy_history[-1]["old_policy"]["description"] == "初始描述"

    assert manager.deactivate_policy(policy.policy_id) is True
    assert manager.policies[policy.policy_id].status == "inactive"


def test_compliance_manager_verify_compliance_handles_implementation():
    manager = ComplianceManager()
    requirement = manager.add_requirement(
        regulation_name=RegulationType.GDPR,
        requirement_type="masking",
        description="脱敏要求",
    )

    summary_before = manager.verify_compliance(RegulationType.GDPR)
    assert summary_before["status"] == "non_compliant"
    assert summary_before["total_requirements"] == 1

    implemented = manager.implement_requirement(
        requirement.requirement_id, {"steps": ["enforce masking"]}
    )
    assert implemented is True
    assert manager.compliance_status[requirement.requirement_id]["status"] == "implemented"

    summary_after = manager.verify_compliance(RegulationType.GDPR)
    assert summary_after["status"] == "compliant"
    assert summary_after["implemented_requirements"] == 1
    assert summary_after["compliance_rate"] == 100


def test_security_auditor_collects_high_risk_findings():
    auditor = SecurityAuditor()
    audit_id = auditor.schedule_audit(
        audit_type=AuditType.SYSTEM,
        auditor="Alice",
        scheduled_date=datetime.now() + timedelta(days=1),
    )

    auditor.conduct_audit(
        audit_id=audit_id,
        findings=[
            {"issue": "encryption disabled", "risk_level": "critical"},
            {"issue": "docs outdated", "risk_level": "low"},
        ],
        risk_level=RiskLevel.CRITICAL,
        recommendations=["enable encryption"],
    )

    report = auditor.get_audit_report(audit_id)
    assert report["risk_level"] == RiskLevel.CRITICAL.value
    assert report["findings_count"] == 2

    high_risk = auditor.get_high_risk_findings()
    assert len(high_risk) == 1
    assert high_risk[0]["finding"]["risk_level"] == "critical"


def test_governance_manager_generate_report_contains_scores_and_recommendations():
    manager = EnterpriseDataGovernanceManager()
    framework = manager.initialize_governance_framework()

    assert framework["framework_status"] == "initialized"
    assert framework["policies_created"] == 2

    manager.compliance_manager.compliance_status["gdpr"] = {"compliance_rate": 80}

    audit_id = manager.security_auditor.schedule_audit(
        audit_type=AuditType.ACCESS,
        auditor="Bob",
        scheduled_date=datetime.now() + timedelta(days=2),
    )
    manager.security_auditor.conduct_audit(
        audit_id=audit_id,
        findings=[{"risk_level": "high", "issue": "role misconfiguration"}],
        risk_level=RiskLevel.HIGH,
        recommendations=["tighten roles"],
    )

    report = manager.generate_governance_report()
    assert report["active_policies_count"] >= 2
    assert report["high_risk_findings_count"] == 1
    assert 0 < report["overall_governance_score"] <= 100
    assert any("提高" in rec for rec in report["recommendations"])
    assert any("高风险" in rec for rec in report["recommendations"])

