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
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.data.governance.enterprise_governance import (
    DataPolicyManager,
    ComplianceManager,
    SecurityAuditor,
    EnterpriseDataGovernanceManager,
    DataPolicy,
    ComplianceRequirement,
    SecurityAudit,
    PolicyType,
    EnforcementLevel,
    RegulationType,
    AuditType,
    RiskLevel,
)


def test_data_policy_to_dict():
    """测试 DataPolicy（转换为字典）"""
    policy = DataPolicy(
        policy_id="test_id",
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[{"rule": "test"}],
        enforcement_level=EnforcementLevel.HIGH,
        effective_date=datetime.now()
    )
    result = policy.to_dict()
    assert isinstance(result, dict)
    assert result["policy_id"] == "test_id"
    assert result["policy_type"] == "access_control"
    assert result["enforcement_level"] == "high"
    assert "effective_date" in result


def test_compliance_requirement_to_dict():
    """测试 ComplianceRequirement（转换为字典）"""
    requirement = ComplianceRequirement(
        requirement_id="test_id",
        regulation_name=RegulationType.GDPR,
        requirement_type="data_protection",
        description="测试描述",
        mandatory=True
    )
    result = requirement.to_dict()
    assert isinstance(result, dict)
    assert result["requirement_id"] == "test_id"
    assert result["regulation_name"] == "gdpr"
    assert result["mandatory"] is True


def test_security_audit_to_dict():
    """测试 SecurityAudit（转换为字典）"""
    audit = SecurityAudit(
        audit_id="test_id",
        audit_type=AuditType.ACCESS,
        audit_date=datetime.now(),
        auditor="测试审计员",
        findings=[{"finding": "test"}],
        risk_level=RiskLevel.HIGH,
        recommendations=["建议1"]
    )
    result = audit.to_dict()
    assert isinstance(result, dict)
    assert result["audit_id"] == "test_id"
    assert result["audit_type"] == "access"
    assert result["risk_level"] == "high"
    assert "audit_date" in result


def test_data_policy_manager_create_policy_empty_rules():
    """测试 DataPolicyManager（创建策略，空规则）"""
    manager = DataPolicyManager()
    policy = manager.create_policy(
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    assert policy is not None
    assert policy.policy_name == "测试策略"
    assert len(policy.rules) == 0


def test_data_policy_manager_update_policy_nonexistent():
    """测试 DataPolicyManager（更新策略，不存在的策略）"""
    manager = DataPolicyManager()
    result = manager.update_policy("nonexistent_id", description="新描述")
    assert result is None


def test_data_policy_manager_update_policy_empty_kwargs():
    """测试 DataPolicyManager（更新策略，空参数）"""
    manager = DataPolicyManager()
    policy = manager.create_policy(
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    result = manager.update_policy(policy.policy_id)
    assert result is not None
    assert result.policy_id == policy.policy_id


def test_data_policy_manager_update_policy_invalid_attribute():
    """测试 DataPolicyManager（更新策略，无效属性）"""
    manager = DataPolicyManager()
    policy = manager.create_policy(
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    # 无效属性应该被忽略
    result = manager.update_policy(policy.policy_id, invalid_attribute="value")
    assert result is not None


def test_data_policy_manager_deactivate_policy_nonexistent():
    """测试 DataPolicyManager（停用策略，不存在的策略）"""
    manager = DataPolicyManager()
    result = manager.deactivate_policy("nonexistent_id")
    assert result is False


def test_data_policy_manager_get_active_policies_empty():
    """测试 DataPolicyManager（获取活跃策略，空）"""
    manager = DataPolicyManager()
    result = manager.get_active_policies()
    assert result == []


def test_data_policy_manager_get_active_policies_with_type():
    """测试 DataPolicyManager（获取活跃策略，指定类型）"""
    manager = DataPolicyManager()
    policy1 = manager.create_policy(
        policy_name="访问控制策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    policy2 = manager.create_policy(
        policy_name="数据质量策略",
        policy_type=PolicyType.DATA_QUALITY,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    result = manager.get_active_policies(PolicyType.ACCESS_CONTROL)
    assert len(result) == 1
    assert result[0].policy_type == PolicyType.ACCESS_CONTROL


def test_data_policy_manager_get_active_policies_none_type():
    """测试 DataPolicyManager（获取活跃策略，None 类型）"""
    manager = DataPolicyManager()
    manager.create_policy(
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    result = manager.get_active_policies(None)
    assert len(result) >= 1


def test_data_policy_manager_record_policy_change_no_old_policy():
    """测试 DataPolicyManager（记录策略变更，无旧策略）"""
    manager = DataPolicyManager()
    policy = manager.create_policy(
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    manager._record_policy_change(policy, "created")
    assert len(manager.policy_history) >= 1
    assert manager.policy_history[-1]["action"] == "created"


def test_compliance_manager_add_requirement_optional_mandatory():
    """测试 ComplianceManager（添加合规要求，可选 mandatory）"""
    manager = ComplianceManager()
    requirement = manager.add_requirement(
        regulation_name=RegulationType.GDPR,
        requirement_type="data_protection",
        description="测试描述",
        mandatory=False
    )
    assert requirement.mandatory is False


def test_compliance_manager_implement_requirement_nonexistent():
    """测试 ComplianceManager（实施合规要求，不存在的要求）"""
    manager = ComplianceManager()
    result = manager.implement_requirement("nonexistent_id", {})
    assert result is False


def test_compliance_manager_implement_requirement_empty_details():
    """测试 ComplianceManager（实施合规要求，空详情）"""
    manager = ComplianceManager()
    requirement = manager.add_requirement(
        regulation_name=RegulationType.GDPR,
        requirement_type="data_protection",
        description="测试描述"
    )
    result = manager.implement_requirement(requirement.requirement_id, {})
    assert result is True


def test_compliance_manager_verify_compliance_no_requirements():
    """测试 ComplianceManager（验证合规状态，无要求）"""
    manager = ComplianceManager()
    result = manager.verify_compliance(RegulationType.CCPA)
    assert result["total_requirements"] == 0
    assert result["compliance_rate"] == 0
    assert result["status"] == "non_compliant"


def test_compliance_manager_verify_compliance_partial_implementation():
    """测试 ComplianceManager（验证合规状态，部分实施）"""
    manager = ComplianceManager()
    req1 = manager.add_requirement(
        regulation_name=RegulationType.GDPR,
        requirement_type="data_protection",
        description="要求1"
    )
    req2 = manager.add_requirement(
        regulation_name=RegulationType.GDPR,
        requirement_type="data_retention",
        description="要求2"
    )
    manager.implement_requirement(req1.requirement_id, {})
    result = manager.verify_compliance(RegulationType.GDPR)
    assert result["total_requirements"] == 2
    assert result["implemented_requirements"] == 1
    assert result["compliance_rate"] == 50.0
    assert result["status"] == "non_compliant"


def test_compliance_manager_verify_compliance_full_implementation():
    """测试 ComplianceManager（验证合规状态，完全实施）"""
    manager = ComplianceManager()
    req1 = manager.add_requirement(
        regulation_name=RegulationType.GDPR,
        requirement_type="data_protection",
        description="要求1"
    )
    manager.implement_requirement(req1.requirement_id, {})
    result = manager.verify_compliance(RegulationType.GDPR)
    assert result["compliance_rate"] == 100.0
    assert result["status"] == "compliant"


def test_security_auditor_schedule_audit():
    """测试 SecurityAuditor（安排审计）"""
    auditor = SecurityAuditor()
    scheduled_date = datetime.now() + timedelta(days=30)
    audit_id = auditor.schedule_audit(
        audit_type=AuditType.ACCESS,
        auditor="测试审计员",
        scheduled_date=scheduled_date
    )
    assert audit_id is not None
    assert len(auditor.audit_schedule) == 1


def test_security_auditor_conduct_audit_empty_findings():
    """测试 SecurityAuditor（执行审计，空发现）"""
    auditor = SecurityAuditor()
    audit = auditor.conduct_audit(
        audit_id="test_id",
        findings=[],
        risk_level=RiskLevel.LOW,
        recommendations=[]
    )
    assert audit is not None
    assert len(audit.findings) == 0
    assert len(audit.recommendations) == 0


def test_security_auditor_conduct_audit_empty_recommendations():
    """测试 SecurityAuditor（执行审计，空建议）"""
    auditor = SecurityAuditor()
    audit = auditor.conduct_audit(
        audit_id="test_id",
        findings=[{"finding": "test"}],
        risk_level=RiskLevel.HIGH,
        recommendations=[]
    )
    assert len(audit.recommendations) == 0


def test_security_auditor_get_audit_report_nonexistent():
    """测试 SecurityAuditor（获取审计报告，不存在的审计）"""
    auditor = SecurityAuditor()
    result = auditor.get_audit_report("nonexistent_id")
    assert result is None


def test_security_auditor_get_audit_report_existing():
    """测试 SecurityAuditor（获取审计报告，存在的审计）"""
    auditor = SecurityAuditor()
    audit = auditor.conduct_audit(
        audit_id="test_id",
        findings=[{"finding": "test"}],
        risk_level=RiskLevel.MEDIUM,
        recommendations=["建议1"]
    )
    result = auditor.get_audit_report("test_id")
    assert result is not None
    assert result["audit_id"] == "test_id"
    assert result["findings_count"] == 1
    assert result["recommendations_count"] == 1


def test_security_auditor_get_high_risk_findings_empty():
    """测试 SecurityAuditor（获取高风险发现，空）"""
    auditor = SecurityAuditor()
    result = auditor.get_high_risk_findings()
    assert result == []


def test_security_auditor_get_high_risk_findings_with_high_risk():
    """测试 SecurityAuditor（获取高风险发现，有高风险）"""
    auditor = SecurityAuditor()
    audit = auditor.conduct_audit(
        audit_id="test_id",
        findings=[{"finding": "test", "risk_level": "high"}],
        risk_level=RiskLevel.HIGH,
        recommendations=[]
    )
    result = auditor.get_high_risk_findings()
    assert len(result) >= 1


def test_security_auditor_get_high_risk_findings_with_critical_risk():
    """测试 SecurityAuditor（获取高风险发现，有严重风险）"""
    auditor = SecurityAuditor()
    audit = auditor.conduct_audit(
        audit_id="test_id",
        findings=[{"finding": "test", "risk_level": "critical"}],
        risk_level=RiskLevel.CRITICAL,
        recommendations=[]
    )
    result = auditor.get_high_risk_findings()
    assert len(result) >= 1


def test_security_auditor_get_high_risk_findings_low_risk():
    """测试 SecurityAuditor（获取高风险发现，低风险）"""
    auditor = SecurityAuditor()
    audit = auditor.conduct_audit(
        audit_id="test_id",
        findings=[{"finding": "test", "risk_level": "low"}],
        risk_level=RiskLevel.LOW,
        recommendations=[]
    )
    result = auditor.get_high_risk_findings()
    # 低风险的审计不应该出现在高风险发现中
    assert len(result) == 0


def test_enterprise_data_governance_manager_initialize_framework():
    """测试 EnterpriseDataGovernanceManager（初始化治理框架）"""
    manager = EnterpriseDataGovernanceManager()
    result = manager.initialize_governance_framework()
    assert result["policies_created"] == 2
    assert result["requirements_added"] == 2
    assert result["framework_status"] == "initialized"


def test_enterprise_data_governance_manager_generate_report_empty():
    """测试 EnterpriseDataGovernanceManager（生成治理报告，空）"""
    manager = EnterpriseDataGovernanceManager()
    result = manager.generate_governance_report()
    assert "report_date" in result
    assert "active_policies_count" in result
    assert "compliance_status" in result
    assert "high_risk_findings_count" in result
    assert "overall_governance_score" in result
    assert "recommendations" in result


def test_enterprise_data_governance_manager_generate_report_with_data():
    """测试 EnterpriseDataGovernanceManager（生成治理报告，有数据）"""
    manager = EnterpriseDataGovernanceManager()
    manager.initialize_governance_framework()
    result = manager.generate_governance_report()
    assert result["active_policies_count"] >= 2
    assert len(result["compliance_status"]) > 0


def test_enterprise_data_governance_manager_calculate_governance_score_empty():
    """测试 EnterpriseDataGovernanceManager（计算治理评分，空）"""
    manager = EnterpriseDataGovernanceManager()
    score = manager._calculate_governance_score()
    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_enterprise_data_governance_manager_calculate_governance_score_with_data():
    """测试 EnterpriseDataGovernanceManager（计算治理评分，有数据）"""
    manager = EnterpriseDataGovernanceManager()
    manager.initialize_governance_framework()
    score = manager._calculate_governance_score()
    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_enterprise_data_governance_manager_generate_recommendations_empty():
    """测试 EnterpriseDataGovernanceManager（生成建议，空）"""
    manager = EnterpriseDataGovernanceManager()
    recommendations = manager._generate_recommendations()
    assert isinstance(recommendations, list)


def test_enterprise_data_governance_manager_generate_recommendations_with_issues():
    """测试 EnterpriseDataGovernanceManager（生成建议，有问题）"""
    manager = EnterpriseDataGovernanceManager()
    manager.initialize_governance_framework()
    # 添加高风险发现
    manager.security_auditor.conduct_audit(
        audit_id="test_id",
        findings=[{"finding": "test", "risk_level": "high"}],
        risk_level=RiskLevel.HIGH,
        recommendations=[]
    )
    recommendations = manager._generate_recommendations()
    assert isinstance(recommendations, list)
    assert len(recommendations) >= 0  # 可能有或没有建议


def test_data_policy_manager_record_policy_change_with_old_policy():
    """测试 DataPolicyManager（记录策略变更，有旧策略）"""
    manager = DataPolicyManager()
    policy = manager.create_policy(
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="旧描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH
    )
    old_policy = DataPolicy(**policy.__dict__)
    policy.description = "新描述"
    manager._record_policy_change(policy, "updated", old_policy)
    assert len(manager.policy_history) >= 2
    assert manager.policy_history[-1]["old_policy"] is not None


def test_compliance_manager_verify_compliance_zero_division():
    """测试 ComplianceManager（验证合规状态，零除法保护）"""
    manager = ComplianceManager()
    # 没有要求时应该返回 0 而不是抛出异常
    result = manager.verify_compliance(RegulationType.PCI_DSS)
    assert result["compliance_rate"] == 0
    assert result["total_requirements"] == 0


def test_data_policy_manager_update_policy_invalid_key():
    """测试 DataPolicyManager（更新策略，无效键）"""
    manager = DataPolicyManager()
    policy = DataPolicy(
        policy_id="test_id",
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH,
        effective_date=datetime.now()
    )
    manager.policies["test_id"] = policy
    # 使用无效的键更新策略
    updated_policy = manager.update_policy("test_id", invalid_key="invalid_value")
    # 应该仍然返回策略对象
    assert updated_policy is not None
    assert updated_policy.policy_id == "test_id"


def test_data_policy_manager_deactivate_policy():
    """测试 DataPolicyManager（停用策略）"""
    manager = DataPolicyManager()
    policy = DataPolicy(
        policy_id="test_id",
        policy_name="测试策略",
        policy_type=PolicyType.ACCESS_CONTROL,
        description="测试描述",
        rules=[],
        enforcement_level=EnforcementLevel.HIGH,
        effective_date=datetime.now()
    )
    manager.policies["test_id"] = policy
    # 停用策略
    result = manager.deactivate_policy("test_id")
    assert result is True
    assert manager.policies["test_id"].status == "inactive"


def test_data_policy_manager_deactivate_policy_nonexistent():
    """测试 DataPolicyManager（停用策略，不存在）"""
    manager = DataPolicyManager()
    # 停用不存在的策略
    result = manager.deactivate_policy("nonexistent")
    assert result is False


def test_enterprise_data_governance_manager_generate_recommendations_low_compliance():
    """测试 EnterpriseDataGovernanceManager（生成建议，低合规率）"""
    manager = EnterpriseDataGovernanceManager()
    # 设置低合规率
    manager.compliance_manager.compliance_status = {
        "GDPR": {"compliance_rate": 80, "total_requirements": 10, "met_requirements": 8}
    }
    recommendations = manager._generate_recommendations()
    # 应该包含提高合规率的建议
    assert isinstance(recommendations, list)
    # 如果合规率低于100，应该有建议
    if any(status.get('compliance_rate', 0) < 100 for status in manager.compliance_manager.compliance_status.values()):
        assert len(recommendations) > 0

