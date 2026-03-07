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


from datetime import datetime
from src.data.governance.enterprise_governance import (
    EnterpriseDataGovernanceManager,
    RegulationType,
    AuditType,
    RiskLevel,
)


def test_governance_report_recommendations_and_score():
    g = EnterpriseDataGovernanceManager()
    init = g.initialize_governance_framework()
    assert init["framework_status"] == "initialized"

    # 实施部分合规要求，保留部分为 pending 以触发建议生成分支
    for req_id in list(g.compliance_manager.requirements.keys())[:1]:
        assert g.compliance_manager.implement_requirement(req_id, {"by": "unit-test"})

    # 增加高风险审计以触发治理评分与建议的分支
    audit_id = g.security_auditor.schedule_audit(audit_type=AuditType.ACCESS, auditor="qa", scheduled_date=datetime.now())
    g.security_auditor.conduct_audit(
        audit_id=audit_id,
        findings=[{"risk_level": "high", "desc": "x"}],
        risk_level=RiskLevel.HIGH,
        recommendations=["fix-high"],
    )

    # 预置合规评分输入，避免依赖内部异步流程
    g.compliance_manager.compliance_status = {
        "gdpr": {"compliance_rate": 50},
        "sox": {"compliance_rate": 0},
    }

    report = g.generate_governance_report()
    assert "overall_governance_score" in report
    assert "recommendations" in report
    # 存在未实现的合规项与高风险审计，应有改进建议
    assert any("合规率" in r or "高风险" in r for r in report.get("recommendations", []))


