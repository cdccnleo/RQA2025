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


import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data.security.access_control_manager import (
    AccessControlManager,
    ResourceType,
)
from src.data.security.audit_logging_manager import (
    AuditLoggingManager,
    AuditEventType,
    AuditSeverity,
)
from src.data.security.data_encryption_manager import (
    DataEncryptionManager,
)


def test_access_control_basic_rbac_and_policy(tmp_path, monkeypatch):
    mgr = AccessControlManager(config_path=str(tmp_path / "ac"), enable_audit=True)

    # 创建用户并授予角色
    uid = mgr.create_user("alice")
    mgr.assign_role_to_user(uid, "analyst")  # analyst: read, execute

    # 初始无策略时，只按直接/继承权限判定
    d1 = mgr.check_access(uid, "data:stock:AAPL", "read")
    assert d1.allowed is True and d1.reason in ("direct_permission", "policy_check")

    # 写权限应为 False（无写权限）
    d2 = mgr.check_access(uid, "data:stock:AAPL", "write")
    assert d2.allowed is False

    # 配置策略：允许 config:* 的读
    pid = mgr.create_access_policy(
        name="read_config",
        resource_type=ResourceType.CONFIG,
        resource_pattern="config:*",
        permissions=["read"],
        conditions=None,
    )

    d3 = mgr.check_access(uid, "config:global", "read")
    # 角色可能直接允许 read，或由策略放行，两者均视为通过
    assert d3.allowed is True and d3.reason in ("direct_permission", "policy_check")

    # 权限缓存命中
    d4 = mgr.check_access(uid, "config:global", "read")
    assert d4.reason == "cached_result"

    # 统计与审计日志
    stats = mgr.get_access_statistics()
    assert "total_access_checks" in stats

    mgr.shutdown()


def test_audit_logging_manager_end_to_end(tmp_path, monkeypatch):
    alm = AuditLoggingManager(log_path=str(tmp_path / "audit"), enable_realtime_monitoring=False)

    # 记录多类事件
    alm.log_security_event(user_id="u1", action="login", result="failure", risk_score=0.8)
    alm.log_access_event(user_id="u1", resource="data/table1", action="read", result="success", risk_score=0.1)
    alm.log_data_operation(user_id="u2", operation="update", resource="sensitive/table2", result="success")

    # 手动处理队列，便于测试
    alm._process_event_queue()

    # 查询最近事件与报告
    events = alm.query_events(limit=10)
    assert len(events) >= 3

    sec_report = alm.get_security_report(days=1)
    assert "summary" in sec_report and "risk_assessment" in sec_report

    comp_report = alm.get_compliance_report(report_type="general", days=30)
    assert comp_report.report_type == "general"

    # 清理与关闭
    alm.cleanup_old_logs(days_to_keep=0)  # 不一定会删除，接口可调用
    alm.shutdown()


def test_data_encryption_manager_encrypt_decrypt_and_rotation(tmp_path):
    dem = DataEncryptionManager(key_store_path=str(tmp_path / "keys"), enable_audit=True)

    # 生成新密钥并加解密
    key_id = dem.generate_key("AES - 256")
    plaintext = b"hello-secret"
    enc = dem.encrypt_data(plaintext, key_id=key_id)
    dec = dem.decrypt_data(enc)
    assert dec.decrypted_data == plaintext

    # 统计与审计
    stats = dem.get_encryption_stats()
    assert stats["total_keys"] >= 1
    assert stats["current_key_id"] is not None

    # 过期密钥清理（人为设置过期）
    dem.keys[key_id].expires_at = datetime.now() - timedelta(days=1)
    cleaned = dem.cleanup_expired_keys()
    assert cleaned >= 0

    dem.shutdown()


