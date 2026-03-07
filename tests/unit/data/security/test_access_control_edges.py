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


import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.data.security.access_control_manager import (
    AccessControlManager,
    ResourceType,
)


@pytest.fixture
def tmp_acm(tmp_path: Path):
    acm = AccessControlManager(config_path=str(tmp_path), enable_audit=True)
    return acm


def test_create_user_duplicate_username_raises(tmp_acm: AccessControlManager):
    uid1 = tmp_acm.create_user("alice")
    assert uid1
    with pytest.raises(ValueError):
        tmp_acm.create_user("alice")


def test_assign_revoke_role_invalid_inputs_raise(tmp_acm: AccessControlManager):
    # invalid user
    with pytest.raises(ValueError):
        tmp_acm.assign_role_to_user("no_user", "admin")
    # create user, invalid role
    uid = tmp_acm.create_user("bob")
    with pytest.raises(ValueError):
        tmp_acm.assign_role_to_user(uid, "no_role")

    # revoke non-existing user
    with pytest.raises(ValueError):
        tmp_acm.revoke_role_from_user("no_user", "admin")


def test_set_role_inheritance_edges(tmp_acm: AccessControlManager):
    # create custom role
    rid = tmp_acm.create_role("viewer", permissions=["read"])
    # parent not exist
    with pytest.raises(ValueError):
        tmp_acm.set_role_inheritance(rid, "no_parent")
    # child not exist
    with pytest.raises(ValueError):
        tmp_acm.set_role_inheritance("no_child", rid)
    # self inheritance
    with pytest.raises(ValueError):
        tmp_acm.set_role_inheritance(rid, rid)


def test_check_access_user_not_found_and_inactive(tmp_acm: AccessControlManager):
    # user not found
    d1 = tmp_acm.check_access("u-none", "data:stock:000001", "read")
    assert d1.allowed is False and d1.reason == "user_not_found"

    # inactive user
    uid = tmp_acm.create_user("charlie")
    tmp_acm.users[uid].is_active = False
    d2 = tmp_acm.check_access(uid, "data:stock:000001", "read")
    assert d2.allowed is False and d2.reason == "user_inactive"


def test_policy_time_range_and_cache_invalidation(tmp_acm: AccessControlManager):
    uid = tmp_acm.create_user("dora")
    # no direct permission, rely on policy
    now = datetime.now()
    pid = tmp_acm.create_access_policy(
        name="stock-read-office-hours",
        resource_type=ResourceType.DATA,
        resource_pattern="data:stock:*",
        permissions=["read"],
        conditions={
            "time_range": {
                "start": (now - timedelta(minutes=1)),
                "end": (now + timedelta(minutes=1)),
            }
        },
    )
    # convert stored strings back to datetime in check (manager expects datetime in context)
    ctx_ok = {"time_range": now}
    d_ok = tmp_acm.check_access(uid, "data:stock:600000", "read", context=ctx_ok)
    assert d_ok.allowed is True and d_ok.reason in ("policy_check", "cached_result")

    # outside time
    ctx_bad = {"time_range": now + timedelta(hours=2)}
    d_bad = tmp_acm.check_access(uid, "data:stock:600000", "read", context=ctx_bad)
    # first call cached allow; ensure cache cleared influences decision
    tmp_acm.clear_permission_cache()
    d_bad2 = tmp_acm.check_access(uid, "data:stock:600000", "read", context=ctx_bad)
    assert d_bad2.allowed is False and d_bad2.reason == "no_matching_policy"

    # deactivate policy and verify cache invalidation by role change that clears cache
    tmp_acm.update_access_policy(pid, {"is_active": False})
    tmp_acm._clear_role_cache("admin")  # force clear all caches
    d_after = tmp_acm.check_access(uid, "data:stock:600000", "read", context=ctx_ok)
    assert d_after.allowed is False


def test_audit_log_write_and_read_filters(tmp_acm: AccessControlManager, tmp_path: Path):
    uid = tmp_acm.create_user("eve")
    # trigger some audit events
    tmp_acm.assign_role_to_user(uid, "analyst")
    tmp_acm.check_access(uid, "data:stock:000001", "read")
    tmp_acm.check_access(uid, "data:stock:000001", "write")

    # read logs
    logs_all = tmp_acm.get_audit_logs(limit=100)
    assert isinstance(logs_all, list)
    assert any(l.get("operation") == "assign_role" for l in logs_all)
    # filter by operation
    access_logs = tmp_acm.get_audit_logs(operation="access_check", limit=100)
    assert all(l.get("operation") == "access_check" for l in access_logs)
    # filter by user
    user_logs = tmp_acm.get_audit_logs(user_id=uid, limit=100)
    assert all(l.get("details", {}).get("user_id") == uid for l in user_logs if l.get("details"))


def test_audit_log_io_error_is_handled(tmp_acm: AccessControlManager, monkeypatch: pytest.MonkeyPatch):
    # force open to raise error
    def boom(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("builtins.open", boom)
    # should not raise
    tmp_acm._audit_log("test_op", {"x": 1})


def test_access_statistics_structure(tmp_acm: AccessControlManager):
    uid = tmp_acm.create_user("frank")
    # some checks
    tmp_acm.check_access(uid, "data:stock:000001", "read")
    tmp_acm.check_access(uid, "data:stock:000001", "read")
    stats = tmp_acm.get_access_statistics()
    assert "total_access_checks" in stats
    assert "allow_rate" in stats
    assert "resource_statistics" in stats


