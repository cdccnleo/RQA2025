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

from src.data.security.access_control_manager import (
    AccessControlManager,
    Permission,
    ResourceType,
)


@pytest.fixture
def access_manager(tmp_path):
    config_dir = tmp_path / "access_control_config"
    return AccessControlManager(config_path=str(config_dir), enable_audit=False)


def test_check_access_uses_direct_permission_and_cache(access_manager):
    user_id = access_manager.create_user("alice")

    initial_decision = access_manager.check_access(
        user_id=user_id,
        resource="data:alpha:item",
        permission=Permission.DELETE.value,
    )
    assert not initial_decision.allowed
    assert initial_decision.reason == "no_matching_policy"

    access_manager.grant_permission_to_user(user_id, Permission.DELETE.value)

    decision_after_grant = access_manager.check_access(
        user_id=user_id,
        resource="data:alpha:item",
        permission=Permission.DELETE.value,
    )
    assert decision_after_grant.allowed
    assert decision_after_grant.reason == "direct_permission"

    cached_decision = access_manager.check_access(
        user_id=user_id,
        resource="data:alpha:item",
        permission=Permission.DELETE.value,
    )
    assert cached_decision.allowed
    assert cached_decision.reason == "cached_result"


def test_policy_conditions_and_cache_invalidation(access_manager):
    user_id = access_manager.create_user("bob")

    policy_id = access_manager.create_access_policy(
        name="reports_reader",
        resource_type=ResourceType.DATA,
        resource_pattern="data:reports:*",
        permissions=[Permission.READ.value],
        conditions={"ip_range": ["10.0.0.1", "10.0.0.2"]},
    )

    allowed_decision = access_manager.check_access(
        user_id=user_id,
        resource="data:reports:q1",
        permission=Permission.READ.value,
        context={"ip_range": "10.0.0.1"},
    )
    assert allowed_decision.allowed
    assert allowed_decision.reason == "policy_check"
    assert allowed_decision.applied_policies == [policy_id]

    access_manager.clear_permission_cache()
    denied_decision = access_manager.check_access(
        user_id=user_id,
        resource="data:reports:q1",
        permission=Permission.READ.value,
        context={"ip_range": "192.168.0.1"},
    )
    assert not denied_decision.allowed
    assert denied_decision.reason == "no_matching_policy"

