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


import re
from datetime import datetime
from time import sleep

import pytest

from src.data.compliance.data_policy_manager import DataPolicyManager


@pytest.fixture()
def manager():
    return DataPolicyManager()


def test_register_policy_generates_id_and_timestamps(manager):
    policy = {
        "name": "basic policy",
        "required_fields": ["id", "email"],
        "enforcement_level": "moderate",
    }

    assert manager.register_policy(policy) is True
    assert "id" in policy
    stored = manager.get_policy(policy["id"])
    assert stored["name"] == "basic policy"
    assert re.match(r"\d{4}-\d{2}-\d{2}T", stored["created_at"])
    assert stored["created_at"] == stored["updated_at"]


def test_register_policy_rejects_invalid_enforcement_level(manager):
    policy = {
        "name": "invalid policy",
        "required_fields": ["id"],
        "enforcement_level": "aggressive",
    }

    assert manager.register_policy(policy) is False
    assert manager.list_policies() == {}


def test_register_policy_rejects_duplicate_id(manager):
    policy1 = {
        "id": "user_policy",
        "name": "first policy",
        "required_fields": ["user_id"],
    }
    policy2 = {
        "id": "user_policy",
        "name": "second policy",
        "required_fields": ["user_id"],
    }

    assert manager.register_policy(policy1) is True
    assert manager.register_policy(policy2) is False
    assert manager.get_policy("user_policy")["name"] == "first policy"


def test_update_policy_merges_fields_and_timestamp(manager):
    policy = {
        "id": "trade_policy",
        "name": "trade policy",
        "required_fields": ["symbol"],
    }
    manager.register_policy(policy)
    original_updated = manager.get_policy("trade_policy")["updated_at"]

    sleep(0.001)  # 保证时间戳变更
    assert manager.update_policy("trade_policy", {"privacy_level": "encrypted"}) is True
    updated_policy = manager.get_policy("trade_policy")
    assert updated_policy["privacy_level"] == "encrypted"
    assert updated_policy["updated_at"] != original_updated


def test_update_policy_rejects_unknown_or_invalid_updates(manager):
    assert manager.update_policy("missing", {"name": "new"}) is False
    policy = {
        "id": "p1",
        "name": "policy",
        "required_fields": ["f1"],
    }
    manager.register_policy(policy)
    assert manager.update_policy("p1", None) is False


def test_delete_and_list_policies(manager):
    policy = {
        "id": "p_delete",
        "name": "delete me",
        "required_fields": ["id"],
    }
    manager.register_policy(policy)
    assert "p_delete" in manager.list_policies()

    assert manager.delete_policy("p_delete") is True
    assert manager.get_policy("p_delete") is None
    assert manager.delete_policy("missing") is False

