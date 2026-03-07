from datetime import datetime
import json
from pathlib import Path

import pytest

from src.infrastructure.security.access.components.policy_manager import PolicyManager
from src.infrastructure.security.access.components.access_checker import AccessRequest, AccessDecision
from src.infrastructure.security.core.types import UserRole


@pytest.fixture()
def policy_config_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_policy_lifecycle_persists_config(policy_config_dir: Path) -> None:
    manager = PolicyManager(config_path=policy_config_dir)

    policy_id = manager.create_policy(
        name="Office Hours",
        resource_pattern="/api/*",
        permissions={"data:read"},
        roles={UserRole.ADMIN},
        description="office access",
        conditions={"time_range": "09:00-17:00"},
    )

    policy_file = policy_config_dir / "policies.json"
    assert policy_file.exists()
    saved = json.loads(policy_file.read_text(encoding="utf-8"))
    assert policy_id in saved

    assert manager.update_policy(policy_id, {"description": "updated"})
    assert manager.get_policy(policy_id).description == "updated"

    assert manager.delete_policy(policy_id) is True
    assert policy_id not in manager.policies


def test_load_policies_from_existing_file(policy_config_dir: Path) -> None:
    policy_file = policy_config_dir / "policies.json"
    policy_file.parent.mkdir(parents=True, exist_ok=True)
    policy_file.write_text(
        json.dumps(
            {
                "policy_existing": {
                    "policy_id": "policy_existing",
                    "name": "Existing",
                    "description": "loaded from disk",
                    "resource_pattern": "/documents/*",
                    "permissions": ["data:write"],
                    "roles": [UserRole.ADMIN.value],
                    "conditions": {},
                }
            }
        ),
        encoding="utf-8",
    )

    manager = PolicyManager(config_path=policy_config_dir)
    loaded = manager.get_policy("policy_existing")
    assert loaded is not None
    assert loaded.name == "Existing"
    assert loaded.permission_values() == {"data:write"}


def test_evaluate_policies_with_conditions(policy_config_dir: Path) -> None:
    manager = PolicyManager(config_path=policy_config_dir)
    manager.create_policy(
        name="Document Read",
        resource_pattern="/documents/*",
        permissions={"data:read", "read"},
        roles={UserRole.ANALYST},
        conditions={"time_range": "08:00-20:00"},
    )

    request = AccessRequest(
        user_id="user-1",
        resource="/documents/annual",
        permission="read",
        context={"current_time": datetime(2025, 1, 1, 9, 30)},
    )

    decision = manager.evaluate_policies(request, {"data:read", "read"})
    assert decision == AccessDecision.ALLOW

    late_request = AccessRequest(
        user_id="user-1",
        resource="/documents/annual",
        permission="read",
        context={"current_time": datetime(2025, 1, 1, 23, 0)},
    )
    assert manager.evaluate_policies(late_request, {"data:read", "read"}) == AccessDecision.DENY

    other_request = AccessRequest(
        user_id="user-1",
        resource="/financial/annual",
        permission="write",
        context={},
    )
    assert manager.evaluate_policies(other_request, {"data:read", "read"}) == AccessDecision.ABSTAIN


def test_get_policies_for_resource_filters(policy_config_dir: Path) -> None:
    manager = PolicyManager(config_path=policy_config_dir)
    broad_id = manager.create_policy(
        name="Broad",
        resource_pattern="/data/*",
        permissions={"data:read"},
        roles={UserRole.TRADER},
    )
    specific_id = manager.create_policy(
        name="Specific",
        resource_pattern="/data/reports/*",
        permissions={"data:export"},
        roles={UserRole.TRADER},
    )

    matched = {policy.policy_id for policy in manager.get_policies_for_resource("/data/reports/monthly")}
    assert matched == {broad_id, specific_id}

    none_matched = manager.get_policies_for_resource("/analytics")
    assert none_matched == []


def test_permission_matching_variants(policy_config_dir: Path) -> None:
    manager = PolicyManager(config_path=policy_config_dir)
    request = AccessRequest(
        user_id="user-42",
        resource="/reports/monthly/summary",
        permission="read",
        context={},
    )

    assert manager._permission_matches("*", request)
    assert manager._permission_matches("read", request)
    assert manager._permission_matches("/reports:read", request)
    assert manager._permission_matches("read:/reports", request)
    assert manager._permission_matches("/reports/monthly:read", request)
    assert manager._permission_matches("read:/reports/monthly", request)
    assert manager._permission_matches("/reports/monthly:*", request)
    assert manager._permission_matches("write", request) is False


def test_condition_evaluation_with_composite_rules(policy_config_dir: Path) -> None:
    manager = PolicyManager(config_path=policy_config_dir)
    policy_id = manager.create_policy(
        name="Composite Condition",
        resource_pattern="/secure/*",
        permissions={"access"},
        roles={UserRole.ADMIN},
        conditions={
            "department": {"operator": "in", "values": ["risk", "security"]},
            "clearance": {"operator": "gt", "values": [3]},
        },
    )

    request = AccessRequest(
        user_id="user-admin",
        resource="/secure/dashboard",
        permission="access",
        context={"department": "risk", "clearance": 4},
    )
    assert manager.evaluate_policies(request, {"access"}) == AccessDecision.ALLOW

    denied_request = AccessRequest(
        user_id="user-low",
        resource="/secure/dashboard",
        permission="access",
        context={"department": "marketing", "clearance": 4},
    )
    assert manager.evaluate_policies(denied_request, {"access"}) == AccessDecision.DENY


def test_time_range_cross_midnight_and_invalid_config(policy_config_dir: Path) -> None:
    manager = PolicyManager(config_path=policy_config_dir)
    manager.create_policy(
        name="Night Shift",
        resource_pattern="/ops/*",
        permissions={"execute"},
        roles={UserRole.ANALYST},
        conditions={"time_range": "22:00-06:00"},
    )

    late_request = AccessRequest(
        user_id="ops-1",
        resource="/ops/tasks",
        permission="execute",
        context={"current_time": datetime(2025, 1, 1, 23, 30)},
    )
    assert manager.evaluate_policies(late_request, {"execute"}) == AccessDecision.ALLOW

    early_request = AccessRequest(
        user_id="ops-2",
        resource="/ops/tasks",
        permission="execute",
        context={"current_time": datetime(2025, 1, 1, 7, 0)},
    )
    assert manager.evaluate_policies(early_request, {"execute"}) == AccessDecision.DENY

    # 非法配置应放行条件判断，交由其他逻辑处理
    manager.create_policy(
        name="Invalid Range",
        resource_pattern="/ops/*",
        permissions={"monitor"},
        roles={UserRole.ANALYST},
        conditions={"time_range": "invalid-range"},
    )
    invalid_request = AccessRequest(
        user_id="ops-3",
        resource="/ops/status",
        permission="monitor",
        context={"current_time": datetime(2025, 1, 1, 7, 0)},
    )
    assert manager.evaluate_policies(invalid_request, {"monitor"}) == AccessDecision.ALLOW
