from typing import Dict, Any, Optional
import uuid
from datetime import datetime


class DataPolicyManager:

    """合规策略管理"""

    def __init__(self):

        self.policies: Dict[str, Dict[str, Any]] = {}

    def _validate_policy(self, policy: Dict[str, Any]) -> bool:
        """验证策略格式"""
        if not policy or not isinstance(policy, dict):
            return False

        # 检查必需字段
        if "name" not in policy:
            return False

        if "required_fields" not in policy:
            return False

        # 验证ID格式（如果提供）
        if "id" in policy:
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', policy["id"]):
                return False

        # 验证执行级别
        if "enforcement_level" in policy:
            valid_levels = ["strict", "moderate", "lenient"]
            if policy["enforcement_level"] not in valid_levels:
                return False

        # 验证隐私级别
        if "privacy_level" in policy:
            valid_levels = ["standard", "encrypted", "none"]
            if policy["privacy_level"] not in valid_levels:
                return False

        return True

    def _add_timestamps(self, policy: Dict[str, Any]) -> None:
        """添加时间戳"""
        now = datetime.now().isoformat()
        if "created_at" not in policy:
            policy["created_at"] = now
        policy["updated_at"] = now

    def register_policy(self, policy: Optional[Dict[str, Any]]) -> bool:
        """注册策略"""
        if not self._validate_policy(policy):
            return False

        policy_id = policy.get("id") or str(uuid.uuid4())

        # 检查ID是否已存在
        if policy_id in self.policies:
            return False

        policy["id"] = policy_id
        self._add_timestamps(policy)
        self.policies[policy_id] = policy
        return True

    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """获取策略"""
        return self.policies.get(policy_id)

    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """更新策略"""
        if policy_id not in self.policies:
            return False

        # 验证更新内容
        if not isinstance(updates, dict):
            return False

        # 更新策略
        self.policies[policy_id].update(updates)
        self._add_timestamps(self.policies[policy_id])
        return True

    def delete_policy(self, policy_id: str) -> bool:
        """删除策略"""
        if policy_id not in self.policies:
            return False

        del self.policies[policy_id]
        return True

    def list_policies(self) -> Dict[str, Dict[str, Any]]:
        """列出所有策略"""
        return self.policies.copy()
