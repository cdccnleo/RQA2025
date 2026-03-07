
import hashlib

from typing import Dict, Any, List
from collections import defaultdict
import json
"""
同步冲突管理器
处理分布式配置同步中的冲突检测和解决
"""


class SyncConflictManager:

    """同步冲突管理器"""

    def __init__(self):

        self._conflicts: List[Dict[str, Any]] = []

    def calculate_config_checksum(self, config: Dict[str, Any]) -> str:
        """计算配置校验和"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def detect_conflicts(self, local_config: Dict[str, Any], remote_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测配置冲突"""
        conflicts = []

        # 获取所有键
        all_keys = set(local_config.keys()) | set(remote_config.keys())

        for key in all_keys:
            local_value = local_config.get(key)
            remote_value = remote_config.get(key)

            if local_value != remote_value:
                conflicts.append({
                    "key": key,
                    "local_value": local_value,
                    "remote_value": remote_value,
                    "conflict_type": "value_mismatch"
                })

        return conflicts

    def resolve_conflicts(self, conflicts: List[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """解决冲突"""
        resolved_config = {}

        for conflict in conflicts:
            key = conflict["key"]
            local_value = conflict["local_value"]
            remote_value = conflict["remote_value"]

            if strategy == "merge":
                # 简单合并策略：优先使用本地值
                resolved_config[key] = local_value if local_value is not None else remote_value
            elif strategy == "overwrite":
                # 覆盖策略：使用远程值
                resolved_config[key] = remote_value
            elif strategy == "ask":
                # 询问策略：这里使用本地值作为默认
                resolved_config[key] = local_value if local_value is not None else remote_value
            else:
                # 默认使用本地值
                resolved_config[key] = local_value if local_value is not None else remote_value

        return resolved_config

    def get_conflicts(self) -> List[Dict[str, Any]]:
        """获取所有冲突"""
        return self._conflicts.copy()

    def clear_conflicts(self) -> int:
        """清空所有冲突"""
        count = len(self._conflicts)
        self._conflicts.clear()
        return count

    def get_conflict_count(self) -> int:
        """获取冲突数量"""
        return len(self._conflicts)

    def get_conflict_summary(self) -> Dict[str, Any]:
        """获取冲突统计"""
        if not self._conflicts:
            return {"total": 0, "by_type": {}}

        type_counts = defaultdict(int)
        for conflict in self._conflicts:
            if conflict is None or not isinstance(conflict, dict):
                conflict_type = "unknown"
            else:
                conflict_type = conflict.get("conflict_type", "unknown")
            type_counts[conflict_type] += 1

        return {
            "total": len(self._conflicts),
            "by_type": dict(type_counts)
        }




