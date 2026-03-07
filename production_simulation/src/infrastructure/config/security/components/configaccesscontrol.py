
from .accessrecord import AccessRecord
from .securityconfig import SecurityConfig
from typing import Dict, List
import threading, time, logging

"""安全配置相关类"""


class ConfigAccessControl:
    """配置访问控制"""

    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self._access_records: List[AccessRecord] = []
        self._lock = threading.RLock()
        self._failed_attempts: Dict[str, List[float]] = {}
        self._locked_users: Dict[str, float] = {}

        # 权限规则
        self.permissions: Dict[str, List[str]] = {
            "admin": ["read", "write", "delete", "audit"],
            "operator": ["read", "write"],
            "viewer": ["read"],
            "system": ["read", "write"]  # 系统内部访问
        }

    def check_access(self, user: str, action: str, resource: str = "") -> bool:
        """检查访问权限"""
        with self._lock:
            # 检查用户是否被锁定
            if user in self._locked_users:
                if time.time() - self._locked_users[user] < self.config.lockout_duration:
                    self._record_access(user, action, resource, False)
                    return False
                else:
                    # 锁定期已过，解锁用户
                    del self._locked_users[user]

            # 检查权限
            user_roles = self._get_user_roles(user)
            allowed_actions = set()
            for role in user_roles:
                allowed_actions.update(self.permissions.get(role, []))

            has_permission = action in allowed_actions

            # 记录访问
            self._record_access(user, action, resource, has_permission)

            # 处理失败尝试
            if not has_permission:
                self._handle_failed_attempt(user)

            return has_permission

    def _get_user_roles(self, user: str) -> List[str]:
        """获取用户角色"""
        # 这里可以集成用户管理系统
        # 暂时使用简单的角色映射
        role_map = {
            "admin": ["admin"],
            "operator": ["operator"],
            "viewer": ["viewer"],
            "system": ["system"]
        }
        return role_map.get(user, ["viewer"])

    def _record_access(self, user: str, action: str, resource: str, success: bool):
        """记录访问"""
        record = AccessRecord(
            timestamp=time.time(),
            user=user,
            action=action,
            resource=resource,
            success=success
        )

        self._access_records.append(record)

        # 保持最近的访问记录
        if len(self._access_records) > 1000:
            self._access_records = self._access_records[-500:]

        if self.config.access_logging:
            logger.info(f"配置访问: 用户={user}, 动作={action}, 资源={resource}, 成功={success}")

    def _handle_failed_attempt(self, user: str):
        """处理失败尝试"""
        current_time = time.time()

        if user not in self._failed_attempts:
            self._failed_attempts[user] = []

        # 清理过期的失败尝试（5分钟内）
        self._failed_attempts[user] = [
            t for t in self._failed_attempts[user]
            if current_time - t < 300
        ]

        self._failed_attempts[user].append(current_time)

        # 检查是否达到最大尝试次数
        if len(self._failed_attempts[user]) >= self.config.max_access_attempts:
            self._locked_users[user] = current_time
            logger.warning(f"用户 {user} 因多次失败尝试被锁定")

    def get_access_logs(self, user: str = "", limit: int = 100) -> List[AccessRecord]:
        """获取访问日志"""
        with self._lock:
            logs = self._access_records

            if user:
                logs = [r for r in logs if r.user == user]

            return logs[-limit:]

    def get_access_records(self, user: str = "", limit: int = 100) -> List[AccessRecord]:
        """获取访问记录（get_access_logs的别名）"""
        return self.get_access_logs(user, limit)

    def clear_access_records(self):
        """清除访问记录"""
        with self._lock:
            self._access_records.clear()


logger = logging.getLogger(__name__)




