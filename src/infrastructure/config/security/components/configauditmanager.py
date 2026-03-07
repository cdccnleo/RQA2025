
from .configauditlog import ConfigAuditLog
from typing import Any, List, Callable
import threading
import logging
import time

"""安全配置相关类"""


class ConfigAuditManager:
    """配置审计管理器"""

    def __init__(self):
        self.audit_logs: List[ConfigAuditLog] = []
        self._lock = threading.RLock()
        self._callbacks: List[Callable] = []

    def log_change(self, action: str, key: str, old_value: Any = None,
                   new_value: Any = None, user: str = "system", reason: str = ""):
        """记录配置变更"""
        audit_log = ConfigAuditLog(
            timestamp=time.time(),
            action=action,
            key=key,
            old_value=old_value,
            new_value=new_value,
            user=user,
            reason=reason
        )

        with self._lock:
            self.audit_logs.append(audit_log)

            # 保持最近的审计日志
            if len(self.audit_logs) > 5000:
                self.audit_logs = self.audit_logs[-2500:]

        logger.info(f"配置审计: {action} {key} by {user}")

        # 触发回调
        for callback in self._callbacks:
            try:
                callback(audit_log)
            except Exception as e:
                logger.error(f"审计回调执行失败: {e}")

    def add_callback(self, callback: Callable):
        """添加审计回调"""
        with self._lock:
            self._callbacks.append(callback)

    def get_audit_logs(self, key: str = "", user: str = "",
                       limit: int = 100) -> List[ConfigAuditLog]:
        """获取审计日志"""
        with self._lock:
            logs = self.audit_logs

            if key:
                logs = [l for l in logs if l.key == key]
            if user:
                logs = [l for l in logs if l.user == user]

            return logs[-limit:]

    def get_change_history(self, key: str) -> List[ConfigAuditLog]:
        """获取配置项的变更历史"""
        return [l for l in self.audit_logs if l.key == key]


logger = logging.getLogger(__name__)




