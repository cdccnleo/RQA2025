import threading
from datetime import datetime
from typing import Any, Dict, Optional

class AuditLogger:
    """线程安全的审计日志记录器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式确保全局唯一实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        """初始化日志器"""
        self._lock = threading.Lock()
        self._handlers = []

    def add_handler(self, handler):
        """添加日志处理器"""
        with self._lock:
            self._handlers.append(handler)

    def log(self,
            action: str,
            user: Optional[str] = None,
            resource: Optional[str] = None,
            status: str = "SUCCESS",
            details: Optional[Dict[str, Any]] = None):
        """
        记录审计日志

        :param action: 执行的操作
        :param user: 执行操作的用户
        :param resource: 操作的资源
        :param status: 操作状态 (SUCCESS/FAILED)
        :param details: 附加详细信息
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user or "system",
            "resource": resource,
            "status": status,
            "details": details or {}
        }

        with self._lock:
            for handler in self._handlers:
                handler(log_entry)

# 全局审计日志实例
audit_log = AuditLogger()
