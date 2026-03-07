
from typing import Any

"""安全配置相关类"""


class ConfigAuditLog:
    """配置审计日志"""
    
    def __init__(self, timestamp: float, action: str, key: str, 
                 old_value: Any = None, new_value: Any = None, 
                 user: str = "system", reason: str = ""):
        self.timestamp = timestamp
        self.action = action
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.user = user
        self.reason = reason




