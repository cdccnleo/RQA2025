

"""安全配置相关类"""


class SecurityConfig:
    """安全配置"""
    
    def __init__(self, encryption_enabled: bool = True, key_rotation_days: int = 30,
                 access_logging: bool = True, audit_enabled: bool = True, 
                 max_access_attempts: int = 5, lockout_duration: int = 300):
        self.encryption_enabled = encryption_enabled
        self.key_rotation_days = key_rotation_days
        self.access_logging = access_logging
        self.audit_enabled = audit_enabled
        self.max_access_attempts = max_access_attempts
        self.lockout_duration = lockout_duration




