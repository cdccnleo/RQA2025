

"""安全配置相关类"""


class AccessRecord:
    """访问记录"""
    
    def __init__(self, timestamp: float, user: str, action: str, resource: str, 
                 success: bool, ip_address: str = "", user_agent: str = ""):
        self.timestamp = timestamp
        self.user = user
        self.action = action
        self.resource = resource
        self.success = success
        self.ip_address = ip_address
        self.user_agent = user_agent




