"""安全相关异常定义"""

from typing import Optional

class SensitiveDataAccessDenied(Exception):
    """当尝试访问敏感数据但权限不足时抛出"""

    def __init__(self, resource: str, required_role: str, current_role: str):
        """
        初始化异常

        Args:
            resource: 尝试访问的资源名称
            required_role: 需要的角色
            current_role: 当前用户角色
        """
        self.resource = resource
        self.required_role = required_role
        self.current_role = current_role
        message = (f"Access denied to sensitive resource '{resource}'. "
                  f"Required role: {required_role}, Current role: {current_role}")
        super().__init__(message)

class SecurityViolationError(Exception):
    """安全违规异常"""
    
    def __init__(self, violation_type: str, details: str, user_id: Optional[str] = None):
        """
        初始化安全违规异常
        
        Args:
            violation_type: 违规类型
            details: 违规详情
            user_id: 用户ID
        """
        self.violation_type = violation_type
        self.details = details
        self.user_id = user_id
        message = f"Security violation: {violation_type} - {details}"
        if user_id:
            message += f" (User: {user_id})"
        super().__init__(message)
