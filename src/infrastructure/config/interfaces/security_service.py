from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ISecurityValidator(ABC):
    """安全验证器接口
    
    功能:
    1. 配置数据验证
    2. 访问控制检查
    3. 敏感数据过滤
    """

    @abstractmethod
    def validate(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """验证配置数据
        
        Args:
            config: 待验证配置
            schema: 验证规则
            
        Returns:
            bool: 是否验证通过
        """
        pass

    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """获取验证错误详情
        
        Returns:
            List[str]: 错误消息列表
        """
        pass


class ISecurityService(ABC):
    """配置安全服务接口

    功能:
    1. 配置签名验证
    2. 敏感操作检查
    3. 双因素认证集成
    """

    @abstractmethod
    def verify_signature(self, config: Dict[str, Any], signature: str) -> bool:
        """验证配置签名

        Args:
            config: 配置字典
            signature: 预期签名

        Returns:
            bool: 签名是否有效
        """
        pass

    @abstractmethod
    def is_sensitive_operation(self, key: str) -> bool:
        """检查是否为敏感操作

        Args:
            key: 配置键

        Returns:
            bool: 是否为敏感操作
        """
        pass

    @abstractmethod
    def require_2fa(self, key: str) -> bool:
        """检查操作是否需要双因素认证

        Args:
            key: 配置键

        Returns:
            bool: 是否需要2FA
        """
        pass
