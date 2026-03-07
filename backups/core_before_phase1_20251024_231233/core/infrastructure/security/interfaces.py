"""基础设施层 - 安全管理层 接口定义"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ISecurityComponent(ABC):

    """Security组件接口"""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件

        Args:
            config: 组件配置

        Returns:
            初始化是否成功
        """

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态

        Returns:
            组件状态信息
        """

    @abstractmethod
    def shutdown(self) -> None:
        """关闭组件"""


class IAuthManager(ABC):

    """认证管理器接口"""

    @abstractmethod
    def authenticate_user(self, username: str, password: str) -> bool:
        """用户认证"""

    @abstractmethod
    def create_session(self, user_id: str) -> str:
        """创建会话"""

    @abstractmethod
    def validate_session(self, session_id: str) -> bool:
        """验证会话"""


class IEncryptor(ABC):

    """加密器接口"""

    @abstractmethod
    def encrypt(self, data: str) -> str:
        """加密数据"""

    @abstractmethod
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""


class IAuditor(ABC):

    """审计器接口"""

    @abstractmethod
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """记录审计事件"""

    @abstractmethod
    def get_logs(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """获取审计日志"""
