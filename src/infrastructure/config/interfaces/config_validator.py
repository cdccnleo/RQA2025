from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class IConfigValidator(ABC):
    """配置验证器抽象接口，定义统一验证规范"""

    @abstractmethod
    def validate(self, config: Dict, schema: Optional[Dict] = None) -> bool:
        """验证配置数据的有效性和完整性

        Args:
            config: 待验证的配置字典
            schema: 可选的验证模式定义

        Returns:
            验证通过返回True，否则返回False

        Raises:
            ConfigValidationError: 当验证失败时抛出
        """
        pass

    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """获取详细的验证错误信息

        Returns:
            错误信息列表，格式为["field: error_message"]
        """
        pass
