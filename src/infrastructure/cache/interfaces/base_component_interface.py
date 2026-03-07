
from typing import Protocol, Dict, Any
"""
基础组件协议

定义所有组件的通用接口方法，避免重复定义。
使用Protocol模式实现结构化子类型，提高代码复用性。
"""


class IBaseComponent(Protocol):
    """
    基础组件协议

    定义所有组件必须实现的通用方法。
    使用Protocol模式，支持结构化子类型，无需显式继承。

    实现要求：
    - 必须提供component_name属性
    - 必须提供component_type属性
    - 必须实现initialize_component方法
    - 必须实现get_component_status方法
    - 必须实现shutdown_component方法
    - 必须实现health_check方法
    """

    @property
    def component_name(self) -> str:
        """
        组件名称标识符

        Returns:
            str: 组件的唯一名称，用于日志和监控标识
        """
        ...

    @property
    def component_type(self) -> str:
        """
        组件类型标识符

        Returns:
            str: 组件类型，如 'cache', 'config', 'monitoring' 等
        """
        ...

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """
        初始化组件

        Args:
            config: 组件配置字典

        Returns:
            bool: 初始化是否成功
        """
        ...

    def get_component_status(self) -> Dict[str, Any]:
        """
        获取组件状态信息

        Returns:
            Dict[str, Any]: 包含组件状态的字典，必须包含：
                - 'status': 'healthy'/'warning'/'error'
                - 'initialized': bool
                - 'last_check': datetime
                - 'error_count': int
        """
        ...

    def shutdown_component(self) -> None:
        """
        关闭组件

        执行必要的清理工作，确保资源正确释放。
        """
        ...

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 组件是否健康
        """
        ...
