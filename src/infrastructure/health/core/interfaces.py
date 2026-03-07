"""
interfaces 模块

提供 interfaces 相关功能和接口。
"""

import logging

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
"""
基础设施层 - 健康检查层 接口定义
"""

logger = logging.getLogger(__name__)


class IHealthComponent(ABC):

    """Health组件接口

    定义Health功能的核心抽象接口。

    # # 功能特性
    - 提供Health功能的标准接口定义
    - 支持扩展和定制化实现
    - 保证功能的一致性和可靠性

    # # 接口定义
    该接口定义了Health组件的基本契约:
    - 核心功能方法定义
    - 错误处理规范
    - 生命周期管理
    - 配置参数要求

    # # 实现要求
    实现类需要满足以下要求:
    1. 实现所有抽象方法
    2. 处理异常情况
    3. 提供必要的配置选项
    4. 保证线程安全（如果适用）

    # # 使用示例
    ```python
    # 创建Health组件实例
    component = ConcreteHealthComponent(config)

    # 使用组件功能
    try:
        result = component.execute_operation()
        print(f"操作结果: {result}")
    except ComponentError as e:
        print(f"组件错误: {e}")
    ```

    # # 注意事项
    - 实现类必须保证异常安全
    - 资源使用需要正确清理
    - 配置参数需要验证
    - 日志记录需要完善

    # # 相关组件
    - 依赖: 基础配置组件
    - 协作: 监控和日志组件
    - 扩展: 具体实现类
    """


class IHealthChecker(ABC):

    """健康检查器接口"""

    @abstractmethod
    def register_service(
        self,
        service_name: str,
        check_function: callable,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """注册服务进行健康检查

        Args:
            service_name: 服务名称
            check_function: 检查函数
            config: 服务配置

        Returns:
            bool: 注册是否成功
        """

    @abstractmethod
    def perform_health_check(self, service_name: str, check_type: str = "basic") -> Dict[str, Any]:
        """执行健康检查

        Args:
            service_name: 服务名称
            check_type: 检查类型

        Returns:
            Dict[str, Any]: 健康检查结果
        """

    @abstractmethod
    def get_service_status(self, service_name: str) -> str:
        """获取服务状态

        Args:
            service_name: 服务名称

        Returns:
            str: 服务状态
        """

    @abstractmethod
    def get_all_service_status(self) -> Dict[str, str]:
        """获取所有服务状态

        Returns:
            Dict[str, str]: 所有服务状态
        """

    @abstractmethod
    def start_monitoring(self) -> bool:
        """启动监控

        Returns:
            bool: 启动是否成功
        """

    @abstractmethod
    def stop_monitoring(self) -> bool:
        """停止监控

        Returns:
            bool: 停止是否成功
        """


class IUnifiedInfrastructureInterface(ABC):
    """统一基础设施接口

    为所有基础设施组件定义统一的接口规范，
    确保架构一致性和可扩展性。
    """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化组件

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """

    @abstractmethod
    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """

    @abstractmethod
    def is_healthy(self) -> bool:
        """检查组件健康状态

        Returns:
            bool: 组件是否健康
        """

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标

        Returns:
            Dict[str, Any]: 组件指标数据
        """

    @abstractmethod
    def cleanup(self) -> bool:
        """清理组件资源

        Returns:
            bool: 清理是否成功
        """


class IAsyncInfrastructureInterface(IUnifiedInfrastructureInterface):
    """异步基础设施接口

    扩展统一接口，添加异步处理能力。
    """

    @abstractmethod
    async def initialize_async(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """异步初始化组件

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """

    @abstractmethod
    async def get_component_info_async(self) -> Dict[str, Any]:
        """异步获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """

    @abstractmethod
    async def is_healthy_async(self) -> bool:
        """异步检查组件健康状态

        Returns:
            bool: 组件是否健康
        """

    @abstractmethod
    async def get_metrics_async(self) -> Dict[str, Any]:
        """异步获取组件指标

        Returns:
            Dict[str, Any]: 组件指标数据
        """

    @abstractmethod
    async def cleanup_async(self) -> bool:
        """异步清理组件资源

        Returns:
            bool: 清理是否成功
        """


class IHealthInfrastructureInterface(IAsyncInfrastructureInterface):
    """健康基础设施接口

    专门为健康检查组件定义的统一接口。
    """

    @abstractmethod
    async def check_health_async(self) -> Dict[str, Any]:
        """异步执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """

    @abstractmethod
    async def check_service_async(self, service_name: str, timeout: float = 5.0) -> Dict[str, Any]:
        """异步检查特定服务健康状态

        Args:
            service_name: 服务名称
            timeout: 超时时间

        Returns:
            Dict[str, Any]: 服务健康检查结果
        """

    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """同步执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """

    @abstractmethod
    def check_service(self, service_name: str, timeout: int = 5) -> Dict[str, Any]:
        """同步检查特定服务健康状态

        Args:
            service_name: 服务名称
            timeout: 超时时间

        Returns:
            Dict[str, Any]: 服务健康检查结果
        """


class IInfrastructureAdapter(ABC):
    """基础设施适配器接口

    定义基础设施服务的适配器标准接口，
    支持不同基础设施服务的统一访问。
    """

    @abstractmethod
    def get_service_name(self) -> str:
        """获取服务名称"""

    @abstractmethod
    def is_service_available(self) -> bool:
        """检查服务是否可用"""

    @abstractmethod
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""

    @abstractmethod
    def execute_operation(self, operation: str, **kwargs) -> Any:
        """执行服务操作"""

    @abstractmethod
    async def execute_operation_async(self, operation: str, **kwargs) -> Any:
        """异步执行服务操作"""


class InfrastructureAdapterFactory:
    """基础设施适配器工厂

    统一管理各类基础设施适配器的创建和注册。
    """

    _adapters: Dict[str, Type[IInfrastructureAdapter]] = {}
    
    def __init__(self):
        """初始化工厂实例"""
        self._adapters: Dict[str, Any] = {}

    @classmethod
    def register_adapter(cls, service_type: str, adapter_class: Type[IInfrastructureAdapter]):
        """注册适配器"""
        cls._adapters[service_type] = adapter_class
        logger.info(f"注册基础设施适配器: {service_type} -> {adapter_class.__name__}")

    @classmethod
    def create_adapter(cls, service_type: str, **kwargs) -> IInfrastructureAdapter:
        """创建适配器实例"""
        if service_type not in cls._adapters:
            raise ValueError(f"未找到适配器类型: {service_type}")

        adapter_class = cls._adapters[service_type]
        try:
            adapter = adapter_class(**kwargs)
            logger.info(f"创建基础设施适配器成功: {service_type}")
            return adapter
        except Exception as e:
            logger.error(f"创建基础设施适配器失败: {service_type}, 错误: {str(e)}")
            raise

    @classmethod
    def get_available_adapters(cls) -> List[str]:
        """获取所有可用适配器类型"""
        return list(cls._adapters.keys())

    @classmethod
    def has_adapter(cls, service_type: str) -> bool:
        """检查是否存在指定类型的适配器"""
        return service_type in cls._adapters

    @classmethod
    def get_adapter(cls, service_type: str) -> Type[IInfrastructureAdapter]:
        """获取适配器类"""
        if service_type not in cls._adapters:
            raise ValueError(f"未找到适配器类型: {service_type}")

        return cls._adapters[service_type]
    
    def register_service(self, service_name: str, adapter: Any):
        """注册服务适配器实例"""
        self._adapters[service_name] = adapter
        logger.info(f"注册服务适配器实例: {service_name}")
    
    def get_adapter(self, service_name: str) -> Any:
        """获取服务适配器实例"""
        if service_name not in self._adapters:
            raise ValueError(f"No adapter registered for service: {service_name}")
        return self._adapters[service_name]
    
    def get_available_services(self) -> List[str]:
        """获取可用服务列表"""
        return list(self._adapters.keys())
    
    def get_factory_info(self) -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "total_adapters": len(self._adapters),
            "registered_services": list(self._adapters.keys()),
            "factory_type": self.__class__.__name__
        }

# 扩展接口可以在这里添加
