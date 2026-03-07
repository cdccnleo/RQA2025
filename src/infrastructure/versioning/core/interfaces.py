"""
版本管理核心接口

定义版本管理系统的统一接口和抽象基类。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Protocol
from .version import Version


class VersionProvider(Protocol):
    """
    版本提供者协议

    定义版本对象的创建和解析接口
    """

    def parse_version(self, version_str: str) -> Version:
        """
        从字符串解析版本

        Args:
            version_str: 版本字符串

        Returns:
            版本对象
        """
        ...

    def create_version(self, major: int = 0, minor: int = 0, patch: int = 0,
                       prerelease: Optional[str] = None, build: Optional[str] = None) -> Version:
        """
        创建版本对象

        Args:
            major: 主版本号
            minor: 次版本号
            patch: 补丁版本号
            prerelease: 预发布标识符
            build: 构建标识符

        Returns:
            版本对象
        """
        ...


class VersionComparatorInterface(Protocol):
    """
    版本比较器接口

    定义版本比较和范围匹配功能
    """

    def compare(self, version1: Union[str, Version], version2: Union[str, Version]) -> int:
        """
        比较两个版本

        Returns:
            -1: version1 < version2
             0: version1 == version2
             1: version1 > version2
        """
        ...

    def satisfies_range(self, version: Union[str, Version], range_spec: str) -> bool:
        """
        检查版本是否满足范围规范

        Args:
            version: 版本
            range_spec: 范围规范

        Returns:
            是否满足范围
        """
        ...


class VersionStorage(ABC):
    """
    版本存储抽象基类

    定义版本数据的存储和管理接口
    """

    @abstractmethod
    def save_version(self, name: str, version: Version, data: Dict[str, Any]) -> bool:
        """
        保存版本数据

        Args:
            name: 版本名称
            version: 版本对象
            data: 版本数据

        Returns:
            保存是否成功
        """

    @abstractmethod
    def load_version(self, name: str, version: Union[str, Version]) -> Optional[Dict[str, Any]]:
        """
        加载版本数据

        Args:
            name: 版本名称
            version: 版本号

        Returns:
            版本数据，不存在返回None
        """

    @abstractmethod
    def list_versions(self, name: str) -> List[Version]:
        """
        列出指定名称的所有版本

        Args:
            name: 版本名称

        Returns:
            版本列表
        """

    @abstractmethod
    def delete_version(self, name: str, version: Union[str, Version]) -> bool:
        """
        删除指定版本

        Args:
            name: 版本名称
            version: 版本号

        Returns:
            删除是否成功
        """


class VersionManagerInterface(ABC):
    """
    版本管理器接口

    定义版本管理的核心功能
    """

    @abstractmethod
    def register_version(self, name: str, version: Union[str, Version]) -> bool:
        """
        注册版本

        Args:
            name: 版本名称
            version: 版本对象或字符串

        Returns:
            注册是否成功
        """

    @abstractmethod
    def get_version(self, name: str) -> Optional[Version]:
        """
        获取版本

        Args:
            name: 版本名称

        Returns:
            版本对象，不存在返回None
        """

    @abstractmethod
    def set_current_version(self, name: str, version: Union[str, Version]) -> bool:
        """
        设置当前版本

        Args:
            name: 版本名称
            version: 版本对象或字符串

        Returns:
            设置是否成功
        """

    @abstractmethod
    def list_versions(self) -> Dict[str, Version]:
        """
        列出所有版本

        Returns:
            版本名称到版本对象的映射
        """


class VersionPolicyInterface(ABC):
    """
    版本策略接口

    定义版本验证和策略管理功能
    """

    @abstractmethod
    def validate_version(self, version: Union[str, Version], policy_name: str = None) -> bool:
        """
        验证版本

        Args:
            version: 版本对象或字符串
            policy_name: 策略名称

        Returns:
            版本是否符合策略
        """

    @abstractmethod
    def add_policy(self, name: str, policy_func) -> None:
        """
        添加策略

        Args:
            name: 策略名称
            policy_func: 策略函数
        """

    @abstractmethod
    def list_policies(self) -> List[str]:
        """
        列出所有策略

        Returns:
            策略名称列表
        """


class DataVersionManagerInterface(ABC):
    """
    数据版本管理器接口

    定义数据版本管理的功能
    """

    @abstractmethod
    def create_version(self, data: Any, metadata: Dict[str, Any] = None) -> str:
        """
        创建数据版本

        Args:
            data: 数据对象
            metadata: 元数据

        Returns:
            版本ID
        """

    @abstractmethod
    def get_version(self, version_id: str) -> Optional[Any]:
        """
        获取指定版本的数据

        Args:
            version_id: 版本ID

        Returns:
            数据对象，不存在返回None
        """

    @abstractmethod
    def list_versions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        列出版本历史

        Args:
            limit: 限制数量

        Returns:
            版本信息列表
        """

    @abstractmethod
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        比较两个版本

        Args:
            version_id1: 版本ID1
            version_id2: 版本ID2

        Returns:
            比较结果
        """


class ConfigVersionManagerInterface(ABC):
    """
    配置版本管理器接口

    定义配置版本管理的功能
    """

    @abstractmethod
    def create_version(self, config_name: str, config_data: Dict[str, Any],
                       creator: str = "system", description: str = "") -> Version:
        """
        创建配置版本

        Args:
            config_name: 配置名称
            config_data: 配置数据
            creator: 创建者
            description: 版本描述

        Returns:
            新版本号
        """

    @abstractmethod
    def get_config(self, config_name: str, version: Union[str, Version] = None) -> Optional[Dict[str, Any]]:
        """
        获取配置

        Args:
            config_name: 配置名称
            version: 版本号，None表示最新版本

        Returns:
            配置数据
        """

    @abstractmethod
    def rollback(self, config_name: str, version: Union[str, Version]) -> bool:
        """
        回滚配置

        Args:
            config_name: 配置名称
            version: 目标版本

        Returns:
            回滚是否成功
        """


# 类型别名
VersionData = Dict[str, Any]
PolicyFunction = callable  # 策略函数类型
