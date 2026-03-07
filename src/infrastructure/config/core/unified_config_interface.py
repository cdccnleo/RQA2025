
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
#!/usr/bin/env python3
"""
统一配置管理接口

定义基础设施层配置管理的标准接口，确保所有配置管理器实现统一的API。
"""


class ConfigSource(Enum):
    """配置源类型"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"


class ConfigPriority(Enum):
    """配置优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class IConfigManager(ABC):
    """
    配置管理器统一接口

    所有配置管理器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点分隔的嵌套访问
            default: 默认值

        Returns:
            配置值
        """

    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值

        Args:
            key: 配置键，支持点分隔的嵌套设置
            value: 配置值

        Returns:
            是否设置成功
        """

    @abstractmethod
    def has(self, key: str) -> bool:
        """
        检查配置是否存在

        Args:
            key: 配置键

        Returns:
            是否存在
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除配置

        Args:
            key: 配置键

        Returns:
            是否删除成功
        """

    @abstractmethod
    def reload(self) -> bool:
        """
        重新加载配置

        Returns:
            是否重新加载成功
        """

    @abstractmethod
    def save(self) -> bool:
        """
        保存配置到持久化存储

        Returns:
            是否保存成功
        """

    @abstractmethod
    def get_all(self, prefix: str = "") -> Dict[str, Any]:
        """
        获取所有配置（可选前缀过滤）

        Args:
            prefix: 前缀过滤

        Returns:
            所有配置的字典
        """

    @abstractmethod
    def validate(self) -> List[str]:
        """
        验证配置有效性

        Returns:
            验证错误列表，空列表表示验证通过
        """

    @abstractmethod
    def get_sources(self) -> List[ConfigSource]:
        """
        获取配置源列表

        Returns:
            配置源列表
        """

    @abstractmethod
    def add_source(self, source: ConfigSource, config: Dict[str, Any], priority: ConfigPriority = ConfigPriority.NORMAL) -> bool:
        """
        添加配置源

        Args:
            source: 配置源类型
            config: 配置数据
            priority: 优先级

        Returns:
            是否添加成功
        """

    @abstractmethod
    def remove_source(self, source: ConfigSource) -> bool:
        """
        移除配置源

        Args:
            source: 配置源类型

        Returns:
            是否移除成功
        """

    @abstractmethod
    def get_source_config(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """
        获取指定源的配置

        Args:
            source: 配置源类型

        Returns:
            配置数据
        """

    @abstractmethod
    def merge_configs(self, configs: List[Dict[str, Any]], strategy: str = "override") -> Dict[str, Any]:
        """
        合并多个配置

        Args:
            configs: 配置列表
            strategy: 合并策略 (override, merge, deep_merge)

        Returns:
            合并后的配置
        """

    @abstractmethod
    def watch(self, key: str, callback: callable) -> bool:
        """
        监听配置变化

        Args:
            key: 配置键
            callback: 回调函数

        Returns:
            是否监听成功
        """

    @abstractmethod
    def unwatch(self, key: str, callback: callable) -> bool:
        """
        取消监听配置变化

        Args:
            key: 配置键
            callback: 回调函数

        Returns:
            是否取消监听成功
        """

    @abstractmethod
    def export(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        导出配置

        Args:
            format: 导出格式 (json, yaml, xml, ini)

        Returns:
            导出的配置字符串或字典
        """

    @abstractmethod
    def import_config(self, config: Union[str, Dict[str, Any]], format: str = "json") -> bool:
        """
        导入配置

        Args:
            config: 配置数据
            format: 配置格式

        Returns:
            是否导入成功
        """


class IConfigValidator(ABC):
    """
    配置验证器接口
    """

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        验证配置

        Args:
            config: 配置字典

        Returns:
            验证错误列表
        """

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        获取配置模式

        Returns:
            配置模式字典
        """


class IConfigLoader(ABC):
    """
    配置加载器接口
    """

    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """
        加载配置

        Args:
            source: 配置源路径或标识符

        Returns:
            配置字典
        """

    @abstractmethod
    def save(self, config: Dict[str, Any], target: str) -> bool:
        """
        保存配置

        Args:
            config: 配置字典
            target: 保存目标路径或标识符

        Returns:
            是否保存成功
        """

    @abstractmethod
    def supports_format(self, format: str) -> bool:
        """
        是否支持指定的格式

        Args:
            format: 格式类型

        Returns:
            是否支持
        """


class IConfigMonitor(ABC):
    """
    配置监控器接口
    """

    @abstractmethod
    def on_config_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """
        配置变化回调

        Args:
            key: 配置键
            old_value: 旧值
            new_value: 新值
        """

    @abstractmethod
    def get_change_history(self, key: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取配置变化历史

        Args:
            key: 配置键（可选，None表示所有）
            limit: 限制返回数量

        Returns:
            变化历史列表
        """




