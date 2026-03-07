
# ConfigScope 已移至 storage/types/configscope.py 中
# IConfigStorage 已统一到 storage/types/iconfigstorage.py 中

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Callable, Type, Optional
"""
统一接口定义文件
提供配置管理相关的标准接口和枚举
"""


class CachePolicy(Enum):
    """缓存策略枚举"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"


class ServiceStatus(Enum):
    """服务状态枚举"""
    # 通用状态
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"

    # 详细状态
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

    # 兼容性别名
    HEALTHY = "healthy"  # 别名 for UP

# ConfigItem 已统一到 storage/types/configitem.py 中


class IConfigManagerComponent(ABC):
    """配置管理器接口"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""

    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""

    @abstractmethod
    def update(self, config: Dict[str, Any]) -> None:
        """更新配置"""

    @abstractmethod
    def watch(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """监听配置变化"""

    @abstractmethod
    def reload(self) -> None:
        """重新加载配置"""

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> bool:
        """验证配置"""


class IConfigManagerFactoryComponent(ABC):
    """配置管理器工厂接口"""

    @abstractmethod
    def create_manager(self, manager_type: str, **kwargs) -> IConfigManagerComponent:
        """创建配置管理器"""

    @abstractmethod
    def register_manager(self, name: str, manager_class: Type[IConfigManagerComponent]) -> None:
        """注册配置管理器类型"""

    @abstractmethod
    def get_available_managers(self) -> Dict[str, Type[IConfigManagerComponent]]:
        """获取可用的配置管理器类型"""


# 向后兼容的别名
IConfigManager = IConfigManagerComponent

# ==================== 策略相关枚举 ====================


class StrategyType(Enum):
    """策略类型枚举"""
    LOADER = "loader"
    VALIDATOR = "validator"
    STORAGE = "storage"
    TRANSFORMER = "transformer"


class ConfigFormat(Enum):
    """配置格式枚举"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    XML = "xml"
    ENV = "env"
    DATABASE = "database"
    CLOUD = "cloud"


class LoaderResult(dict):
    """加载结果容器，兼容字典语义并支持解包元数据"""

    __slots__ = ("metadata",)

    def __init__(self, data: Optional[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(data or {})
        self.metadata: Dict[str, Any] = metadata or {}

    def __iter__(self):
        yield dict(self)
        yield self.metadata

    def as_tuple(self):
        return dict(self), self.metadata


class ConfigSourceType(Enum):
    """配置源类型枚举"""
    FILE = "file"
    DATABASE = "database"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    MEMORY = "memory"

# ==================== 策略接口 ====================


class IConfigStrategy(ABC):
    """配置策略接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""

    @property
    @abstractmethod
    def type(self) -> StrategyType:
        """策略类型"""

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        执行策略

        Returns:
        执行结果
        """

    @abstractmethod
    def can_handle(self, **kwargs) -> bool:
        """
        检查是否可以处理

        Returns:
        是否可以处理
        """


class ConfigLoaderStrategy(IConfigStrategy, ABC):
    """配置加载策略基类"""

    def __init__(self, name: str = ""):
        self._name = name or self.__class__.__name__
        self._supported_formats: List[ConfigFormat] = []
        self._supported_sources: List[ConfigSourceType] = []
        self._enabled = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> StrategyType:
        return StrategyType.LOADER

    def is_enabled(self) -> bool:
        """检查策略是否启用"""
        return self._enabled

    def enable(self):
        """启用策略"""
        self._enabled = True

    def disable(self):
        """禁用策略"""
        self._enabled = False

    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """
        加载配置

        Args:
        source: 配置源

        Returns:
        配置数据
        """

    @abstractmethod
    def can_handle_source(self, source: str) -> bool:
        """
        检查是否可以处理指定的配置源

        Args:
        source: 配置源

        Returns:
        是否可以处理
        """

    @abstractmethod
    def get_supported_formats(self) -> List[ConfigFormat]:
        """
        获取支持的配置格式

        Returns:
        支持的格式列表
        """

    def execute(self, **kwargs) -> Any:
        """
        执行策略

        Args:
            **kwargs: 执行参数

        Returns:
            执行结果
        """
        source = kwargs.get('source', '')
        if source:
            return self.load(source)
        return None

    def can_handle(self, **kwargs) -> bool:
        """
        检查是否可以处理

        Args:
            **kwargs: 检查参数

        Returns:
            是否可以处理
        """
        source = kwargs.get('source', '')
        if source:
            return self.can_handle_source(source)
        return False




