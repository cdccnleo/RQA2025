
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from enum import Enum
from typing import Union

try:
    # 尝试导入统一接口中的StrategyType，保持一致性
    from src.infrastructure.config.interfaces.unified_interface import StrategyType as UnifiedStrategyType  # type: ignore
except ImportError:
    UnifiedStrategyType = None  # type: ignore


class StrategyType(Enum):
    """策略类型枚举"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    LOADER = "loader"
    VALIDATOR = "validator"
    PROCESSOR = "processor"


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


class ConfigSourceType(Enum):
    """配置源类型枚举"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"


@dataclass
class StrategyConfig:
    """策略配置"""
    type: str = "loader"
    name: str = ""
    enabled: bool = True
    priority: int = 0
    timeout: float = 30.0
    retry_count: int = 3
    config: Optional[Dict[str, Any]] = None


@dataclass
class LoadResult:
    """加载结果"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None and self.success:
            self.metadata = {}

    @property
    def error_message(self) -> Optional[str]:
        return self.error

    @error_message.setter
    def error_message(self, value: Optional[str]) -> None:
        self.error = value


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class BaseConfigStrategy(ABC):
    """配置策略基类"""

    def __init__(self, config: Union[StrategyConfig, str, Dict[str, Any]]):
        if isinstance(config, StrategyConfig):
            strategy_config = config
        elif isinstance(config, str):
            strategy_config = StrategyConfig(name=config)
        elif isinstance(config, dict):
            strategy_config = StrategyConfig(**config)
        else:
            raise TypeError(f"不支持的配置类型: {type(config)!r}")

        self.config = strategy_config
        self._name = strategy_config.name or self.__class__.__name__
        self._strategy_type = strategy_config.type
        self._enabled = strategy_config.enabled
        self._priority = strategy_config.priority
        self._supported_formats: List[Any] = []
        self._supported_sources: List[Any] = []
        # 向后兼容的公开属性
        self.enabled = self._enabled
        self.priority = self._priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def strategy_type(self) -> Union[str, StrategyType]:
        candidate = self._strategy_type

        # 优先使用统一接口的枚举类型
        if UnifiedStrategyType:
            try:
                if isinstance(candidate, UnifiedStrategyType):
                    return candidate
                return UnifiedStrategyType(candidate)
            except Exception:
                pass

        # 回退到当前模块定义的枚举
        try:
            if isinstance(candidate, StrategyType):
                return candidate
            return StrategyType(candidate)
        except Exception:
            return candidate

    def execute(self, source: str) -> Dict[str, Any]:
        """执行策略逻辑"""
        raise NotImplementedError("子类必须实现 execute 方法以返回具体的配置数据")

    def load_config(self, source: str) -> LoadResult:
        """兼容旧接口的加载方法"""
        return self.load(source)

    def load(self, source: str, config_format: Optional[ConfigFormat] = None) -> LoadResult:
        """加载配置"""
        if not self._enabled:
            return LoadResult(success=False, error="策略未启用", source=source, metadata=None)

        if not self.can_handle(source, config_format):
            return LoadResult(success=False, error="不支持的配置源", source=source, metadata={"strategy": self.name})

        try:
            data = self.execute(source)
            return LoadResult(
                success=True,
                data=data,
                source=source,
                source_type=self._strategy_type,
                metadata={"strategy": self.name}
            )
        except Exception as exc:
            return LoadResult(
                success=False,
                error=str(exc),
                source=source,
                source_type=self._strategy_type,
                metadata={"strategy": self.name}
            )

    def can_handle_source(self, source: str) -> bool:
        """检查是否可以处理指定源"""
        return True

    def can_handle(self, source: str, config_format: Optional[ConfigFormat] = None) -> bool:
        """检查是否可以处理指定源及格式"""
        if not self.can_handle_source(source):
            return False
        if config_format is None or not self._supported_formats:
            return True
        return config_format in self._supported_formats

    def get_supported_formats(self) -> List[Any]:
        """获取支持的格式"""
        return list(self._supported_formats)

    def get_supported_sources(self) -> List[Any]:
        """获取支持的源"""
        return list(self._supported_sources)

    def is_enabled(self) -> bool:
        """检查策略是否启用"""
        return self._enabled

    def get_priority(self) -> int:
        """获取优先级"""
        return self._priority

    def set_enabled(self, enabled: bool) -> None:
        """设置启用状态"""
        self._enabled = enabled
        self.enabled = enabled

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        self.set_enabled(False)

    def set_priority(self, priority: int) -> None:
        """设置优先级"""
        self._priority = priority
        self.priority = priority

    def is_enabled(self) -> bool:
        """检查策略是否启用"""
        return self.enabled

    def get_priority(self) -> int:
        """获取优先级"""
        return self.priority

    def set_enabled(self, enabled: bool) -> None:
        """设置启用状态"""
        self.enabled = enabled

    def set_priority(self, priority: int) -> None:
        """设置优先级"""
        self.priority = priority


class FileConfigStrategy(BaseConfigStrategy):
    """文件配置策略"""

    def load_config(self, source: str) -> LoadResult:
        """从文件加载配置"""
        try:
            # 这里应该实现文件读取逻辑
            return LoadResult(
                success=True,
                data={"file_loaded": True},
                source_type="file",
                metadata={"source": source}
            )
        except Exception as e:
            return LoadResult(
                success=False,
                error_message=str(e),
                source_type="file"
            )


class EnvironmentConfigStrategy(BaseConfigStrategy):
    """环境变量配置策略"""

    def load_config(self, source: str) -> LoadResult:
        """从环境变量加载配置"""
        try:
            # 这里应该实现环境变量读取逻辑
            import os
            env_vars = {k: v for k, v in os.environ.items() if k.startswith(source)}
            return LoadResult(
                success=True,
                data=env_vars,
                source_type="environment",
                metadata={"prefix": source}
            )
        except Exception as e:
            return LoadResult(
                success=False,
                error_message=str(e),
                source_type="environment"
            )
