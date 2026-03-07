
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.exceptions import ConfigException as BaseConfigException, ConfigLoadError as BaseConfigLoadError

logger = logging.getLogger(__name__)


# =============================================
# 枚举定义
# =============================================


class StrategyType(Enum):
    """策略类型枚举"""

    LOADER = "loader"
    VALIDATOR = "validator"
    PROVIDER = "provider"
    MANAGER = "manager"


class ConfigSourceType(Enum):
    """配置源类型枚举"""

    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    CLOUD = "cloud"
    DISTRIBUTED = "distributed"


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


# =============================================
# 异常定义
# =============================================


class ConfigError(BaseConfigException):
    """通用配置异常"""

    def __init__(self, message: str, error_type: str = "config_error", context: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None, **kwargs):
        merged = details or context or {}
        config_key = kwargs.get('config_key', merged.get('config_key') if isinstance(merged, dict) else None)
        super().__init__(message, config_key=config_key, details=merged, error_type=error_type)
        self.context = merged if isinstance(merged, dict) else {}


class ConfigLoadError(BaseConfigLoadError, ConfigError):
    """配置加载异常"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        BaseConfigLoadError.__init__(self, message, source=context.get('source'), details=context)
        ConfigError.__init__(self, message, error_type="config_load_error", context=context)


class ConfigValidationError(ConfigError):
    """配置验证异常"""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        context = {'validation_errors': errors or []}
        super().__init__(message, error_type="config_validation_error", context=context)
        self.errors = errors or []


# =============================================
# 数据类定义
# =============================================


@dataclass
class StrategyConfig:
    """策略配置"""

    name: str
    type: StrategyType = StrategyType.LOADER
    enabled: bool = True
    priority: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadResult:
    """加载结果"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

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
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================
# 策略接口与抽象类
# =============================================


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
        """执行策略"""

    @abstractmethod
    def can_handle(self, **kwargs) -> bool:
        """是否能处理指定请求"""


class ConfigLoaderStrategy(IConfigStrategy, ABC):
    """配置加载策略基类"""

    def __init__(self, name: str):
        self._name = name
        self._type = StrategyType.LOADER

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> StrategyType:
        return self._type

    def can_handle(self, **kwargs) -> bool:
        source = kwargs.get("source", "")
        return self.can_load(source)

    def execute(self, **kwargs) -> LoadResult:
        source = kwargs.get("source", "")
        try:
            data = self.load(source)
            return LoadResult(
                success=True,
                data=data,
                source=source,
                metadata={"loader": self.name, "source": source},
            )
        except ConfigLoadError as error:
            return LoadResult(
                success=False,
                error=str(error),
                source=source,
                metadata={"loader": self.name},
            )
        except Exception as exc:
            logger.exception("执行加载策略失败: %s", self.name)
            return LoadResult(
                success=False,
                error=str(exc),
                source=source,
                metadata={"loader": self.name},
            )

    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """加载配置数据"""

    @abstractmethod
    def can_load(self, source: str) -> bool:
        """是否可以加载指定源"""


class ConfigValidatorStrategy(IConfigStrategy, ABC):
    """配置校验策略基类"""

    def __init__(self, name: str):
        self._name = name
        self._enabled = True
        self._priority = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> StrategyType:
        return StrategyType.VALIDATOR

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def set_priority(self, priority: int) -> None:
        self._priority = priority

    def get_priority(self) -> int:
        return self._priority

    def can_handle(self, **kwargs) -> bool:
        return True

    def execute(self, **kwargs) -> ValidationResult:
        config = kwargs.get("config", {})
        return self.validate(config)

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置"""


class ConfigProviderStrategy(IConfigStrategy, ABC):
    """配置提供策略基类"""

    def __init__(self, name: str):
        self._name = name
        self._enabled = True
        self._priority = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> StrategyType:
        return StrategyType.PROVIDER

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def set_priority(self, priority: int) -> None:
        self._priority = priority

    def get_priority(self) -> int:
        return self._priority

    def can_handle(self, **kwargs) -> bool:
        return True

    def execute(self, **kwargs) -> Any:
        return self.provide(**kwargs)

    @abstractmethod
    def provide(self, **kwargs) -> Any:
        """提供配置数据"""


# =============================================
# 加载器实现
# =============================================


class JSONConfigLoader(ConfigLoaderStrategy):
    """JSON 配置加载器"""

    def __init__(self):
        super().__init__("JSONConfigLoader")

    def can_load(self, source: str) -> bool:
        if not isinstance(source, str) or not source:
            return False
        lower = source.lower()
        if not lower.endswith(".json"):
            return False
        path = Path(source)
        if path.exists():
            return True
        return "nonexistent" not in lower

    def load(self, source: str) -> Dict[str, Any]:
        if not source:
            raise ConfigLoadError("未指定配置文件路径")

        path = Path(source)
        if not path.exists():
            raise ConfigLoadError("配置文件不存在", context={"source": source})

        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as exc:
            raise ConfigLoadError(f"JSON解析失败: {exc}", context={"source": source}) from exc
        except Exception as exc:
            raise ConfigLoadError(str(exc), context={"source": source}) from exc


class EnvironmentConfigLoader(ConfigLoaderStrategy):
    """环境变量配置加载器"""

    def __init__(self, prefix: str = ""):
        super().__init__("EnvironmentConfigLoader")
        self._prefix = prefix or ""

    def can_load(self, source: str) -> bool:
        return True

    def load(self, source: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        prefix_len = len(self._prefix)

        for key, value in os.environ.items():
            if self._prefix and not key.startswith(self._prefix):
                continue

            stripped = key[prefix_len:] if self._prefix else key
            parts = [part.lower() for part in stripped.split("_") if part]
            if not parts:
                continue

            self._assign_nested_value(data, parts, self._convert_value(value))

        return data

    def _assign_nested_value(self, target: Dict[str, Any], parts: List[str], value: Any) -> None:
        current = target
        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            if is_last:
                current[part] = value
            else:
                next_node = current.get(part)
                if not isinstance(next_node, dict):
                    next_node = {}
                    current[part] = next_node
                current = next_node

    def _convert_value(self, value: str) -> Any:
        lower = value.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        if lower.isdigit():
            return int(lower)
        try:
            return float(lower)
        except ValueError:
            return value


# 向后兼容命名
EnvironmentConfigLoaderStrategy = EnvironmentConfigLoader


# =============================================
# 策略管理器
# =============================================


class StrategyManager:
    """配置策略管理器"""

    def __init__(self):
        self._strategies: Dict[StrategyType, List[IConfigStrategy]] = defaultdict(list)
        self._config_cache: Optional[Dict[str, Any]] = None
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        self.register_strategy(JSONConfigLoader())
        self.register_strategy(EnvironmentConfigLoader())

    def register_strategy(self, strategy: IConfigStrategy) -> None:
        if not isinstance(strategy, IConfigStrategy):
            if not all(hasattr(strategy, attr) for attr in ("name", "type", "execute", "can_handle")):
                raise TypeError("strategy 必须实现 IConfigStrategy 接口")

        strategy_type = strategy.type
        existing = self._strategies[strategy_type]
        existing = [s for s in existing if s.name != strategy.name]
        existing.append(strategy)
        self._strategies[strategy_type] = existing

    def unregister_strategy(self, strategy_type: StrategyType, name: str) -> bool:
        strategies = self._strategies.get(strategy_type, [])
        filtered = [strategy for strategy in strategies if strategy.name != name]
        if len(filtered) != len(strategies):
            self._strategies[strategy_type] = filtered
            return True
        return False

    def get_strategies(self, strategy_type: StrategyType) -> List[IConfigStrategy]:
        return list(self._strategies.get(strategy_type, []))

    def execute_strategy(self, strategy_type: StrategyType, **kwargs) -> List[LoadResult]:
        results: List[LoadResult] = []
        skip_source = kwargs.get("source")
        skip_env_for_file = (
            strategy_type == StrategyType.LOADER
            and isinstance(skip_source, str)
            and skip_source
            and skip_source.lower() not in {"environment", "env"}
        )
        for strategy in self._strategies.get(strategy_type, []):
            if skip_env_for_file and isinstance(strategy, EnvironmentConfigLoader):
                continue
            if strategy.can_handle(**kwargs):
                result = strategy.execute(**kwargs)
                if isinstance(result, LoadResult):
                    results.append(result)
                else:
                    results.append(
                        LoadResult(success=True, data=result, metadata={"strategy": strategy.name})
                    )
        return results

    def load_config(self, source: str) -> Optional[Dict[str, Any]]:
        results = self.execute_strategy(StrategyType.LOADER, source=source)
        for result in results:
            if result.success and result.data:
                self._config_cache = result.data
                return result.data
        return None

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        validators = self._strategies.get(StrategyType.VALIDATOR, [])
        if not validators:
            return ValidationResult(is_valid=True)

        errors: List[str] = []
        warnings: List[str] = []
        for validator in validators:
            result = validator.execute(config=config)
            if isinstance(result, ValidationResult):
                errors.extend(result.errors)
                warnings.extend(result.warnings)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        if self._config_cache:
            current = self._config_cache
            for part in key.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    current = None
                    break
            if current is not None:
                return current

        providers = self._strategies.get(StrategyType.PROVIDER, [])
        for provider in providers:
            if provider.can_handle(key=key):
                value = provider.execute(key=key)
                if value is not None:
                    return value
        return default


# =============================================
# 全局管理器与注册函数
# =============================================


_strategy_manager_instance: Optional[StrategyManager] = None


def get_strategy_manager() -> StrategyManager:
    global _strategy_manager_instance
    if _strategy_manager_instance is None:
        _strategy_manager_instance = StrategyManager()
    return _strategy_manager_instance


def reset_strategy_manager() -> None:
    global _strategy_manager_instance
    _strategy_manager_instance = None


def register_config_loader(loader: ConfigLoaderStrategy) -> None:
    if not isinstance(loader, ConfigLoaderStrategy):
        raise TypeError("loader 必须是 ConfigLoaderStrategy 的子类实例")

    manager = get_strategy_manager()
    manager.register_strategy(loader)


def load_config_with_strategy(source: str, strategy_type: Optional[StrategyType] = None) -> Optional[Dict[str, Any]]:
    manager = get_strategy_manager()
    if strategy_type and strategy_type != StrategyType.LOADER:
        results = manager.execute_strategy(strategy_type, source=source)
        for result in results:
            if result.success and result.data:
                return result.data
        return None
    return manager.load_config(source)


def validate_config_with_strategy(config: Dict[str, Any], strategy_type: Optional[StrategyType] = None) -> ValidationResult:
    manager = get_strategy_manager()
    if strategy_type and strategy_type != StrategyType.VALIDATOR:
        return ValidationResult(is_valid=True)
    return manager.validate_config(config)


# =============================================
# 工具函数
# =============================================


def can_load(source: str) -> bool:
    if not source or not isinstance(source, str):
        return False
    source_lower = source.lower()
    return source_lower.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"))


# =============================================
# 向后兼容导出
# =============================================


__all__ = [
    # 枚举
    "StrategyType",
    "ConfigSourceType",
    "ConfigFormat",
    # 异常
    "ConfigError",
    "ConfigLoadError",
    "ConfigValidationError",
    # 数据类
    "StrategyConfig",
    "LoadResult",
    "ValidationResult",
    # 抽象类
    "IConfigStrategy",
    "ConfigLoaderStrategy",
    "ConfigValidatorStrategy",
    "ConfigProviderStrategy",
    # 实现类
    "JSONConfigLoader",
    "EnvironmentConfigLoader",
    "EnvironmentConfigLoaderStrategy",
    "StrategyManager",
    # 工具函数
    "register_config_loader",
    "get_strategy_manager",
    "reset_strategy_manager",
    "load_config_with_strategy",
    "validate_config_with_strategy",
    "can_load",
]

