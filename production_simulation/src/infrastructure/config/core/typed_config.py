
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Optional,
    Type,
    TypeVar,
    Generic,
    Union,
    List,
    Dict,
    get_origin,
    get_args,
)
from collections.abc import Iterable
from weakref import WeakKeyDictionary


T = TypeVar('T')


_CONFIG_CACHE: Dict[Any, 'TypedConfigBase'] = {}
_MANAGER_KEYS: 'WeakKeyDictionary[Any, int]' = WeakKeyDictionary()
_MANAGER_COUNTER = 0


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


def _is_type_like(value: Any) -> bool:
    """判断传入对象是否看起来像类型提示"""
    if value is None or value == 'default':
        return False
    if isinstance(value, type):
        return True
    origin = get_origin(value)
    return origin is not None


def _get_manager_key(manager: Any) -> Any:
    if manager is None:
        return None
    try:
        return _MANAGER_KEYS[manager]
    except KeyError:
        global _MANAGER_COUNTER
        _MANAGER_COUNTER += 1
        _MANAGER_KEYS[manager] = _MANAGER_COUNTER
        return _MANAGER_COUNTER


class ConfigTypeError(TypeError):
    """配置类型错误"""
    pass


class ConfigAccessError(KeyError):
    """配置访问错误"""
    def __str__(self) -> str:
        return self.args[0] if self.args else super().__str__()


class ConfigValueError(ValueError):
    """配置值错误"""
    pass


@dataclass
class TypedConfigValue(Generic[T]):
    """类型化配置值"""

    key: str
    type_hint: Type[T]
    default: Optional[T] = None
    description: str = ""
    value: Optional[T] = None

    def __post_init__(self):
        """初始化后处理 - 添加私有属性以兼容测试"""
        self._value = self.value
        self._loaded = False

    def clone(self) -> 'TypedConfigValue[T]':
        """创建当前配置值的副本，重置缓存状态"""
        copied = TypedConfigValue(
            key=self.key,
            type_hint=self.type_hint,
            default=self.default,
            description=self.description,
            value=self.value,
        )
        copied._value = None
        copied._loaded = False
        return copied

    def get_value(self) -> T:
        """获取配置值"""
        return self.value

    def set_value(self, value: T) -> None:
        """设置配置值"""
        self.value = value
        self._value = value
        self._loaded = True

    def _convert_to_type(self, raw_value: Any, target_type: Type[Any]) -> Any:
        """将值转换为目标类型，供Union等场景复用，可在测试中模拟异常"""

        if target_type == str:
            return str(raw_value)
        if target_type == int:
            return int(raw_value)
        if target_type == float:
            return float(raw_value)
        if target_type == bool:
            if isinstance(raw_value, bool):
                return raw_value
            if isinstance(raw_value, (int, float)):
                return bool(raw_value)
            if isinstance(raw_value, str):
                normalized = raw_value.strip().lower()
                truthy = {"true", "1", "yes", "on"}
                falsy = {"false", "0", "no", "off"}
                if normalized in truthy:
                    return True
                if normalized in falsy:
                    return False
            raise ValueError(f"无法将值 {raw_value!r} 转换为布尔值")
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            if isinstance(raw_value, target_type):
                return raw_value
            if isinstance(raw_value, str):
                normalized = raw_value.strip()
                try:
                    return target_type[normalized.upper()]
                except KeyError as exc:
                    raise ValueError from exc
            return target_type(raw_value)
        return TypedConfigValue(key=self.key, type_hint=target_type)._convert_value(raw_value)

    def get(self, config_manager: Any) -> T:
        """从配置管理器获取值（兼容测试接口）
        
        Args:
            config_manager: 配置管理器实例
            
        Returns:
            配置值
        """
        # 如果已加载，返回缓存值
        if self._loaded and self._value is not None:
            return self._value
        
        try:
            # 从配置管理器获取
            raw_value = config_manager.get(self.key)
            converted = self._convert_value(raw_value)
            self._value = converted
            self.value = converted
            self._loaded = True
            return converted
        except Exception as e:
            # 如果有默认值，使用默认值
            if self.default is not None:
                self._value = self.default
                self.value = self.default
                self._loaded = True
                return self.default
            raise ConfigAccessError(f"无法加载配置值 '{self.key}': {e}")

    def _convert_value(self, raw_value: Any) -> T:
        """转换值到目标类型
        
        Args:
            raw_value: 原始值
            
        Returns:
            转换后的值
        """
        # 处理None值
        if raw_value is None:
            if self.default is not None:
                return self.default

            origin = get_origin(self.type_hint)
            if origin is Union and type(None) in get_args(self.type_hint):
                return None

            raise ConfigValueError(f"配置值 '{self.key}' 为空且没有默认值")

        target = self.type_hint
        origin = get_origin(target)

        try:
            # 基础类型
            if target == str:
                return str(raw_value)
            if target == int:
                try:
                    return int(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ConfigTypeError(f"无法将值 {raw_value!r} 转换为 int") from exc
            if target == float:
                try:
                    return float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ConfigTypeError(f"无法将值 {raw_value!r} 转换为 float") from exc
            if target == bool:
                if isinstance(raw_value, bool):
                    return raw_value
                if isinstance(raw_value, (int, float)):
                    return bool(raw_value)
                if isinstance(raw_value, str):
                    normalized = raw_value.strip().lower()
                    truthy = {"true", "1", "yes", "on"}
                    falsy = {"false", "0", "no", "off"}
                    if normalized in truthy:
                        return True
                    if normalized in falsy:
                        return False
                    raise ConfigTypeError(f"无法将值 {raw_value!r} 转换为布尔值")
                raise ConfigTypeError(f"无法将值 {raw_value!r} 转换为布尔值")
            if target == list:
                if isinstance(raw_value, list):
                    return raw_value
                if isinstance(raw_value, tuple):
                    return list(raw_value)
                if isinstance(raw_value, Iterable) and not isinstance(raw_value, (str, bytes, dict)):
                    return list(raw_value)
                raise ConfigTypeError(f"值 {raw_value!r} 不是列表")
            if target == dict:
                if isinstance(raw_value, dict):
                    return raw_value
                try:
                    converted = dict(raw_value)
                    return converted
                except Exception as exc:
                    raise ConfigTypeError(f"值 {raw_value!r} 不是字典") from exc

            # 泛型 List/Dict
            if origin in (list, List):
                if isinstance(raw_value, list):
                    sequence = raw_value
                elif isinstance(raw_value, tuple):
                    sequence = list(raw_value)
                else:
                    raise ConfigTypeError(f"值 {raw_value!r} 不是列表")

                element_type = get_args(target)[0] if get_args(target) else Any
                if element_type in (Any, object):
                    return sequence

                converted_items = []
                for item in sequence:
                    if isinstance(item, element_type):
                        converted_items.append(item)
                    else:
                        try:
                            converted_items.append(element_type(item))
                        except Exception as exc:
                            raise ConfigTypeError(f"列表元素 {item!r} 无法转换为 {element_type}") from exc
                return converted_items

            if origin in (dict, Dict):
                if not isinstance(raw_value, dict):
                    raise ConfigTypeError(f"值 {raw_value!r} 不是字典")
                return raw_value

            if origin is Union:
                args = [arg for arg in get_args(target) if arg is not type(None)]
                last_error: Optional[Exception] = None
                for arg in args:
                    if arg in (Any, object):
                        return raw_value
                    if isinstance(arg, type) and isinstance(raw_value, arg):
                        return raw_value
                for arg in args:
                    try:
                        if isinstance(arg, type):
                            converted = self._convert_to_type(raw_value, arg)
                        else:
                            converted = TypedConfigValue(key=self.key, type_hint=arg)._convert_value(raw_value)
                        if isinstance(arg, type) and not isinstance(converted, arg):
                            continue
                        return converted
                    except (ConfigTypeError, ConfigValueError, ValueError, TypeError) as exc:
                        last_error = exc
                        continue
                target_name = args[0].__name__ if args and hasattr(args[0], '__name__') else str(args[0]) if args else str(target)
                raise ConfigTypeError(
                    f"无法将值 {raw_value!r} 转换为 {target_name}; 无法将值 {raw_value!r} 转换为任何 Union 类型"
                ) from last_error

            if isinstance(target, type) and issubclass(target, Enum):
                if isinstance(raw_value, target):
                    return raw_value
                if isinstance(raw_value, str):
                    normalized = raw_value.strip()
                    try:
                        return target[normalized.upper()]
                    except KeyError:
                        for member in target:
                            if str(member.value).lower() == normalized.lower():
                                return member
                    raise ConfigTypeError(f"无法将值 {raw_value!r} 转换为枚举类型 {target.__name__}")
                try:
                    return target(raw_value)
                except ValueError as exc:
                    raise ConfigTypeError(f"无法将值 {raw_value!r} 转换为枚举类型 {target.__name__}") from exc

            if hasattr(target, '__dataclass_fields__'):
                if isinstance(raw_value, target):
                    return raw_value
                if isinstance(raw_value, dict):
                    return target(**raw_value)
                raise ConfigTypeError("无法将非字典值转换为数据类")

            return raw_value

        except ConfigTypeError:
            raise
        except Exception as exc:
            raise ConfigTypeError(f"无法将值 {raw_value!r} 转换为类型 {target}") from exc

    def validate(self) -> bool:
        """验证值类型"""
        if self.value is None:
            return self.default is not None
        return isinstance(self.value, self.type_hint)


class TypedConfigBase:
    """类型化配置基类"""

    def __init__(self, config_manager: Optional[Any] = None, env: str = "default"):
        """初始化类型化配置
        
        Args:
            config_manager: 可选的配置管理器（兼容测试）
            env: 环境名称
        """
        self._config_manager = config_manager
        self._env = env
        self._config_values: Dict[str, TypedConfigValue[Any]] = {}
        self._initialize_config_values()

    def _initialize_config_values(self) -> None:
        """初始化类属性中定义的TypedConfigValue"""
        for attr_name, attr_value in self.__class__.__dict__.items():
            if isinstance(attr_value, TypedConfigValue):
                clone = attr_value.clone()
                self._config_values[attr_name] = clone
                object.__setattr__(self, attr_name, clone)
        
    def __getattribute__(self, name: str) -> Any:
        """重载属性访问以支持config_value装饰器"""
        # 先尝试正常的属性访问
        try:
            attr = object.__getattribute__(self, name)
            # 如果是TypedConfigValue，从配置管理器获取值
            if isinstance(attr, TypedConfigValue):
                stored = self._config_values.get(name, attr)
                if self._config_manager:
                    return stored.get(self._config_manager)
                return stored.get_value()
            return attr
        except AttributeError:
            # 尝试从_config_values获取
            config_values = object.__getattribute__(self, '_config_values')
            if name in config_values:
                config_val = config_values[name]
                if isinstance(config_val, TypedConfigValue):
                    config_manager = object.__getattribute__(self, '_config_manager')
                    if config_manager:
                        return config_val.get(config_manager)
                    return config_val.get_value()
                return config_val
            raise

    def set_config(self, key: str, value: TypedConfigValue) -> None:
        """设置配置项"""
        if isinstance(value, TypedConfigValue):
            clone = value.clone()
            self._config_values[key] = clone
            object.__setattr__(self, key, clone)
        else:
            self._config_values[key] = value

    def get_config(self, key: str) -> Optional[TypedConfigValue]:
        """获取配置项"""
        return self._config_values.get(key)

    def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        config = self.get_config(key)
        if isinstance(config, TypedConfigValue):
            if self._config_manager:
                return config.get(self._config_manager)
            return config.get_value()
        return config if config is not None else default

    def validate(self) -> ValidationResult:
        """验证当前配置集合"""
        errors: List[str] = []
        warnings: List[str] = []

        for key, config_value in self._config_values.items():
            if isinstance(config_value, TypedConfigValue):
                try:
                    # 如果有配置管理器，确保值已经加载以触发转换
                    if self._config_manager and not getattr(config_value, "_loaded", False):
                        config_value.get(self._config_manager)

                    if not config_value.validate():
                        errors.append(f"配置项 '{key}' 类型验证失败")
                except Exception as exc:
                    errors.append(f"配置项 '{key}' 验证异常: {exc}")
            else:
                warnings.append(f"配置项 '{key}' 非 TypedConfigValue 类型，跳过验证")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors or None,
            warnings=warnings or None,
        )


class TypedConfigSimple(TypedConfigBase):
    """简单类型化配置"""

    def __init__(self, config_manager: Optional[Any] = None):
        """初始化简单类型化配置

        Args:
            config_manager: 可选的配置管理器
        """
        super().__init__(config_manager)
        self._config: Dict[str, Any] = {}
        self._types: Dict[str, Type[Any]] = {}

    def set_string(self, key: str, value: str, description: str = "") -> None:
        """设置字符串配置"""
        self.set_config(key, TypedConfigValue(key=key, type_hint=str, default=value, description=description, value=value))
        self._config[key] = value
        self._types[key] = str

    def set_int(self, key: str, value: int, description: str = "") -> None:
        """设置整数配置"""
        self.set_config(key, TypedConfigValue(key=key, type_hint=int, default=value, description=description, value=value))
        self._config[key] = value
        self._types[key] = int

    def set_bool(self, key: str, value: bool, description: str = "") -> None:
        """设置布尔配置"""
        self.set_config(key, TypedConfigValue(key=key, type_hint=bool, default=value, description=description, value=value))
        self._config[key] = value
        self._types[key] = bool
    
    def set_typed(self, key: str, value: Any, type_hint: Type) -> None:
        """设置类型化值（通用方法）"""
        self.set_config(key, TypedConfigValue(key=key, type_hint=type_hint, default=value, value=value))
        self._config[key] = value
        self._types[key] = type_hint
    
    def get_typed(self, key: str, type_hint: Type) -> Any:
        """获取类型化值"""
        if key not in self._config:
            return None
        expected = self._types.get(key)
        if expected is not None and expected is not type_hint:
            return None
        return self._config.get(key)
    
    def validate_type(self, value: Any, type_hint: Type) -> bool:
        """验证类型"""
        return isinstance(value, type_hint)


class TypedConfig(TypedConfigSimple):
    """类型化配置（向后兼容的简单别名）"""
    pass


class TypedConfiguration(TypedConfigBase):
    """高级类型化配置"""

    def __init__(self, config_manager: Optional[Any] = None):
        """初始化高级类型化配置
        
        Args:
            config_manager: 可选的配置管理器
        """
        super().__init__(config_manager)
        self._validators = {}

    def add_validator(self, key: str, validator_func) -> None:
        """添加验证器"""
        self._validators[key] = validator_func

    def validate_all(self) -> bool:
        """验证所有配置"""
        for key, validator in self._validators.items():
            config = self.get_config(key)
            if isinstance(config, TypedConfigValue):
                value = config.get(self._config_manager) if self._config_manager else config.get_value()
            else:
                value = config
            if config and not validator(value):
                return False
        return True


def config_value(
    key: str,
    type_hint: Any = 'default',
    default: Optional[Any] = None,
    description: str = "",
) -> TypedConfigValue[Any]:
    """创建类型化配置值（兼容多种调用方式）"""

    original_hint = type_hint
    hint = original_hint
    default_value = default
    desc = description

    if not _is_type_like(original_hint):
        if desc == "" and default_value is not None:
            desc = default_value
            default_value = original_hint
        else:
            default_value = original_hint
        hint = Any
    else:
        hint = original_hint

    if hint is None or hint == 'default':
        hint = Any

    return TypedConfigValue(key=key, type_hint=hint, default=default_value, description=desc)


def get_typed_config(
    config_cls: Union[Type['TypedConfigBase'], 'TypedConfigBase'],
    config_manager: Optional[Any] = None,
    env: str = "default",
) -> 'TypedConfigBase':
    """获取或创建类型化配置实例，支持缓存"""

    if isinstance(config_cls, TypedConfigBase):
        return config_cls

    if not isinstance(config_cls, type) or not issubclass(config_cls, TypedConfigBase):
        raise TypeError("config_cls 必须是 TypedConfigBase 的子类")

    manager_key = _get_manager_key(config_manager)
    cache_key = (config_cls, manager_key, env)

    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None and cached._config_manager is config_manager:
        return cached

    instance = config_cls(config_manager, env)
    _CONFIG_CACHE[cache_key] = instance
    return instance


class TypedConfigComplex(TypedConfigBase):
    """复杂类型化配置"""

    def __init__(self, config_manager: Optional[Any] = None):
        """初始化复杂类型化配置
        
        Args:
            config_manager: 可选的配置管理器
        """
        super().__init__(config_manager)
        self._nested_configs = {}

    def set_nested_config(self, key: str, config: TypedConfigBase) -> None:
        """设置嵌套配置"""
        self._nested_configs[key] = config

    def get_nested_config(self, key: str) -> Optional[TypedConfigBase]:
        """获取嵌套配置"""
        return self._nested_configs.get(key)

    def set_complex_value(self, key: str, value: Union[Dict, List, Any], description: str = "") -> None:
        """设置复杂类型值"""
        # 对于复杂类型，我们使用Any作为类型提示
        config_value_obj = TypedConfigValue(key=key, type_hint=type(value), default=value, description=description, value=value)
        self.set_config(key, config_value_obj)

    def validate_complex_config(self) -> ValidationResult:
        """验证复杂配置"""
        errors = []
        warnings = []

        for key, config in self._nested_configs.items():
            if hasattr(config, 'validate_all'):
                if not config.validate_all():
                    errors.append(f"嵌套配置 {key} 验证失败")

        for key, config_value in self._config_values.items():
            if not config_value.validate():
                errors.append(f"配置项 {key} 类型验证失败")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )
