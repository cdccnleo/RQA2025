"""
类型安全的配置接口模块

提供了以下功能：
- 基于类型注解的配置访问
- 自动类型转换
- 配置值缓存
- 默认值支持
"""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, get_type_hints
import inspect
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from functools import lru_cache

T = TypeVar('T')

class ConfigValueError(Exception):
    """配置值错误"""
    pass

class ConfigTypeError(Exception):
    """配置类型错误"""
    pass

class ConfigAccessError(Exception):
    """配置访问错误"""
    pass

@dataclass
class TypedConfigValue(Generic[T]):
    """类型安全的配置值"""
    key: str
    type_hint: Type[T]
    default: Optional[T] = None
    description: str = ""
    _value: Optional[T] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def get(self, config_manager) -> T:
        """获取配置值"""
        if not self._loaded:
            try:
                raw_value = config_manager.get(self.key)
                self._value = self._convert_value(raw_value)
                self._loaded = True
            except Exception as e:
                if self.default is not None:
                    self._value = self.default
                    self._loaded = True
                else:
                    raise ConfigAccessError(f"无法加载配置值 '{self.key}': {str(e)}")
        return self._value

    def _convert_value(self, value: Any) -> T:
        """转换值类型"""
        if value is None:
            if self.default is not None:
                return self.default
            raise ConfigValueError(f"配置值 '{self.key}' 为空且没有默认值")

        # 处理Union类型
        origin = getattr(self.type_hint, "__origin__", None)
        if origin is Union:
            args = getattr(self.type_hint, "__args__", [])
            # 处理Optional[T]，即Union[T, None]
            if type(None) in args and len(args) == 2:
                if value is None:
                    return None
                non_none_type = next(arg for arg in args if arg is not type(None))
                return self._convert_to_type(value, non_none_type)

            # 尝试转换为Union中的任意类型
            for arg in args:
                try:
                    return self._convert_to_type(value, arg)
                except (ValueError, TypeError):
                    continue
            raise ConfigTypeError(f"无法将值 '{value}' 转换为任何 Union 类型: {self.type_hint}")

        # 处理List类型
        if origin is list:
            if not isinstance(value, list):
                raise ConfigTypeError(f"值 '{value}' 不是列表")
            item_type = getattr(self.type_hint, "__args__", [Any])[0]
            return [self._convert_to_type(item, item_type) for item in value]

        # 处理Dict类型
        if origin is dict:
            if not isinstance(value, dict):
                raise ConfigTypeError(f"值 '{value}' 不是字典")
            key_type, val_type = getattr(self.type_hint, "__args__", [Any, Any])
            return {
                self._convert_to_type(k, key_type): self._convert_to_type(v, val_type)
                for k, v in value.items()
            }

        # 处理普通类型
        return self._convert_to_type(value, self.type_hint)

    def _convert_to_type(self, value: Any, target_type: Type) -> Any:
        """将值转换为目标类型"""
        # 处理枚举类型
        if inspect.isclass(target_type) and issubclass(target_type, Enum):
            if isinstance(value, str):
                try:
                    return target_type[value]
                except KeyError:
                    try:
                        return target_type(value)
                    except ValueError:
                        raise ConfigTypeError(f"无法将 '{value}' 转换为枚举类型 {target_type.__name__}")
            elif isinstance(value, int):
                try:
                    return target_type(value)
                except ValueError:
                    raise ConfigTypeError(f"无法将 '{value}' 转换为枚举类型 {target_type.__name__}")

        # 处理基本类型
        if target_type is bool and isinstance(value, str):
            if value.lower() in ('true', 'yes', '1', 'on'):
                return True
            if value.lower() in ('false', 'no', '0', 'off'):
                return False
            raise ConfigTypeError(f"无法将字符串 '{value}' 转换为布尔值")

        # 处理数据类
        if hasattr(target_type, '__dataclass_fields__'):
            if not isinstance(value, dict):
                raise ConfigTypeError(f"无法将非字典值转换为数据类 {target_type.__name__}")

            # 获取数据类字段
            fields = {f.name: f for f in target_type.__dataclass_fields__.values()}
            kwargs = {}

            for field_name, field_info in fields.items():
                if field_name in value:
                    field_type = field_info.type
                    field_value = value[field_name]
                    kwargs[field_name] = self._convert_to_type(field_value, field_type)
                elif hasattr(field_info, 'default') and field_info.default is not field_info.default_factory:
                    kwargs[field_name] = field_info.default
                elif hasattr(field_info, 'default_factory') and field_info.default_factory is not field_info.default:
                    kwargs[field_name] = field_info.default_factory()

            return target_type(**kwargs)

        # 直接转换
        try:
            if target_type is str and not isinstance(value, str):
                return str(value)
            elif target_type is int and not isinstance(value, int):
                return int(value)
            elif target_type is float and not isinstance(value, float):
                return float(value)
            elif target_type is bool and not isinstance(value, bool):
                return bool(value)
            elif target_type is list and not isinstance(value, list):
                return list(value)
            elif target_type is dict and not isinstance(value, dict):
                return dict(value)
            return value
        except (ValueError, TypeError) as e:
            raise ConfigTypeError(f"无法将 '{value}' 转换为 {target_type.__name__}: {str(e)}")

class TypedConfigBase:
    """类型安全的配置基类"""
    def __init__(self, config_manager, env="default"):
        self._config_manager = config_manager
        self._env = env
        self._config_values = {}
        self._initialize_config_values()

    def _initialize_config_values(self):
        """初始化配置值"""
        # 获取类的类型注解
        type_hints = get_type_hints(self.__class__)

        # 遍历类的属性
        for name, attr in inspect.getmembers(self.__class__):
            if isinstance(attr, TypedConfigValue):
                # 使用类型注解更新TypedConfigValue
                if name in type_hints:
                    attr.type_hint = type_hints[name]
                self._config_values[name] = attr

    def __getattribute__(self, name):
        """重写属性访问，实现配置值的懒加载"""
        # 获取普通属性
        try:
            attr = super().__getattribute__(name)
            # 如果是TypedConfigValue，则获取实际值
            if isinstance(attr, TypedConfigValue):
                return attr.get(self._config_manager)
            return attr
        except AttributeError:
            # 检查是否在_config_values中
            config_values = super().__getattribute__('_config_values')
            if name in config_values:
                return config_values[name].get(self._config_manager)
            raise

def config_value(key: str, default: Any = None, description: str = "") -> Any:
    """创建类型安全的配置值

    用法:
        class MyConfig(TypedConfigBase):
            server_port: int = config_value("server.port", 8080)
            debug_mode: bool = config_value("debug", False)
    """
    return TypedConfigValue(key=key, type_hint=Any, default=default, description=description)

@lru_cache(maxsize=32)
def get_typed_config(config_class: Type[TypedConfigBase], config_manager, env="default") -> TypedConfigBase:
    """获取类型安全的配置实例（带缓存）"""
    return config_class(config_manager, env)
