#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置测试的通用fixture和Mock类

提供所有配置测试需要的标准Mock实现
"""

from unittest.mock import Mock
from typing import Any, Dict, Optional
from dataclasses import dataclass


# ========== 策略模式相关Mock ==========

class MockConfigStrategy:
    """完整实现的Mock策略类 - 修复抽象方法问题"""

    def __init__(self, name: str, strategy_type: str = "FILE", enabled: bool = True):
        self._name = name
        self._type = strategy_type
        self._enabled = enabled
        self._priority = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def get_priority(self) -> int:
        return self._priority

    def set_priority(self, priority: int):
        self._priority = priority

    # 实现抽象方法 load_config
    def load_config(self, source: str) -> Dict[str, Any]:
        """加载配置 - 完整实现"""
        return {
            "success": True,
            "data": {"test": "value", "source": source},
            "source_type": self._type
        }

    def execute(self, **kwargs):
        """执行策略"""
        return {"success": True, "data": {"test": "value"}}


@dataclass
class MockLoadResult:
    """Mock加载结果"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    source_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MockValidationResult:
    """Mock验证结果"""
    is_valid: bool
    errors: Optional[list] = None
    warnings: Optional[list] = None


# ========== TypedConfig相关Mock ==========

class MockTypedConfigValue:
    """完整实现的TypedConfigValue Mock"""

    def __init__(self, key: str, type_hint: type, default: Any = None, description: str = ""):
        self.key = key
        self.type_hint = type_hint
        self.default = default
        self.description = description
        self.value = None  # 修复：添加value属性
        self._value = None  # 修复：添加_value私有属性
        self._loaded = False  # 修复：添加_loaded状态

    def get(self, config_manager):
        """获取配置值"""
        if self._loaded:
            return self._value

        try:
            value = config_manager.get(self.key)
            converted = self._convert_value(value)
            self._value = converted
            self._loaded = True
            return converted
        except Exception:
            if self.default is not None:
                self._value = self.default
                self._loaded = True
                return self.default
            raise ConfigAccessError(f"无法加载配置值 '{self.key}'")

    def _convert_value(self, value):
        """转换值类型"""
        if value is None:
            if self.default is not None:
                return self.default
            # 检查是否是Optional类型
            import typing
            if hasattr(self.type_hint, '__origin__'):
                if self.type_hint.__origin__ is typing.Union:
                    args = getattr(self.type_hint, '__args__', ())
                    if type(None) in args:
                        return None
            raise ConfigValueError(f"配置值 '{self.key}' 为空且没有默认值")

        # 类型转换
        if self.type_hint == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif self.type_hint == int:
            return int(value)
        elif self.type_hint == str:
            return str(value)
        elif self.type_hint == float:
            return float(value)
        elif self.type_hint == list:
            if isinstance(value, list):
                return value
            return [value]
        elif self.type_hint == dict:
            if isinstance(value, dict):
                return value
            return {"value": value}

        return value

    def get_value(self):
        """获取值"""
        return self._value

    def set_value(self, value):
        """设置值"""
        self._value = value
        self.value = value

    def validate(self) -> bool:
        """验证值类型"""
        if self._value is None:
            return self.default is not None
        return isinstance(self._value, self.type_hint)


class MockTypedConfigBase:
    """完整实现的TypedConfigBase Mock"""

    def __init__(self, config_manager=None):
        """初始化 - 修复：支持config_manager参数"""
        self._config_manager = config_manager
        self._config_values = {}

    def set_config(self, key: str, value):
        """设置配置项"""
        self._config_values[key] = value

    def get_config(self, key: str):
        """获取配置项"""
        return self._config_values.get(key)

    def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        config = self.get_config(key)
        if config:
            if hasattr(config, 'get_value'):
                return config.get_value()
            return config
        return default


class MockTypedConfigSimple(MockTypedConfigBase):
    """简单类型化配置Mock"""

    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self._config = {}  # 修复：添加_config属性

    def set_typed(self, key: str, value: Any, type_hint: type):
        """设置类型化值"""
        self._config[key] = value
        self.set_config(key, MockTypedConfigValue(key, type_hint, default=value))

    def get_typed(self, key: str, type_hint: type) -> Any:
        """获取类型化值"""
        return self._config.get(key)

    def validate_type(self, value: Any, type_hint: type) -> bool:
        """验证类型"""
        return isinstance(value, type_hint)


# ========== 配置管理器Mock ==========

class MockConfigManager:
    """统一的配置管理器Mock"""

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        self._data = initial_data or {}
        self._validation_rules = []

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if '.' in key:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        if '.' in key:
            keys = key.split('.')
            current = self._data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            self._data[key] = value
        return True

    def delete(self, key: str) -> bool:
        """删除配置值"""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def validate_config(self) -> bool:
        """验证配置"""
        return True

    def get_all_sections(self) -> list:
        """获取所有段"""
        if isinstance(self._data, dict):
            return list(self._data.keys())
        return []

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置数据"""
        return self._data

    @property
    def validation_rules(self):
        """验证规则"""
        return self._validation_rules


# ========== 异常类 ==========

class ConfigAccessError(KeyError):
    """配置访问错误"""
    pass


class ConfigValueError(ValueError):
    """配置值错误"""
    pass


class ConfigTypeError(TypeError):
    """配置类型错误"""
    pass


# ========== 辅助函数 ==========

def create_mock_strategy(name: str = "test_strategy",
                        strategy_type: str = "FILE",
                        enabled: bool = True) -> MockConfigStrategy:
    """创建Mock策略实例"""
    return MockConfigStrategy(name, strategy_type, enabled)


def create_mock_config_manager(initial_data: Optional[Dict[str, Any]] = None) -> MockConfigManager:
    """创建Mock配置管理器"""
    return MockConfigManager(initial_data)


def create_mock_typed_config_value(key: str,
                                    type_hint: type,
                                    default: Any = None) -> MockTypedConfigValue:
    """创建Mock类型化配置值"""
    return MockTypedConfigValue(key, type_hint, default)


__all__ = [
    'MockConfigStrategy',
    'MockLoadResult',
    'MockValidationResult',
    'MockTypedConfigValue',
    'MockTypedConfigBase',
    'MockTypedConfigSimple',
    'MockConfigManager',
    'ConfigAccessError',
    'ConfigValueError',
    'ConfigTypeError',
    'create_mock_strategy',
    'create_mock_config_manager',
    'create_mock_typed_config_value',
]
