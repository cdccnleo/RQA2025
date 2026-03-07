#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入验证机制完善工具
为系统关键函数添加输入验证和边界检查
"""

import re
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import inspect


class InputValidationEnhancer:
    """输入验证增强器"""

    def __init__(self):
        self.validation_rules = self._define_validation_rules()

    def _define_validation_rules(self) -> Dict[str, Dict]:
        """定义验证规则"""
        return {
            'numeric': {
                'types': [int, float],
                'validators': [self._validate_numeric_range],
                'default_min': 0,
                'default_max': float('inf')
            },
            'string': {
                'types': [str],
                'validators': [self._validate_string_length, self._validate_string_format],
                'default_min_length': 1,
                'default_max_length': 1000
            },
            'list': {
                'types': [list],
                'validators': [self._validate_list_length],
                'default_min_length': 0,
                'default_max_length': 10000
            },
            'dict': {
                'types': [dict],
                'validators': [self._validate_dict_keys],
                'required_keys': []
            },
            'config': {
                'types': [dict],
                'validators': [self._validate_config_structure],
                'required_fields': ['host', 'port', 'timeout']
            }
        }

    def add_input_validation(self, func: Callable, param_rules: Dict[str, Dict]) -> Callable:
        """为函数添加输入验证"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 验证参数
            for param_name, param_value in bound_args.arguments.items():
                if param_name in param_rules:
                    rule = param_rules[param_name]
                    self._validate_parameter(param_name, param_value, rule)

            # 调用原函数
            return func(*args, **kwargs)

        return wrapper

    def _validate_parameter(self, param_name: str, value: Any, rule: Dict):
        """验证单个参数"""
        # 检查是否为None（如果不允许）
        if 'allow_none' in rule and not rule['allow_none'] and value is None:
            raise ValueError(f"参数 {param_name} 不能为None")

        if value is None:
            return  # 如果允许None，直接返回

        # 检查类型
        if 'types' in rule:
            if not isinstance(value, tuple(rule['types'])):
                expected_types = [t.__name__ for t in rule['types']]
                raise TypeError(
                    f"参数 {param_name} 类型错误，期望: {expected_types}，实际: {type(value).__name__}")

        # 应用验证器
        if 'validators' in rule:
            for validator in rule['validators']:
                validator(param_name, value, rule)

    def _validate_numeric_range(self, param_name: str, value: Any, rule: Dict):
        """验证数值范围"""
        if isinstance(value, (int, float)):
            min_val = rule.get('min', rule.get('default_min', float('-inf')))
            max_val = rule.get('max', rule.get('default_max', float('inf')))

            if not (min_val <= value <= max_val):
                raise ValueError(f"参数 {param_name} 值 {value} 超出范围 [{min_val}, {max_val}]")

    def _validate_string_length(self, param_name: str, value: str, rule: Dict):
        """验证字符串长度"""
        if isinstance(value, str):
            min_len = rule.get('min_length', rule.get('default_min_length', 0))
            max_len = rule.get('max_length', rule.get('default_max_length', 1000))

            if not (min_len <= len(value) <= max_len):
                raise ValueError(f"参数 {param_name} 长度 {len(value)} 超出范围 [{min_len}, {max_len}]")

    def _validate_string_format(self, param_name: str, value: str, rule: Dict):
        """验证字符串格式"""
        if isinstance(value, str) and 'pattern' in rule:
            if not re.match(rule['pattern'], value):
                raise ValueError(f"参数 {param_name} 格式不正确")

    def _validate_list_length(self, param_name: str, value: list, rule: Dict):
        """验证列表长度"""
        if isinstance(value, list):
            min_len = rule.get('min_length', rule.get('default_min_length', 0))
            max_len = rule.get('max_length', rule.get('default_max_length', 10000))

            if not (min_len <= len(value) <= max_len):
                raise ValueError(f"参数 {param_name} 长度 {len(value)} 超出范围 [{min_len}, {max_len}]")

    def _validate_dict_keys(self, param_name: str, value: dict, rule: Dict):
        """验证字典键"""
        if isinstance(value, dict):
            required_keys = rule.get('required_keys', [])
            for key in required_keys:
                if key not in value:
                    raise ValueError(f"参数 {param_name} 缺少必需键: {key}")

    def _validate_config_structure(self, param_name: str, value: dict, rule: Dict):
        """验证配置结构"""
        if isinstance(value, dict):
            required_fields = rule.get('required_fields', [])
            for field in required_fields:
                if field not in value:
                    raise ValueError(f"配置参数 {param_name} 缺少必需字段: {field}")

                # 验证字段值
                field_value = value[field]
                if field == 'port' and isinstance(field_value, int):
                    if not (1 <= field_value <= 65535):
                        raise ValueError(f"端口号 {field_value} 无效")
                elif field == 'timeout' and isinstance(field_value, (int, float)):
                    if field_value <= 0:
                        raise ValueError(f"超时时间 {field_value} 必须大于0")


class BoundaryConditionValidator:
    """边界条件验证器"""

    def __init__(self):
        self.enhancer = InputValidationEnhancer()

    def add_validation_to_function(self, func: Callable, param_rules: Dict[str, Dict]) -> Callable:
        """为函数添加验证"""
        return self.enhancer.add_input_validation(func, param_rules)

    def create_config_validator(self, required_fields: List[str]) -> Callable:
        """创建配置验证器"""
        def config_validator(config: Dict[str, Any]) -> Dict[str, Any]:
            """验证配置参数"""
            if not isinstance(config, dict):
                raise TypeError("配置必须是字典类型")

            validated_config = {}

            for field in required_fields:
                if field not in config:
                    raise ValueError(f"配置缺少必需字段: {field}")
                validated_config[field] = config[field]

            # 验证端口号
            if 'port' in validated_config:
                port = validated_config['port']
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    raise ValueError(f"端口号 {port} 无效")

            # 验证超时时间
            if 'timeout' in validated_config:
                timeout = validated_config['timeout']
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    raise ValueError(f"超时时间 {timeout} 无效")

            # 验证主机地址
            if 'host' in validated_config:
                host = validated_config['host']
                if not isinstance(host, str) or not host.strip():
                    raise ValueError("主机地址不能为空")

            return validated_config

        return config_validator

    def create_numeric_validator(self, min_val: float = float('-inf'),
                                 max_val: float = float('inf'),
                                 allow_zero: bool = True) -> Callable:
        """创建数值验证器"""
        def numeric_validator(value: Any) -> float:
            """验证数值"""
            if not isinstance(value, (int, float)):
                raise TypeError(f"期望数值类型，实际: {type(value)}")

            if not allow_zero and value == 0:
                raise ValueError("值不能为零")

            if not (min_val <= value <= max_val):
                raise ValueError(f"值 {value} 超出范围 [{min_val}, {max_val}]")

            return float(value)

        return numeric_validator

    def create_string_validator(self, min_length: int = 0,
                                max_length: int = 1000,
                                pattern: Optional[str] = None) -> Callable:
        """创建字符串验证器"""
        def string_validator(value: Any) -> str:
            """验证字符串"""
            if not isinstance(value, str):
                raise TypeError(f"期望字符串类型，实际: {type(value)}")

            if not (min_length <= len(value) <= max_length):
                raise ValueError(f"字符串长度 {len(value)} 超出范围 [{min_length}, {max_length}]")

            if pattern and not re.match(pattern, value):
                raise ValueError("字符串格式不正确")

            return value.strip()

        return string_validator

    def create_list_validator(self, min_length: int = 0,
                              max_length: int = 10000,
                              item_validator: Optional[Callable] = None) -> Callable:
        """创建列表验证器"""
        def list_validator(value: Any) -> list:
            """验证列表"""
            if not isinstance(value, list):
                raise TypeError(f"期望列表类型，实际: {type(value)}")

            if not (min_length <= len(value) <= max_length):
                raise ValueError(f"列表长度 {len(value)} 超出范围 [{min_length}, {max_length}]")

            # 验证每个元素
            if item_validator:
                validated_items = []
                for item in value:
                    validated_items.append(item_validator(item))
                return validated_items

            return value

        return list_validator

# 实用工具函数


def safe_divide(dividend: float, divisor: float, default: float = 0.0) -> float:
    """安全的除法运算"""
    if not isinstance(divisor, (int, float)) or divisor == 0:
        return default
    return dividend / divisor


def safe_get_config(config: Dict, key: str, default: Any = None) -> Any:
    """安全的配置获取"""
    return config.get(key, default) if isinstance(config, dict) else default


def safe_list_access(lst: list, index: int, default: Any = None) -> Any:
    """安全的列表访问"""
    if isinstance(lst, list) and 0 <= index < len(lst):
        return lst[index]
    return default


def safe_dict_access(dct: dict, key: str, default: Any = None) -> Any:
    """安全的字典访问"""
    return dct.get(key, default) if isinstance(dct, dict) else default


def validate_port(port: Any) -> int:
    """验证端口号"""
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"无效端口号: {port}")
    return port


def validate_timeout(timeout: Any) -> float:
    """验证超时时间"""
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"无效超时时间: {timeout}")
    return float(timeout)


def validate_host(host: Any) -> str:
    """验证主机地址"""
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"无效主机地址: {host}")
    return host.strip()

# 装饰器


def validate_config(required_fields: List[str]):
    """配置验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取config参数
            config = kwargs.get('config')
            if config is None and len(args) > 0:
                # 假设config是第一个参数
                config = args[0]

            if config is not None:
                for field in required_fields:
                    if field not in config:
                        raise ValueError(f"配置缺少必需字段: {field}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_numeric_range(min_val: float = float('-inf'),
                           max_val: float = float('inf'),
                           allow_zero: bool = True):
    """数值范围验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 验证所有数值参数
            for arg_name, arg_value in kwargs.items():
                if isinstance(arg_value, (int, float)):
                    if not allow_zero and arg_value == 0:
                        raise ValueError(f"参数 {arg_name} 不能为零")
                    if not (min_val <= arg_value <= max_val):
                        raise ValueError(f"参数 {arg_name} 值 {arg_value} 超出范围 [{min_val}, {max_val}]")

            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试验证器
    validator = BoundaryConditionValidator()

    # 测试数值验证器
    numeric_validator = validator.create_numeric_validator(min_val=0, max_val=100)

    try:
        result = numeric_validator(50)
        print(f"数值验证通过: {result}")
    except ValueError as e:
        print(f"数值验证失败: {e}")

    try:
        result = numeric_validator(-5)
        print(f"数值验证通过: {result}")
    except ValueError as e:
        print(f"数值验证失败: {e}")

    # 测试字符串验证器
    string_validator = validator.create_string_validator(min_length=1, max_length=10)

    try:
        result = string_validator("hello")
        print(f"字符串验证通过: {result}")
    except ValueError as e:
        print(f"字符串验证失败: {e}")

    # 测试配置验证器
    config_validator = validator.create_config_validator(['host', 'port', 'timeout'])

    test_config = {
        'host': 'localhost',
        'port': 8080,
        'timeout': 30
    }

    try:
        result = config_validator(test_config)
        print(f"配置验证通过: {result}")
    except ValueError as e:
        print(f"配置验证失败: {e}")
