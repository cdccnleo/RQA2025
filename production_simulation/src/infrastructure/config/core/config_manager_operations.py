"""
config_manager_operations 模块

提供 config_manager_operations 相关功能和接口。
"""

import os
import re
import copy

from datetime import datetime
from .config_manager_storage import UnifiedConfigManagerWithStorage
from .config_listeners import ConfigListenerManager
from .constants import (
    MAX_CONFIG_KEY_LENGTH, MIN_CONFIG_KEY_LENGTH
)
from .exceptions import (
    ConfigValidationError, ConfigKeyError, ConfigTypeError,
    handle_config_validation_exception
)
from typing import Dict, Any, Optional, List, Callable, Union
import logging
import time

logger = logging.getLogger(__name__)


class UnifiedConfigManagerWithOperations(UnifiedConfigManagerWithStorage):
    """带完整操作功能的配置管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化操作功能"""
        super().__init__(config)
        # 初始化监听器管理器
        self._listener_manager = ConfigListenerManager()
        # 如果传入了配置数据，加载到_data中
        if config:
            for key, value in config.items():
                if isinstance(value, dict):
                    self._data[key] = copy.deepcopy(value)
        # _watchers 现在通过属性访问监听器管理器的数据
        self._initialized: bool = False  # 添加初始化状态属性

    @property
    def _watchers(self) -> Dict[str, List[Callable]]:
        """获取监听器字典 (向后兼容性)"""
        # 如果有_core_manager，优先使用它的监听器管理器
        if hasattr(self, '_core_manager') and hasattr(self._core_manager, '_listener_manager'):
            return self._core_manager._listener_manager._watchers
        # 否则使用自己的监听器管理器（如果存在）
        if hasattr(self, '_listener_manager'):
            return self._listener_manager._watchers
        return {}

    @property
    def _validation_rules(self) -> Dict[str, Any]:
        """获取验证规则字典 (向后兼容性)"""
        # 避免递归，直接访问实例字典
        return self.__dict__.get('_validation_rules', {})

    @_validation_rules.setter
    def _validation_rules(self, rules: Dict[str, Any]):
        stored = rules or {}
        self.__dict__['_validation_rules'] = stored
        if hasattr(self, '_validator') and hasattr(self._validator, '_validation_rules'):
            self._validator._validation_rules = stored

    @_validation_rules.deleter
    def _validation_rules(self):
        if '_validation_rules' in self.__dict__:
            del self.__dict__['_validation_rules']
        if hasattr(self, '_validator') and hasattr(self._validator, '_validation_rules'):
            self._validator._validation_rules = {}

    @handle_config_validation_exception("config_validation")
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证配置

        Args:
            config: 要验证的配置，默认为当前配置

        Returns:
            bool: 是否有效

        Raises:
            ConfigValidationError: 当配置验证失败时抛出
        """
        config_to_validate = config or self._data

        # 空配置验证
        if not config_to_validate:
            raise ConfigValidationError("配置不能为空", config_key=None)

        # 获取验证规则
        validation_rules = self._get_validation_rules()

        # 根据是否有验证规则选择验证方法
        if validation_rules:
            return self._validate_with_rules(config_to_validate, validation_rules)
        else:
            return self._validate_basic(config_to_validate)

    def _get_validation_rules(self) -> Optional[Dict[str, Any]]:
        """获取验证规则"""
        # 优先从实例属性获取
        validation_rules = getattr(self, '_validation_rules', None)
        if validation_rules:
            return validation_rules

        # 从配置对象获取
        config_attr = getattr(self, 'config', None)
        if isinstance(config_attr, dict):
            rules = config_attr.get("validation_rules")
            if rules:
                return rules
        elif hasattr(config_attr, 'get'):
            try:
                rules = config_attr.get("validation_rules")
                if rules:
                    return rules
            except Exception:
                pass

        # 从默认配置获取
        config_settings = getattr(self, '_config_settings', None)
        if isinstance(config_settings, dict):
            rules = config_settings.get("validation_rules")
            if rules:
                return rules

        # 从数据中获取
        if hasattr(self, '_data') and self._data:
            validation_rules = self._data.get("validation_rules")
            if validation_rules:
                return validation_rules

        return None

    def _validate_basic(self, config: Dict[str, Any]) -> bool:
        """
        进行基本配置验证

        Args:
            config: 要验证的配置

        Returns:
            bool: 是否有效

        Raises:
            ConfigValidationError: 当验证失败时抛出
        """
        # 检查配置是否为字典
        if not isinstance(config, dict):
            raise ConfigTypeError(
                f"配置必须是字典类型，实际类型: {type(config).__name__}",
                expected_type="dict",
                actual_type=type(config).__name__,
                value=config
            )

        # 检查键的有效性
        for key, value in config.items():
            self._validate_config_key(key)

        return True

    def _validate_config_key(self, key: str) -> None:
        """
        验证配置键的有效性

        Args:
            key: 配置键

        Raises:
            ConfigKeyError: 当键无效时抛出
        """
        # 键必须是字符串
        if not isinstance(key, str):
            raise ConfigKeyError(
                f"配置键必须是字符串类型，实际类型: {type(key).__name__}",
                key=str(key)
            )

        # 键不能为空
        if not key:
            raise ConfigKeyError("配置键不能为空", key=key)

        # 键长度检查
        if len(key) < MIN_CONFIG_KEY_LENGTH:
            raise ConfigKeyError(
                f"配置键长度不能小于{MIN_CONFIG_KEY_LENGTH}个字符",
                key=key
            )

        if len(key) > MAX_CONFIG_KEY_LENGTH:
            raise ConfigKeyError(
                f"配置键长度不能超过{MAX_CONFIG_KEY_LENGTH}个字符",
                key=key
            )

        # 键不能包含危险字符
        dangerous_chars = ['<', '>', ';', '&']
        for char in dangerous_chars:
            if char in key:
                raise ConfigKeyError(
                    f"配置键不能包含危险字符 '{char}'",
                    key=key
                )

    @handle_config_validation_exception("rule_validation")
    def _validate_with_rules(self, config: Dict[str, Any], rules: Dict[str, Any]) -> bool:
        """
        使用规则验证配置

        Args:
            config: 配置字典
            rules: 验证规则

        Returns:
            bool: 是否有效
        """
        try:
            self._validate_rules_recursive(config, rules, "")
            return True
        except ConfigValidationError:
            return False

    def _validate_rules_recursive(self, config: Dict[str, Any], rules: Dict[str, Any], prefix: str) -> None:
        """
        递归验证规则

        Args:
            config: 配置字典
            rules: 验证规则
            prefix: 键前缀
        """
        for key, rule in rules.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # 如果规则是字典且包含字段规则，递归处理
            if isinstance(rule, dict) and self._is_section_rule(rule):
                # 这是section规则，检查section是否存在
                if key not in config:
                    # 检查是否有必需的子字段
                    has_required = self._has_required_fields(rule)
                    if has_required:
                        raise ConfigValidationError(f"必需的配置section缺失: {key}", config_key=key)
                    continue

                section_config = config[key]
                if not isinstance(section_config, dict):
                    raise ConfigValidationError(f"配置section必须是字典: {key}", config_key=key)

                # 递归验证子规则
                self._validate_rules_recursive(section_config, rule, full_key)
            else:
                # 这是字段规则
                self._validate_single_rule(config, key, rule)

    def _is_section_rule(self, rule: Dict[str, Any]) -> bool:
        """检查是否为section规则"""
        return isinstance(rule, dict) and any(
            isinstance(v, dict) and any(k in ['type', 'required', 'min', 'max'] for k in v.keys())
            for v in rule.values()
        )

    def _has_required_fields(self, rule: Dict[str, Any]) -> bool:
        """检查规则中是否有必需字段"""
        return any(
            isinstance(sub_rule, dict) and sub_rule.get('required', False)
            for sub_rule in rule.values()
        )

    def _validate_single_rule(self, config: Dict[str, Any], key: str, rule: Dict[str, Any]) -> None:
        """
        验证单个配置规则

        Args:
            config: 配置字典
            key: 配置键
            rule: 验证规则

        Raises:
            ConfigValidationError: 当验证失败时抛出
        """
        # 检查必需字段
        if key not in config:
            if rule.get("required", False):
                raise ConfigValidationError(f"必需的配置项缺失: {key}", config_key=key)
            return

        value = config[key]

        # 类型检查
        self._validate_type(key, value, rule)

        # 字符串特定检查
        if isinstance(value, str):
            self._validate_string_constraints(key, value, rule)

        # 数值范围检查
        if isinstance(value, (int, float)):
            self._validate_numeric_constraints(key, value, rule)

        # 枚举检查
        if "enum" in rule:
            self._validate_enum_constraint(key, value, rule)

        # 模式检查 (正则表达式)
        if "pattern" in rule and isinstance(value, str):
            self._validate_pattern_constraint(key, value, rule)

    def _validate_type(self, key: str, value: Any, rule: Dict[str, Any]) -> None:
        """
        验证配置项类型

        Args:
            key: 配置键
            value: 配置值
            rule: 验证规则

        Raises:
            ConfigTypeError: 当类型不匹配时抛出
        """
        expected_type = rule.get("type")
        if not expected_type:
            return

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "list": list
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            raise ConfigTypeError(
                f"配置项 '{key}' 类型错误，期望 {expected_type}，实际 {type(value).__name__}",
                expected_type=expected_type,
                actual_type=type(value).__name__,
                value=value,
                config_key=key
            )

    def _validate_string_constraints(self, key: str, value: str, rule: Dict[str, Any]) -> None:
        """
        验证字符串约束

        Args:
            key: 配置键
            value: 字符串值
            rule: 验证规则

        Raises:
            ConfigValidationError: 当验证失败时抛出
        """
        # 长度检查
        min_length = rule.get("min_length")
        if min_length is not None and len(value) < min_length:
            raise ConfigValidationError(
                f"配置项 '{key}' 值长度 {len(value)} 低于最小长度 {min_length}",
                config_key=key,
                details={"value": value, "min_length": min_length, "actual_length": len(value)}
            )

        max_length = rule.get("max_length")
        if max_length is not None and len(value) > max_length:
            raise ConfigValidationError(
                f"配置项 '{key}' 值长度 {len(value)} 超过最大长度 {max_length}",
                config_key=key,
                details={"value": value, "max_length": max_length, "actual_length": len(value)}
            )

    def _validate_numeric_constraints(self, key: str, value: Union[int, float], rule: Dict[str, Any]) -> None:
        """
        验证数值约束

        Args:
            key: 配置键
            value: 数值
            rule: 验证规则

        Raises:
            ConfigValidationError: 当验证失败时抛出
        """
        min_val = rule.get("min")
        if min_val is not None and value < min_val:
            raise ConfigValidationError(
                f"配置项 '{key}' 值 {value} 低于最小值 {min_val}",
                config_key=key,
                details={"value": value, "min": min_val}
            )

        max_val = rule.get("max")
        if max_val is not None and value > max_val:
            raise ConfigValidationError(
                f"配置项 '{key}' 值 {value} 超过最大值 {max_val}",
                config_key=key,
                details={"value": value, "max": max_val}
            )

    def _validate_enum_constraint(self, key: str, value: Any, rule: Dict[str, Any]) -> None:
        """
        验证枚举约束

        Args:
            key: 配置键
            value: 配置值
            rule: 验证规则

        Raises:
            ConfigValidationError: 当验证失败时抛出
        """
        allowed_values = rule.get("enum", [])
        if value not in allowed_values:
            raise ConfigValidationError(
                f"配置项 '{key}' 值 {value} 不在允许值范围内: {allowed_values}",
                config_key=key,
                details={"value": value, "allowed_values": allowed_values}
            )

    def _validate_pattern_constraint(self, key: str, value: str, rule: Dict[str, Any]) -> None:
        """
        验证模式约束（正则表达式）

        Args:
            key: 配置键
            value: 字符串值
            rule: 验证规则

        Raises:
            ConfigValidationError: 当验证失败时抛出
        """
        pattern = rule.get("pattern")
        if pattern and not re.match(pattern, value):
            raise ConfigValidationError(
                f"配置项 '{key}' 值 '{value}' 不匹配模式 {pattern}",
                config_key=key,
                details={"value": value, "pattern": pattern}
            )

    # ==================== 增强功能 ====================

    def get_with_fallback(self, key: str, fallback_keys: List[str], default: Any = None) -> Any:
        """
        获取配置值，支持多个后备键

        Args:
        key: 主键
        fallback_keys: 后备键列表
        default: 默认值

        Returns:
        Any: 配置值
        """
        # 首先尝试主键
        value = self.get(key)
        if value is not None:
            return value

        # 尝试后备键
        for fallback_key in fallback_keys:
            value = self.get(fallback_key)
            if value is not None:
                logger.debug(f"Using fallback key '{fallback_key}' for '{key}'")
                return value

        return default

    def set_with_validation(self, key: str, value: Any, validation_rules: Optional[Dict] = None) -> bool:
        """
        设置配置值并验证

        Args:
        key: 配置键
        value: 配置值
        validation_rules: 验证规则

        Returns:
        bool: 是否成功
        """
        try:
            config_settings = getattr(self, '_config_settings', None)
            config = config_settings if config_settings is not None else (getattr(self, 'config', {}) or {})
            # 如果启用了验证
            if config.get("validation_enabled", True):
                test_config = {key: value}

                # 优先使用传入的验证规则，然后是实例级别的验证规则，最后是配置中的验证规则
                rules = validation_rules
                if not rules:
                    rules = getattr(self, '_validation_rules', None)
                if not rules:
                    rules = config.get("validation_rules", {})

                # 如果有验证规则，进行验证
                if rules:
                    applicable_rules = {k: v for k, v in rules.items() if k == key}
                    if applicable_rules and not self._validate_with_rules(test_config, applicable_rules):
                        logger.error(f"Validation failed for key '{key}' with value {value}")
                        return False

            # 设置值
            return self.set(key, value)

        except Exception as e:
            logger.error(f"Error setting value with validation for key '{key}': {e}")
            return False

    def batch_update(self, updates: Dict[str, Any], validate: bool = True) -> Dict[str, bool]:
        """
        批量更新配置

        Args:
        updates: 更新字典
        validate: 是否验证

        Returns:
        Dict[str, bool]: 每个键的更新结果
        """
        results = {}

        try:
            config_settings = getattr(self, '_config_settings', None)
            config = config_settings if config_settings is not None else (getattr(self, 'config', {}) or {})
            for key, value in updates.items():
                if validate and config.get("validation_enabled", True):
                    # 逐个验证 - 使用实例级别的验证规则
                    test_config = {key: value}
                    instance_rules = getattr(self, '_validation_rules', None)
                    if instance_rules:
                        # 使用实例级别的验证规则
                        applicable_rules = {k: v for k, v in instance_rules.items() if k == key}
                        if applicable_rules and not self._validate_with_rules(test_config, applicable_rules):
                            results[key] = False
                            continue
                    else:
                        # 使用配置中的验证规则
                        rules = config.get("validation_rules", {})
                        applicable_rules = {k: v for k, v in rules.items() if k == key}
                        if applicable_rules and not self._validate_with_rules(test_config, applicable_rules):
                            results[key] = False
                            continue

                results[key] = self.set(key, value)

        except Exception as e:
            logger.error(f"Error during batch update: {e}")
            # 标记所有剩余的键为失败
            for key in updates:
                if key not in results:
                    results[key] = False

        return results

    def get_config_snapshot(self) -> Dict[str, Any]:
        """
        获取配置快照（包含元数据）

        Returns:
        Dict[str, Any]: 配置快照
        """
        config = getattr(self, 'config', {}) or {}
        return {
            "data": self._data.copy(),
            "metadata": {
                "timestamp": time.time(),
                "version": config.get("version", "1.0"),
                "source": config.get("config_file", "unknown"),
                "sections_count": len(self._data),
                "total_keys": sum(len(section) if isinstance(section, dict) else 1
                                  for section in self._data.values())
            }
        }

    def watch(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """
        监听配置变化

        Args:
        key: 配置键
        callback: 回调函数
        """
        self._listener_manager.add_watcher(key, callback)

    def unwatch(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """
        移除配置监听器

        Args:
        key: 配置键
        callback: 要移除的回调函数
        """
        self._listener_manager.remove_watcher(key, callback)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
        key: 配置键 (格式: section.key)
        default: 默认值

        Returns:
        Any: 配置值
        """
        # 验证键的有效性
        if key is None:
            raise TypeError("Key cannot be None")
        if not key or not isinstance(key, str):
            return default

        try:
            # 分割键，支持多级嵌套
            parts = key.split('.')
            if len(parts) < 1:
                return default

            if len(parts) == 1:
                # 单个部分，直接返回section的值
                section = parts[0]
                return self._data.get(section, default)
            else:
                # 多级嵌套
                # 获取section
                section = parts[0]
                section_data = self._data.get(section)
                if not section_data or not isinstance(section_data, dict):
                    return default

                # 递归访问嵌套键
                current_data = section_data
                for part in parts[1:]:
                    if not isinstance(current_data, dict):
                        return default
                    current_data = current_data.get(part)
                    if current_data is None:
                        return default

                return current_data

        except TypeError:
            # 重新抛出TypeError
            raise
        except Exception as e:
            logger.error(f"Error getting config value for key '{key}': {e}")
            return default

    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值

        Args:
        key: 配置键 (格式: section.key)
        value: 配置值

        Returns:
        bool: 是否成功
        """
        try:
            # 验证键参数
            self._validate_set_key_parameter(key)

            # 解析键结构
            key_parts = self._parse_key_structure(key)

            # 获取旧值并设置新值
            old_value = self._set_config_value(key_parts, value)

            # 触发变更通知
            self._notify_config_change(key, value, old_value, key_parts['section'])

            return True

        except TypeError:
            # 重新抛出TypeError
            raise
        except Exception as e:
            logger.error(f"Error setting config value for key '{key}': {e}")
            return False

    def _validate_set_key_parameter(self, key: str):
        """验证设置操作的键参数"""
        if key is None:
            raise TypeError("Key cannot be None")
        if not key or not isinstance(key, str):
            logger.error("Invalid config key")
            raise ValueError("Invalid config key")

        # 检查键的格式
        if key == "." or key.startswith(".") or key.endswith(".") or ".." in key:
            logger.error(f"Invalid config key format: {key}")
            raise ValueError(f"Invalid config key format: {key}")

        # 检查键长度
        if len(key) > 100:  # 与validate方法保持一致
            logger.error(f"Config key too long: {key}")
            raise ValueError(f"Config key too long: {key}")

        # 检查危险字符
        dangerous_chars = ['<', '>', ';', ' ', '/', '\\', ':', '@', '#']
        if any(char in key for char in dangerous_chars):
            logger.error(f"Config key contains dangerous characters: {key}")
            raise ValueError(f"Config key contains dangerous characters: {key}")

    def _parse_key_structure(self, key: str) -> Dict[str, Any]:
        """解析配置键的结构"""
        parts = key.split('.')
        if len(parts) < 1:
            logger.error(f"Invalid config key format: {key}")
            raise ValueError(f"Invalid config key format: {key}")

        return {
            'parts': parts,
            'section': parts[0],
            'is_nested': len(parts) > 1,
            'final_key': parts[-1] if len(parts) > 1 else None
        }

    def _set_config_value(self, key_structure: Dict[str, Any], value: Any) -> Any:
        """设置配置值并返回旧值"""
        parts = key_structure['parts']
        section = key_structure['section']

        if not key_structure['is_nested']:
            # 单个部分，直接设置section的值
            old_value = self._data.get(section)
            self._data[section] = value
            return old_value
        else:
            # 多级嵌套
            return self._set_nested_config_value(section, parts[1:], value)

    def _set_nested_config_value(self, section: str, nested_parts: list, value: Any) -> Any:
        """设置嵌套配置值"""
        if section not in self._data:
            self._data[section] = {}

        # 递归设置嵌套值
        current = self._data[section]
        for i, part in enumerate(nested_parts[:-1]):  # 除了最后一个部分
            if not isinstance(current, dict):
                error_path = f"{section}.{'.'.join(nested_parts[:i+1])}"
                logger.error(f"Cannot set nested key: '{error_path}' is not a dictionary")
                raise ValueError(f"Cannot set nested key: '{error_path}' is not a dictionary")

            if part not in current:
                current[part] = {}
            current = current[part]

        # 设置最终值
        final_key = nested_parts[-1]
        if not isinstance(current, dict):
            logger.error(f"Cannot set nested key: parent is not a dictionary")
            raise ValueError(f"Cannot set nested key: parent is not a dictionary")

        old_value = current.get(final_key)
        current[final_key] = value
        return old_value

    def _notify_config_change(self, key: str, value: Any, old_value: Any, section: str):
        """通知配置变更"""
        # 只有当值发生变化时才触发监听器
        if old_value != value:
            # 使用监听器管理器触发监听器
            self._listener_manager.trigger_listeners(key, value, old_value)

    def update(self, config: Dict[str, Any]) -> None:
        """
        更新配置

        Args:
        config: 配置字典
        """
        try:
            if config is None:
                raise ValueError("Config cannot be None")
            if not isinstance(config, dict):
                raise ValueError("Config must be a dictionary")

            # 检查是否是嵌套配置（包含字典值）
            has_nested_dicts = any(isinstance(value, dict) for value in config.values())

            if has_nested_dicts:
                # 使用深度合并来处理嵌套配置
                self._data = self._deep_merge_dict(self._data, config)
            else:
                # 对于简单配置，使用原来的逻辑
                for key, value in config.items():
                    self.set(key, value)
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise ValueError("Failed to update config") from e

    def reload(self) -> None:
        """
        重新加载配置
        """
        try:
            self.reload_config()
        except Exception as e:
            logger.error(f"Error reloading config: {e}")

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        验证配置

        Args:
        config: 配置字典

        Returns:
        bool: 是否有效
        """
        # 验证配置字典
        if not isinstance(config, dict):
            return False

        # 检查键的有效性
        for key in config.keys():
            if not isinstance(key, str):
                return False
            if not key:  # 空键
                return False
            if len(key) > 100:  # 键长度限制
                return False
            if '<' in key or '>' in key or ';' in key or '&' in key:  # 危险字符检查
                return False

        return True

    def delete(self, section: str, key: str) -> bool:
        """
        删除配置项

        Args:
        section: 配置section名称
        key: 配置键名

        Returns:
        bool: 是否删除成功
        """
        try:
            if section in self._data and key in self._data[section]:
                del self._data[section][key]
                # 通知监听器
                self._notify_watchers(f"{section}.{key}", None)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting config {section}.{key}: {e}")
            return False

    def delete_section(self, section: str) -> bool:
        """
        删除整个配置section

        Args:
        section: 配置section名称

        Returns:
        bool: 是否删除成功
        """
        try:
            if section in self._data:
                del self._data[section]
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting section {section}: {e}")
            return False

    def clear_all(self) -> bool:
        """
        清空所有配置
        """
        try:
            self._data.clear()
            # 通知所有监听器
            for key in list(self._watchers.keys()):
                self._notify_watchers(key, None)
            return True
        except Exception as e:
            logger.error(f"Error clearing all config: {e}")
            return False

    def has_section(self, section: str) -> bool:
        """
        检查section是否存在

        Args:
        section: 配置section名称

        Returns:
        bool: 是否存在
        """
        return section in self._data

    def get_sections(self) -> list:
        """
        获取所有sections

        Returns:
        list: 所有section名称列表
        """
        return list(self._data.keys())

    def get_all_sections(self) -> List[str]:
        """
        获取所有配置节名称列表

        Returns:
        List[str]: 配置节名称列表
        """
        return list(self._data.keys())

    def set_section(self, section: str, config: Dict[str, Any]) -> bool:
        """
        设置完整section

        Args:
        section: 配置section名称
        config: 配置字典

        Returns:
        bool: 是否设置成功
        """
        try:
            if not isinstance(config, dict):
                return False
            self._data[section] = config.copy()
            return True
        except Exception as e:
            logger.error(f"Error setting section {section}: {e}")
            return False

    def load_from_file(self, file_path: str) -> bool:
        """
        从文件加载配置

        Args:
        file_path: 配置文件路径

        Returns:
        bool: 是否加载成功
        """
        return self.load_config(file_path)

    def save_to_file(self, file_path: str) -> bool:
        """
        保存配置到文件

        Args:
        file_path: 保存文件路径

        Returns:
        bool: 是否保存成功
        """
        return self.save_config(file_path)

    def backup_config(self, backup_dir: str) -> bool:
        """
        备份配置

        Args:
        backup_dir: 备份目录

        Returns:
        bool: 是否备份成功
        """
        try:
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"config_backup_{timestamp}.json")

            return self.save_to_file(backup_file)
        except Exception as e:
            logger.error(f"Error backing up config: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要

        Returns:
        Dict[str, Any]: 配置摘要信息
        """
        try:
            # 计算总键数
            total_keys = 0
            for section_data in self._data.values():
                if isinstance(section_data, dict):
                    total_keys += len(section_data)
                else:
                    total_keys += 1  # 非字典section算作1个key

            # 构建sections信息
            sections_info = {}
            for section_name, section_data in self._data.items():
                if isinstance(section_data, dict):
                    sections_info[section_name] = {
                        "keys_count": len(section_data),
                        "keys": list(section_data.keys())
                    }
                else:
                    sections_info[section_name] = {
                        "keys_count": 1,
                        "keys": [str(section_data)[:20]]  # 显示值的字符串表示（前20个字符）
                    }

            return {
                "total_sections": len(self._data),
                "total_keys": total_keys,
                "sections": sections_info
            }
        except Exception as e:
            logger.error(f"Error getting config summary: {e}")
            return {}

    def set_validation_rules(self, rules: Dict[str, Any]) -> bool:
        """
        设置验证规则

        Args:
        rules: 验证规则字典

        Returns:
        bool: 是否设置成功
        """
        try:
            # 直接设置实例字典中的验证规则
            self.__dict__['_validation_rules'] = rules
            return True
        except Exception as e:
            logger.error(f"Error setting validation rules: {e}")
            return False

    def initialize(self) -> bool:
        """
        初始化配置管理器

        Returns:
        bool: 是否初始化成功
        """
        try:
            # 标记为已初始化
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing config manager: {e}")
            return False

    def _notify_watchers(self, key: str, value: Any) -> None:
        """
        通知监听器配置变化

        Args:
        key: 配置键
        value: 新值
        """
        if key in self._watchers:
            for callback in self._watchers[key]:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.error(f"Error in watcher callback for {key}: {e}")

    def _deep_merge_dict(self, target: Dict[str, Any], source: Dict[str, Any], override: bool = True) -> Dict[str, Any]:
        """
        深度合并字典

        Args:
            target: 目标字典
            source: 源字典
            override: 是否覆盖已存在的键

        Returns:
            Dict[str, Any]: 合并后的字典
        """
        result = target.copy()

        for key, value in source.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self._deep_merge_dict(result[key], value, override)
            elif override or key not in result:
                # 覆盖模式或键不存在时才设置新值
                result[key] = value

        return result

    def merge_config(self, config: Dict[str, Any], section: Optional[str] = None, override: bool = True) -> bool:
        """
        合并配置

        Args:
            config: 要合并的配置
            section: 目标节（可选）
            override: 是否覆盖已存在的键

        Returns:
            bool: 是否成功
        """
        try:
            if not isinstance(config, dict):
                return False

            if section:
                if section not in self._data:
                    self._data[section] = {}
                if isinstance(self._data[section], dict) and isinstance(config, dict):
                    self._data[section] = self._deep_merge_dict(
                        self._data[section], config, override)
                else:
                    if override:
                        self._data[section] = config
                    # 非覆盖模式下，如果section已存在且不是字典，则不覆盖
            else:
                # 对顶级配置进行深度合并
                self._data = self._deep_merge_dict(self._data, config, override)
            return True
        except Exception as e:
            logger.error(f"Error merging config: {e}")
            return False

    def restore_from_backup(self, backup_file: str) -> bool:
        """
        从备份恢复配置

        Args:
        backup_file: 备份文件路径

        Returns:
        bool: 是否恢复成功
        """
        try:
            return self.load_from_file(backup_file)
        except Exception as e:
            logger.error(f"Error restoring config from backup {backup_file}: {e}")
            return False




