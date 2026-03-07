
# import tomli  # Python < 3.11
# import tomli as tomllib  # 使用tomli作为tomllib的替代

import logging
import threading
import time
import os
import json
import asyncio
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from ..config_exceptions import ConfigError

# 条件导入tomllib/tomli
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        tomllib = None

try:
    import tomli
except ImportError:
    tomli = None

# 导入ConfigValidator
try:
    from ...validators.validator_composition import ConfigValidator
except ImportError:
    ConfigValidator = None
# from src.infrastructure.config.core.imports import Dict, Any, Optional, List, Callable
# from src.infrastructure.config.core.imports import Path
# from src.infrastructure.config.core.imports import json
# from src.infrastructure.config.core.imports import logging
# from src.infrastructure.config.core.imports import os
from ..interfaces.unified_interface import ConfigFormat
#!/usr/bin/env python3
"""
配置管理通用方法库

提供通用的配置操作方法，避免重复实现
"""

logger = logging.getLogger(__name__)


class ConfigCommonMethods:
    """配置管理通用方法库"""

    @staticmethod
    def validate_config_generic(
        config: Dict[str, Any],
        validation_rules: Optional[Dict[str, Any]] = None,
        custom_validators: Optional[List[Callable]] = None
    ) -> bool:
        """
        通用配置验证方法

        Args:
            config: 要验证的配置
            validation_rules: 验证规则
            custom_validators: 自定义验证器列表

        Returns:
            bool: 验证是否通过
        """
        try:
            # 基本类型验证
            if not isinstance(config, dict):
                logger.error("Config must be a dictionary")
                return False

            # 应用验证规则
            if validation_rules:
                if not ConfigCommonMethods._validate_with_rules(config, validation_rules):
                    return False

            # 应用自定义验证器
            if custom_validators:
                for validator in custom_validators:
                    try:
                        if not validator(config):
                            logger.error(f"Custom validator {validator.__name__} failed")
                            return False
                    except Exception as e:
                        logger.error(f"Custom validator {validator.__name__} error: {e}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error in generic config validation: {e}")
            return False

    @staticmethod
    def _validate_with_rules(config: Dict[str, Any], rules: Dict[str, Any]) -> bool:
        """使用规则验证配置"""
        # 实现基本的验证逻辑
        for key, rule in rules.items():
            if key in config:
                value = config[key]
                if not ConfigCommonMethods._validate_value(value, rule):
                    logger.error(f"Validation failed for key '{key}': {rule}")
                    return False
        return True

    @staticmethod
    def _validate_value(value: Any, rule: Any) -> bool:
        """验证单个值"""
        if not isinstance(rule, dict):
            return True

        # 类型验证
        if not ConfigCommonMethods._validate_type_constraint(value, rule):
            return False

        # 必需性验证
        if not ConfigCommonMethods._validate_required_constraint(value, rule):
            return False

        # 范围验证
        if not ConfigCommonMethods._validate_range_constraints(value, rule):
            return False

        return True

    @staticmethod
    def _validate_type_constraint(value: Any, rule: Dict[str, Any]) -> bool:
        """验证类型约束"""
        if 'type' not in rule:
            return True

        expected_type = rule['type']
        type_validators = {
            'string': lambda v: isinstance(v, str),
            'number': lambda v: isinstance(v, (int, float)),
            'boolean': lambda v: isinstance(v, bool),
            'list': lambda v: isinstance(v, list),
        }

        validator = type_validators.get(expected_type)
        return validator(value) if validator else True

    @staticmethod
    def _validate_required_constraint(value: Any, rule: Dict[str, Any]) -> bool:
        """验证必需性约束"""
        if rule.get('required', False) and value is None:
            return False
        return True

    @staticmethod
    def _validate_range_constraints(value: Any, rule: Dict[str, Any]) -> bool:
        """验证范围约束"""
        if not isinstance(value, (int, float)):
            return True

        if 'min' in rule and value < rule['min']:
            return False

        if 'max' in rule and value > rule['max']:
            return False

        return True

    @staticmethod
    def load_config_generic(
        source: str,
        format_hint: Optional[ConfigFormat] = None,
        encoding: str = 'utf-8',
        **kwargs
    ) -> Dict[str, Any]:
        """
        通用配置加载方法

        Args:
            source: 配置源路径
            format_hint: 格式提示
            encoding: 文件编码
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 加载的配置

        Raises:
            ConfigError: 加载失败时抛出
        """
        try:
            # 确定格式
            config_format = format_hint or ConfigCommonMethods._detect_format(source)

            # 根据格式加载
            if config_format == ConfigFormat.JSON:
                return ConfigCommonMethods._load_json(source, encoding)
            elif config_format == ConfigFormat.YAML:
                return ConfigCommonMethods._load_yaml(source, encoding)
            elif config_format == ConfigFormat.TOML:
                return ConfigCommonMethods._load_toml(source, encoding)
            else:
                raise ConfigError(f"Unsupported config format: {config_format}")

        except Exception as e:
            logger.error(f"Failed to load config from {source}: {e}")
            raise ConfigError(f"Failed to load config from {source}: {e}")

    @staticmethod
    def _detect_format(source: str) -> ConfigFormat:
        """检测配置文件格式"""
        path = Path(source)
        suffix = path.suffix.lower()

        if suffix in ['.json']:
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix in ['.toml']:
            return ConfigFormat.TOML
        else:
            # 默认使用JSON
            return ConfigFormat.JSON

    @staticmethod
    def _load_json(source: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """加载JSON配置"""
        with open(source, 'r', encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def _load_yaml(source: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """加载YAML配置"""
        try:
            import yaml
            with open(source, 'r', encoding=encoding) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ConfigError("PyYAML is required for YAML config files")

    @staticmethod
    def _load_toml(source: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """加载TOML配置"""
        try:
            with open(source, 'rb') as f:
                return tomllib.load(f)
        except ImportError:
            try:
                with open(source, 'rb') as f:
                    return tomli.load(f)
            except ImportError:
                raise ConfigError("tomli is required for TOML config files")

    @staticmethod
    def save_config_generic(
        config: Dict[str, Any],
        target: str,
        format_hint: Optional[ConfigFormat] = None,
        encoding: str = 'utf-8',
        indent: int = 2,
        **kwargs
    ) -> bool:
        """
        通用配置保存方法

        Args:
            config: 要保存的配置
            target: 目标路径
            format_hint: 格式提示
            encoding: 文件编码
            indent: 缩进空格数
            **kwargs: 额外参数

        Returns:
            bool: 是否成功
        """
        try:
            # 确定格式
            config_format = format_hint or ConfigCommonMethods._detect_format(target)

            # 确保目标目录存在
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 根据格式保存
            if config_format == ConfigFormat.JSON:
                return ConfigCommonMethods._save_json(config, target, encoding, indent)
            elif config_format == ConfigFormat.YAML:
                return ConfigCommonMethods._save_yaml(config, target, encoding, indent)
            elif config_format == ConfigFormat.TOML:
                return ConfigCommonMethods._save_toml(config, target, encoding)
            else:
                # 默认使用JSON
                return ConfigCommonMethods._save_json(config, target, encoding, indent)

        except Exception as e:
            logger.error(f"Failed to save config to {target}: {e}")
            return False

    @staticmethod
    def _save_json(config: Dict[str, Any], target: str, encoding: str = 'utf-8', indent: int = 2) -> bool:
        """保存JSON配置"""
        with open(target, 'w', encoding=encoding) as f:
            json.dump(config, f, indent=indent, ensure_ascii=False)
        return True

    @staticmethod
    def _save_yaml(config: Dict[str, Any], target: str, encoding: str = 'utf-8', indent: int = 2) -> bool:
        """保存YAML配置"""
        try:
            import yaml
            with open(target, 'w', encoding=encoding) as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=indent)
            return True
        except ImportError:
            logger.error("PyYAML is required for YAML config files")
            return False

    @staticmethod
    def _save_toml(config: Dict[str, Any], target: str, encoding: str = 'utf-8') -> bool:
        """保存TOML配置"""
        try:
            # TOML保存功能暂时不实现，返回False表示不支持
            logger.warning("TOML saving is not implemented")
            return False
        except Exception as e:
            logger.error(f"Error saving TOML config: {e}")
            return False




