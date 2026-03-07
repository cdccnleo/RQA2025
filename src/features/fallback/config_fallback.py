#!/usr/bin/env python3
"""
配置管理降级服务

在基础设施层配置管理不可用时提供降级的配置管理功能
"""

import json
from typing import Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FallbackConfigManager:

    """
    降级配置管理器

    提供基本的配置管理功能，当基础设施层配置管理不可用时使用
    """

    def __init__(self, config_file: str = "features_config.json"):
        """
        初始化降级配置管理器

        Args:
            config_file: 配置文件名
        """
        self.config_file = Path("config") / config_file
        self._config: Dict[str, Any] = {}
        self._defaults = self._get_defaults()

        # 确保配置目录存在
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self._load_config()

    def _get_defaults(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'features': {
                'enable_feature_selection': False,
                'enable_standardization': True,
                'enable_caching': True,
                'cache_ttl': 3600,
                'max_workers': 4,
                'chunk_size': 1000
            },
            'technical_indicators': [
                'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stoch'
            ],
            'sentiment_analysis': {
                'enabled': False,
                'model_path': 'models / sentiment',
                'confidence_threshold': 0.6
            },
            'performance': {
                'enable_monitoring': False,
                'metrics_interval': 60,
                'alert_threshold': 0.8
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf - 8') as f:
                    loaded_config = json.load(f)
                    # 合并默认配置和加载的配置
                    self._config = self._merge_configs(self._defaults, loaded_config)
                logger.info(f"配置已从 {self.config_file} 加载")
            else:
                # 使用默认配置
                self._config = self._defaults.copy()
                logger.info("使用默认配置")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            self._config = self._defaults.copy()

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置字典"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点分隔的嵌套键，如 'features.enable_caching'

            default: 默认值

        Returns:
            配置值
        """
        try:
            keys = key.split('.')
            value = self._config

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value
        except Exception as e:
            logger.warning(f"获取配置 {key} 失败: {e}")
            return default

    def set_config(self, key: str, value: Any) -> bool:
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值

        Returns:
            设置是否成功
        """
        try:
            keys = key.split('.')
            config = self._config

            # 导航到最后一个键的父级
            for k in keys[:-1]:
                if k not in config or not isinstance(config[k], dict):
                    config[k] = {}
                config = config[k]

            # 设置值
            config[keys[-1]] = value

            # 保存配置
            self._save_config()

            logger.info(f"配置 {key} 已设置为 {value}")
            return True

        except Exception as e:
            logger.error(f"设置配置 {key} 失败: {e}")
            return False

    def has_config(self, key: str) -> bool:
        """
        检查配置是否存在

        Args:
            key: 配置键

        Returns:
            配置是否存在
        """
        return self.get_config(key) is not None

    def get_all_configs(self) -> Dict[str, Any]:
        """
        获取所有配置

        Returns:
            所有配置的副本
        """
        return self._config.copy()

    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf - 8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.debug(f"配置已保存到 {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def reload_config(self) -> bool:
        """
        重新加载配置

        Returns:
            重新加载是否成功
        """
        try:
            self._load_config()
            logger.info("配置重新加载完成")
            return True
        except Exception as e:
            logger.error(f"重新加载配置失败: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """
        重置为默认配置

        Returns:
            重置是否成功
        """
        try:
            self._config = self._defaults.copy()
            self._save_config()
            logger.info("配置已重置为默认值")
            return True
        except Exception as e:
            logger.error(f"重置配置失败: {e}")
            return False

    def get_config_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节

        Args:
            section: 配置节名称

        Returns:
            配置节字典
        """
        section_config = self.get_config(section, {})
        if isinstance(section_config, dict):
            return section_config.copy()
        return {}

    def update_config_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        更新配置节

        Args:
            section: 配置节名称
            updates: 更新内容

        Returns:
            更新是否成功
        """
        try:
            current_section = self.get_config_section(section)
            updated_section = self._merge_configs(current_section, updates)

            return self.set_config(section, updated_section)

        except Exception as e:
            logger.error(f"更新配置节 {section} 失败: {e}")
            return False

    def validate_config(self) -> List[str]:
        """
        验证配置有效性

        Returns:
            验证错误列表
        """
        errors = []

        # 检查必需的配置项
        required_keys = [
            'features.enable_standardization',
            'features.enable_caching',
            'technical_indicators'
        ]

        for key in required_keys:
            if not self.has_config(key):
                errors.append(f"缺少必需配置: {key}")

        # 检查配置值类型
        if self.has_config('features.cache_ttl'):
            ttl = self.get_config('features.cache_ttl')
            if not isinstance(ttl, (int, float)) or ttl <= 0:
                errors.append("features.cache_ttl 必须是正数")

        if self.has_config('features.max_workers'):
            workers = self.get_config('features.max_workers')
            if not isinstance(workers, int) or workers <= 0:
                errors.append("features.max_workers 必须是正整数")

        return errors
