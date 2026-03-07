
import yaml
import copy
import shutil
import yaml

from .common_methods import ConfigCommonMethods
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import logging
import os
#!/usr/bin/env python3
"""
配置管理器存储功能 (拆分自unified_manager.py)

包含存储相关的所有方法：加载、保存、导出、导入等
"""

logger = logging.getLogger(__name__)

_DELETED = object()


class _ConfigView(dict):
    """展示合并后的配置视图，同时保留用户配置用于比较。"""

    def __init__(self, overrides: Dict[str, Any], defaults: Dict[str, Any]):
        self._overrides = overrides
        self._defaults = defaults
        super().__init__()
        self._sync()

    def _sync(self) -> None:
        merged = dict(self._defaults)
        for key, value in self._overrides.items():
            if value is _DELETED:
                merged.pop(key, None)
            else:
                merged[key] = value
        super().clear()
        super().update(merged)

    def __setitem__(self, key: str, value: Any) -> None:
        self._overrides[key] = value
        self._defaults[key] = value
        self._sync()

    def __delitem__(self, key: str) -> None:
        self._overrides[key] = _DELETED
        self._sync()

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            self.__setitem__(key, value)

    def pop(self, key: str, default: Any = None) -> Any:
        value = self.get(key, default)
        if key in self:
            self.__delitem__(key)
        return value

    def clear(self) -> None:
        for key in list(self.keys()):
            self.__delitem__(key)

    def copy(self) -> Dict[str, Any]:
        return dict(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            other_dict = dict(other)
            if other_dict == dict(self):
                return True
            filtered_overrides = {
                k: v for k, v in self._overrides.items() if v is not _DELETED
            }
            if other_dict == filtered_overrides:
                return True
            return False
        return super().__eq__(other)


class UnifiedConfigManagerWithStorage:
    """带存储功能的配置管理器基类"""

    DEFAULT_CONFIG_SETTINGS = {
        "auto_reload": True,
        "validation_enabled": True,
        "encryption_enabled": False,
        "backup_enabled": True,
        "max_backup_files": 3,
        "config_file": "config.json",
        "cache": {"ttl": 300},
        "database": {"host": "localhost", "port": 5432},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化存储功能基类"""
        self._user_config: Dict[str, Any] = {}
        self._config_settings: Dict[str, Any] = {}
        self._config_view: Dict[str, Any] = {}
        self.config = copy.deepcopy(config) if config else {}
        self._data: Dict[str, Any] = {}
        self._initialized: bool = False

    def _reset_explicit_keys(self):
        core_manager = getattr(self, '_core_manager', None)
        explicit_keys = getattr(core_manager, '_explicit_keys', None)
        if explicit_keys is not None:
            explicit_keys.clear()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config_view

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        if isinstance(value, _ConfigView):
            value = dict(value)
        if isinstance(value, dict):
            self._user_config = copy.deepcopy(value)
        else:
            self._user_config = {}
        self._config_settings = self._merge_defaults(
            copy.deepcopy(self.DEFAULT_CONFIG_SETTINGS),
            self._user_config
        )
        self._config_view = _ConfigView(self._user_config, self._config_settings)

    def _merge_defaults(self, defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并默认配置与用户配置（仅填充缺失项）。"""
        result = defaults
        for key, value in overrides.items():
            if value is _DELETED:
                result.pop(key, None)
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_defaults(result[key], value)
            else:
                result[key] = value
        return result

    def _get_setting(self, key: str, default: Any = None, use_defaults: bool = True) -> Any:
        """获取配置项，优先使用用户配置。"""
        if key in self._user_config:
            value = self._user_config.get(key)
            if value is _DELETED:
                return default
            return value
        if use_defaults:
            return self._config_settings.get(key, default)
        return default

    def get_section(self, section_name: str) -> Optional[Dict[str, Any]]:
        """
        获取配置节

        Args:
        section_name: 节名称

        Returns:
        Optional[Dict[str, Any]]: 配置节数据的副本
        """
        section_data = self._data.get(section_name)
        if section_data and isinstance(section_data, dict):
            return copy.deepcopy(section_data)
        return section_data

    def load_config(self, config_source: Optional[Any] = None) -> bool:
        """
        加载配置

        Args:
        config_source: 配置源（可以是文件路径或配置字典）

        Returns:
        bool: 是否成功
        """
        try:
            # 如果config_source是字典，直接合并到_data中
            if isinstance(config_source, dict):
                self._data.clear()
                self._data.update(config_source)
                self._reset_explicit_keys()
                logger.info("Config loaded successfully from dictionary")
                return True

            # 如果config_source是字符串或None，作为文件路径处理
            config_path = config_source
            if config_path is None:
                config_path = self._get_setting("config_file", "config.json")

            # 确保config_path不是None
            if not config_path:
                config_path = "config.json"

            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}")
                return False

            # 使用通用方法加载配置
            loaded_data = ConfigCommonMethods.load_config_generic(config_path)

            # 合并配置
            if isinstance(loaded_data, dict):
                self._data.clear()
                self._data.update(loaded_data)
                self._reset_explicit_keys()
                logger.info(f"Config loaded successfully from: {config_path}")
                return True
            else:
                logger.error(f"Invalid config file format: {config_path}")
                return False

        except Exception as e:
            logger.error(f"Error loading config from {config_source}: {e}")
            return False

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        保存配置文件

        Args:
        config_path: 配置文件路径

        Returns:
        bool: 是否成功
        """
        try:
            # 确保config_path不是None
            if config_path is None:
                config_path = self._get_setting("config_file", "config.json")

            # 再次检查确保不是None
            if not config_path:
                config_path = "config.json"

            # 创建备份
            if self._get_setting("backup_enabled", True) and os.path.exists(config_path):
                self._create_backup(config_path)

            # 使用通用方法保存配置
            success = ConfigCommonMethods.save_config_generic(
                config=self._data,
                target=config_path,
                indent=2
            )

            if success:
                logger.info(f"Config saved successfully to: {config_path}")

            return success

        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            return False

    def get_all_sections(self) -> List[str]:
        """
        获取所有配置节名称

        Returns:
        List[str]: 所有配置节名称列表
        """
        return list(self._data.keys())

    def reload_config(self) -> bool:
        """
        重新加载配置

        Returns:
        bool: 是否成功
        """
        try:
            config_attr = getattr(self, 'config', None)
            if isinstance(config_attr, dict) and 'config_file' in config_attr:
                config_path = config_attr.get('config_file')
            else:
                user_value = self._user_config.get("config_file")
                if user_value is _DELETED:
                    config_path = None
                else:
                    config_path = self._get_setting("config_file", use_defaults=False)
                    if not config_path and isinstance(self._config_settings, dict):
                        config_path = self._config_settings.get("config_file")

            if not config_path:
                logger.warning("No config file configured for reload")
                return False

            return self.load_config(config_path)
        except Exception as e:
            logger.error(f"Error reloading config: {e}")
            return False

    def merge_config(self, config: Dict[str, Any], section: Optional[str] = None) -> bool:
        """
        合并配置

        Args:
        config: 要合并的配置
        section: 目标节（可选）

        Returns:
        bool: 是否合并成功
        """
        try:
            if section:
                if section not in self._data:
                    self._data[section] = {}
                self._data[section].update(config)
            else:
                self._data.update(config)

            logger.info(f"Config merged successfully, section: {section}")
            return True
        except Exception as e:
            logger.error(f"Error merging config: {e}")
            return False

    def export_config(self, format: str = "json", file_path: Optional[str] = None):
        """
        导出配置

        Args:
        format: 导出格式 ("json", "yaml")
        file_path: 导出文件路径

        Returns:
        如果file_path为None，返回配置字符串；否则返回bool表示是否成功
        """
        try:
            if file_path is None:
                # 返回配置字符串
                if format.lower() == "json":
                    return json.dumps(self._data, indent=2, ensure_ascii=False)
                elif format.lower() == "yaml":
                    try:
                        return yaml.dump(self._data, default_flow_style=False, allow_unicode=True)
                    except ImportError:
                        raise ImportError("PyYAML is required for YAML export")
                else:
                    # 对于不支持的格式，返回字符串表示
                    return str(self._data)
            else:
                # 保存到文件
                if format.lower() == "json":
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self._data, f, indent=2, ensure_ascii=False)
                elif format.lower() == "yaml":
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True)
                    except ImportError:
                        raise ImportError("PyYAML is required for YAML export")
                else:
                    raise ValueError(f"Unsupported export format: {format}")

                logger.info(f"Config exported successfully to: {file_path}")
                return True

        except Exception as e:
            logger.error(f"Error exporting config: {e}")
            if file_path is None:
                # 对于不支持的格式，返回字符串表示
                return str(self._data)
            else:
                return False

    def import_config(self, file_path: str, merge: bool = True) -> bool:
        """
        导入配置

        Args:
        file_path: 导入文件路径
        merge: 是否合并（True）或替换（False）

        Returns:
        bool: 是否成功
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Import file not found: {file_path}")
                return False

            # 加载文件
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_data = json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_data = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported import file format: {file_path}")
                return False

            if not isinstance(imported_data, dict):
                logger.error(f"Invalid import file format: {file_path}")
                return False

            # 合并或替换
            if merge:
                self._data.update(imported_data)
            else:
                self._data = imported_data

            logger.info(f"Config imported successfully from: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing config from {file_path}: {e}")
            return False

    def _create_backup(self, config_path: str) -> bool:
        """
        创建配置文件备份

        Args:
        config_path: 配置文件路径

        Returns:
        bool: 是否成功
        """
        try:
            if not os.path.exists(config_path):
                return True

            # 确定备份目录
            backup_dir = os.path.join(os.path.dirname(config_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)

            # 生成备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"config_backup_{timestamp}.json"
            backup_path = os.path.join(backup_dir, backup_filename)

            # 复制文件
            shutil.copy2(config_path, backup_path)

            # 清理旧备份
            self._cleanup_old_backups(backup_dir)

            logger.info(f"Backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def _cleanup_old_backups(self, backup_dir: str) -> None:
        """
        清理旧备份文件

        Args:
        backup_dir: 备份目录
        """
        try:
            max_backups = self._get_setting("max_backup_files", 5)
            if max_backups <= 0:
                return

            # 获取所有备份文件
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.startswith("config_backup_") and file.endswith(".json"):
                    file_path = os.path.join(backup_dir, file)
                    mtime = os.path.getmtime(file_path)
                    backup_files.append((file_path, mtime))

            # 按修改时间排序
            backup_files.sort(key=lambda x: x[1], reverse=True)

            # 删除多余的备份
            for file_path, _ in backup_files[max_backups:]:
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed old backup: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old backup {file_path}: {e}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")




