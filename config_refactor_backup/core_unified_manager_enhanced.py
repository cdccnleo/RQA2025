#!/usr/bin/env python3
"""
统一配置管理器

提供统一的配置管理功能
"""

import yaml
from datetime import datetime
import json
import os
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from infrastructure.config.core.unified_interface import IConfigManagerComponent

logger = logging.getLogger(__name__)


class UnifiedConfigManager(IConfigManagerComponent):
    """统一配置管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._data: Dict[str, Dict[str, Any]] = {}

        self._initialized = False

        # 默认配置
        self.default_config = {
            "auto_reload": True,
            "validation_enabled": True,
            "encryption_enabled": False,
            "backup_enabled": True,
            "max_backup_files": 5,
            "config_file": "config.json"
        }

        self.default_config.update(self.config)
        self.config = self.default_config

    def initialize(self) -> bool:
        """初始化配置管理器"""
        try:
            self._initialized = True
            return True
        except Exception:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（接口实现）"""
        # 解析section和key
        if '.' in key:
            section, key_part = key.split('.', 1)
        else:
            section = 'default'
            key_part = key

        # 边界条件检查
        if not isinstance(section, str) or not section.strip():
            return default

        if not isinstance(key_part, str) or not key_part.strip():
            return default

        # 检查key长度，防止过长key导致性能问题
        if len(section) > 100 or len(key_part) > 100:
            return default

        # 检查是否包含危险字符
        dangerous_chars = ['<', '>', '|', '&', ';', '$', '`']
        if any(char in section + key_part for char in dangerous_chars):
            return default

        if section not in self._data:
            return default
        return self._data[section].get(key_part, default)

    def set(self, key: str, value: Any) -> bool:
        """设置配置值（接口实现）"""
        # 解析section和key
        if '.' in key:
            section, key_part = key.split('.', 1)
        else:
            section = 'default'
            key_part = key

        try:
            # 边界条件检查
            if not isinstance(section, str) or not section.strip():
                return False

            if not isinstance(key_part, str) or not key_part.strip():
                return False

            # 检查key长度
            if len(section) > 100 or len(key_part) > 100:
                return False

            # 检查是否包含危险字符
            dangerous_chars = ['<', '>', '|', '&', ';', '$', '`']
            if any(char in section + key_part for char in dangerous_chars):
                return False

            if section not in self._data:
                self._data[section] = {}

            self._data[section][key_part] = value
            return True
        except Exception:
            return False

    def delete(self, section: str, key: str) -> bool:
        """删除配置值"""
        try:
            # 边界条件检查
            if not isinstance(section, str) or not section.strip():
                return False

            if not isinstance(key, str) or not key.strip():
                return False

            # 检查section是否存在
            if section not in self._data:
                return False

            # 检查key是否存在
            if key not in self._data[section]:
                return False

            # 删除key
            del self._data[section][key]

            # 如果section为空，删除整个section
            if not self._data[section]:
                del self._data[section]

            return True
        except Exception:
            return False

    def update(self, config: Dict[str, Any]) -> None:
        """更新配置（接口实现）"""
        try:
            for key, value in config.items():
                if isinstance(value, dict):
                    # 如果值是字典，递归设置嵌套键
                    for nested_key, nested_value in value.items():
                        full_key = f"{key}.{nested_key}"
                        self.set(full_key, nested_value)
                else:
                    self.set(key, value)
        except Exception as e:
            raise ValueError(f"Failed to update config: {e}")

    def watch(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """监听配置变化（接口实现）"""
        # 简化实现，实际应该有更复杂的监听机制
        # 这里只是记录监听器
        if not hasattr(self, '_watchers'):
            self._watchers = {}
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)

    def reload(self) -> None:
        """重新加载配置（接口实现）"""
        try:
            config_file = self.config.get("config_file")
            if config_file and os.path.exists(config_file):
                self.load_config(config_file)
        except Exception as e:
            raise ValueError(f"Failed to reload config: {e}")

    def validate(self, config: Dict[str, Any]) -> bool:
        """验证配置（接口实现）"""
        try:
            # 基本验证
            if not isinstance(config, dict):
                return False

            # 检查配置项的合法性
            for key, value in config.items():
                if not isinstance(key, str) or not key.strip():
                    return False

                # 检查key长度
                if len(key) > 100:
                    return False

                # 检查危险字符
                dangerous_chars = ['<', '>', '|', '&', ';', '$', '`']
                if any(char in key for char in dangerous_chars):
                    return False

            return True
        except Exception:
            return False

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """获取配置节"""
        if section in self._data:
            return self._data[section].copy()
        return None

    def load_config(self, config_file: str) -> bool:
        """加载配置文件"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 如果是顶级配置对象，直接设置为default section
                    if isinstance(data, dict) and not any(isinstance(v, dict) and k != 'default' for k, v in data.items()):
                        self._data['default'] = data
                    else:
                        self._data.update(data)
                return True
            return False
        except Exception:
            return False

    def save_config(self, config_file: str) -> bool:
        """保存配置文件"""
        try:
            # 确保目录存在
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def get_all_sections(self) -> List[str]:
        """获取所有配置节"""
        return list(self._data.keys())

    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            config_file = self.config.get("config_file")
            if config_file:
                return self.load_config(config_file)
            return False
        except Exception:
            return False

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置"""
        try:
            # 如果没有提供config，使用当前配置
            if config is None:
                config = self._data

            # 基本验证
            if not isinstance(config, dict):
                return False

            # 如果有验证规则，使用规则验证
            if hasattr(self, '_validation_rules') and self._validation_rules:
                return self._validate_with_rules(config, self._validation_rules)

            # 检查配置项的合法性
            for key, value in config.items():
                if not isinstance(key, str) or not key.strip():
                    return False

                # 检查key长度
                if len(key) > 100:
                    return False

                # 检查危险字符
                dangerous_chars = ['<', '>', '|', '&', ';', '$', '`']
                if any(char in key for char in dangerous_chars):
                    return False

                # 检查值的基本合法性
                if isinstance(value, str) and not value.strip():
                    # 空字符串值无效
                    return False

                if isinstance(value, (int, float)) and value is None:
                    # None值对于数字类型无效
                    return False

            return True

        except Exception:
            return False

    def _validate_with_rules(self, config: Dict[str, Any], rules: Dict[str, Any]) -> bool:
        """使用验证规则验证配置"""
        try:
            for section_name, section_rules in rules.items():
                if section_name not in config:
                    continue

                section_config = config[section_name]
                if not isinstance(section_config, dict):
                    return False

                # 验证每个字段
                for field_name, field_rules in section_rules.items():
                    if field_name not in section_config:
                        if field_rules.get('required', False):
                            return False
                        continue

                    field_value = section_config[field_name]

                    # 类型检查
                    expected_type = field_rules.get('type')
                    if expected_type:
                        if expected_type == 'string' and not isinstance(field_value, str):
                            return False
                        elif expected_type == 'integer' and not isinstance(field_value, int):
                            return False
                        elif expected_type == 'number' and not isinstance(field_value, (int, float)):
                            return False
                        elif expected_type == 'boolean' and not isinstance(field_value, bool):
                            return False

                    # 范围检查
                    if 'min' in field_rules and field_value < field_rules['min']:
                        return False
                    if 'max' in field_rules and field_value > field_rules['max']:
                        return False

                    # 必需字段检查
                    if field_rules.get('required') and (field_value is None or (isinstance(field_value, str) and not field_value.strip())):
                        return False

            return True
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取配置管理器状态"""
        return {
            "initialized": self._initialized,
            "sections_count": len(self._data),
            "total_keys": sum(len(section) for section in self._data.values()),
            "config": self.config.copy()
        }

    def cleanup(self):
        """清理资源"""
        self._data.clear()
        self._initialized = False

    def merge_config(self, config: Dict[str, Any], override: bool = False) -> bool:
        """合并配置"""
        try:
            if override:
                # 覆盖模式：直接更新所有配置
                self._data.update(config)
            else:
                # 非覆盖模式：只添加不存在的section或key
                for section, values in config.items():
                    if section not in self._data:
                        # section不存在，直接添加
                        self._data[section] = values.copy() if isinstance(values, dict) else values
                    elif isinstance(values, dict):
                        # section存在，只添加不存在的key
                        if not isinstance(self._data[section], dict):
                            self._data[section] = {}
                        for key, value in values.items():
                            if key not in self._data[section]:
                                self._data[section][key] = value
                        # 如果values不是dict，则不覆盖已有的section
            return True
        except Exception:
            return False

    def export_config(self, format: str = "json") -> str:
        """导出配置"""
        try:
            if format.lower() == "json":
                return json.dumps(self._data, indent=2, ensure_ascii=False)
            else:
                return str(self._data)
        except Exception:
            return "{}"

# 为测试兼容性添加的方法

    def get_sections(self) -> List[str]:
        """获取所有sections（测试兼容性方法）"""
        return self.get_all_sections()

    def has_section(self, section: str) -> bool:
        """检查section是否存在"""
        return section in self._data

    def set_section(self, section: str, data: Dict[str, Any]) -> bool:
        """设置完整section"""
        try:
            if not isinstance(section, str) or not section.strip():
                return False
            if not isinstance(data, dict):
                return False
            self._data[section] = data.copy()
            return True
        except Exception:
            return False

    def delete_section(self, section: str) -> bool:
        """删除section"""
        try:
            if section in self._data:
                del self._data[section]
                return True
            return False
        except Exception:
            return False

    def clear_all(self) -> bool:
        """清空所有配置"""
        try:
            self._data.clear()
            return True
        except Exception:
            return False

    def load_from_file(self, file_path: str) -> bool:
        """从文件加载配置（测试兼容性方法）"""
        return self.load_config(file_path)

    def save_to_file(self, file_path: str) -> bool:
        """保存配置到文件（测试兼容性方法）"""
        return self.save_config(file_path)

    def backup_config(self, backup_dir: str) -> bool:
        """备份配置"""
        try:
            # 创建备份目录
            os.makedirs(backup_dir, exist_ok=True)

            # 生成备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"config_backup_{timestamp}.json")

            # 保存备份
            return self.save_config(backup_file)
        except Exception:
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        total_keys = 0
        for section_data in self._data.values():
            if isinstance(section_data, dict):
                total_keys += len(section_data)
            else:
                total_keys += 1  # 非字典section算作1个key

        summary = {
            "total_sections": len(self._data),
            "total_keys": total_keys,
            "sections": {}
        }

        for section_name, section_data in self._data.items():
            if isinstance(section_data, dict):
                summary["sections"][section_name] = {
                    "keys_count": len(section_data),
                    "keys": list(section_data.keys())[:5]  # 只显示前5个key
                }
            else:
                summary["sections"][section_name] = {
                    "keys_count": 1,
                    "keys": [str(section_data)[:20]]  # 显示值的字符串表示
                }

        return summary

    def restore_from_backup(self, backup_file: str) -> bool:
        """从备份恢复配置"""
        try:
            if os.path.exists(backup_file):
                return self.load_config(backup_file)
            return False
        except Exception:
            return False

    def set_validation_rules(self, rules: Dict[str, Any]) -> bool:
        """设置验证规则"""
        try:
            if not hasattr(self, '_validation_rules'):
                self._validation_rules = {}
            self._validation_rules.update(rules)
            return True
        except Exception:
            return False

    def enable_hot_reload(self, enabled: bool = True) -> bool:
        """启用热重载"""
        try:
            self.config['auto_reload'] = enabled
            return True
        except Exception:
            return False

# ==================== 配置管理增强功能 ====================

    def load_from_environment_variables(self, prefix: str = "RQA_") -> bool:
        """从环境变量加载配置（增强版）"""
        try:
            env_config = {}
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower().replace('_', '.')

                    # 智能类型转换
                    converted_value = self._convert_env_value(value)
                    env_config[config_key] = converted_value

            self.update(env_config)
            return True
        except Exception as e:
            logger.error(f"Failed to load environment variables: {e}")
            return False

    def _convert_env_value(self, value: str) -> Any:
        """智能转换环境变量值"""
        if not value:
            return value

        # 布尔值转换
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 数字转换
        try:
            if '.' in value and value.replace('.', '').replace('-', '').isdigit():
                return float(value)
            elif value.replace('-', '').isdigit():
                return int(value)
        except ValueError:
            pass

        # JSON转换
        if value.startswith(('{', '[', '"')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # 列表转换（逗号分隔）
        if ',' in value and not value.startswith(('{', '[')):
            return [item.strip() for item in value.split(',')]

        return value

    def get_config_with_source_info(self, key: str, default: Any = None) -> Dict[str, Any]:
        """获取配置值及来源信息"""
        return {
            'value': self.get(key, default),
            'source': 'merged_config',
            'available': key in str(self._data),
            'type': type(self.get(key, default)).__name__
        }

    def validate_config_integrity(self) -> Dict[str, Any]:
        """验证配置完整性"""
        validation_result = {
            'is_valid': True,
            'missing_keys': [],
            'type_mismatches': [],
            'recommendations': []
        }

        # 检查关键配置项
        required_keys = [
            'logging.level',
            'system.debug'
        ]

        for key in required_keys:
            if self.get(key) is None:
                validation_result['missing_keys'].append(key)
                validation_result['is_valid'] = False

        if validation_result['missing_keys']:
            validation_result['recommendations'].append('添加缺失的必需配置项')

        return validation_result

    def load_from_yaml_file(self, file_path: str) -> bool:
        """从 YAML 文件加载配置"""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return False

            with open(file_path_obj, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}

            self.update(yaml_data)
            return True

        except Exception as e:
            logger.error(f"Failed to load YAML file {file_path}: {e}")
            return False

    def export_config_with_metadata(self) -> Dict[str, Any]:
        """导出配置及元数据"""
        total_keys = 0
        for section_data in self._data.values():
            if isinstance(section_data, dict):
                total_keys += len(section_data)
            else:
                total_keys += 1  # 非字典section算作1个key

        return {
            'timestamp': time.time(),
            'config_data': self._data.copy(),
            'sections_count': len(self._data),
            'total_keys': total_keys,
            'status': self.get_status(),
            'format_version': '1.0'
        }

    def refresh_from_sources(self) -> bool:
        """从所有源刷新配置"""
        try:
            # 重新加载环境变量
            self.load_from_environment_variables()

            # 重新加载配置文件（如果存在）
            config_file = self.config.get("config_file")
            if config_file and os.path.exists(config_file):
                self.load_config(config_file)

            return True
        except Exception as e:
            logger.error(f"Failed to refresh config from sources: {e}")
            return False
