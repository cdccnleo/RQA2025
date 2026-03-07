"""
configuration_manager 模块

提供 configuration_manager 相关功能和接口。
"""

import json
import os

from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from typing import Dict, List, Optional, Any
"""
配置管理器

负责系统配置的集中管理和动态更新
"""


class ConfigurationManager:
    """配置管理器"""

    def __init__(self, config_file: Optional[str] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        self.config_file = config_file
        self.config: Dict[str, Any] = {}

        # 默认配置
        self._load_default_config()

        # 从文件加载配置
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)

    def _load_default_config(self):
        """加载默认配置"""
        self.config = {
            "system": {
                "name": "Monitoring Alert System",
                "version": "1.0.0",
                "environment": "production"
            },
            "performance_monitor": {
                "update_interval": 5,
                "history_size": 1000,
                "enabled": True
            },
            "alert_manager": {
                "max_active_alerts": 100,
                "alert_retention_days": 30,
                "enabled": True
            },
            "notification_manager": {
                "channels": {
                    "email": {"enabled": True, "recipients": []},
                    "sms": {"enabled": False, "recipients": []},
                    "webhook": {"enabled": False, "url": ""},
                    "log": {"enabled": True, "level": "WARNING"}
                }
            },
            "test_monitor": {
                "max_concurrent_tests": 10,
                "test_timeout": 300,
                "enabled": True
            },
            "alert_rules": {
                "check_interval": 30,
                "auto_resolve_timeout": 3600
            },
            "health_monitor": {
                "cpu_warning_threshold": 80.0,
                "cpu_critical_threshold": 90.0,
                "memory_warning_threshold": 85.0,
                "memory_critical_threshold": 90.0,
                "disk_warning_threshold": 90.0,
                "disk_critical_threshold": 95.0
            },
            "logging": {
                "level": "INFO",
                "max_file_size": 10485760,  # 10MB
                "backup_count": 5
            }
        }

    def _load_from_file(self, config_file: str):
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)

            # 深度合并配置
            self._deep_merge(self.config, file_config)

            self.logger.log_info(f"配置已从文件加载: {config_file}")

        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "从文件加载配置失败",
                "config_file": config_file
            })

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get_config(self, key: Optional[str] = None) -> Any:
        """获取配置值"""
        if key is None:
            return self.config.copy()

        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return None

    def set_config(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config

        # 创建嵌套结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value
        self.logger.log_info(f"配置已更新: {key} = {value}")

    def update_config(self, updates: Dict[str, Any]):
        """批量更新配置"""
        try:
            self._deep_merge(self.config, updates)
            self.logger.log_info(f"配置已批量更新，包含 {len(updates)} 个更新项")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "批量更新配置失败"})

    def save_to_file(self, config_file: Optional[str] = None):
        """保存配置到文件"""
        try:
            file_path = config_file or self.config_file
            if not file_path:
                raise ValueError("未指定配置文件路径")

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            self.logger.log_info(f"配置已保存到文件: {file_path}")

        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "保存配置到文件失败",
                "config_file": file_path
            })

    def reset_to_defaults(self):
        """重置为默认配置"""
        self._load_default_config()
        self.logger.log_info("配置已重置为默认值")

    def validate_config(self) -> List[str]:
        """验证配置有效性"""
        errors = []

        try:
            # 验证必需的配置项
            required_sections = ["system", "performance_monitor", "alert_manager"]
            for section in required_sections:
                if section not in self.config:
                    errors.append(f"缺少必需的配置节: {section}")

            # 验证数值范围
            if self.get_config("performance_monitor.update_interval") <= 0:
                errors.append("performance_monitor.update_interval 必须大于0")

            if self.get_config("alert_manager.max_active_alerts") <= 0:
                errors.append("alert_manager.max_active_alerts 必须大于0")

            # 验证阈值合理性
            health_config = self.get_config("health_monitor")
            if health_config:
                if health_config.get("cpu_warning_threshold", 0) >= health_config.get("cpu_critical_threshold", 100):
                    errors.append("CPU警告阈值不能大于等于严重阈值")

                if health_config.get("memory_warning_threshold", 0) >= health_config.get("memory_critical_threshold", 100):
                    errors.append("内存警告阈值不能大于等于严重阈值")

        except Exception as e:
            errors.append(f"配置验证过程中发生错误: {str(e)}")

        return errors

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "total_sections": len(self.config),
            "sections": list(self.config.keys()),
            "has_custom_config": self.config_file is not None and os.path.exists(self.config_file or ""),
            "validation_errors": self.validate_config()
        }

    def export_config(self, format: str = "json") -> str:
        """导出配置"""
        if format.lower() == "json":
            return json.dumps(self.config, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def import_config(self, config_data: str, format: str = "json"):
        """导入配置"""
        try:
            if format.lower() == "json":
                new_config = json.loads(config_data)
                self.update_config(new_config)
                self.logger.log_info("配置已从字符串导入")
            else:
                raise ValueError(f"不支持的导入格式: {format}")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "导入配置失败"})
            raise

    def get_config_diff(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """比较配置差异"""
        def dict_diff(d1, d2, path=""):
            diff = {}
            for key in set(d1.keys()) | set(d2.keys()):
                new_path = f"{path}.{key}" if path else key

                if key not in d2:
                    diff[new_path] = {"type": "removed", "old_value": d1[key]}
                elif key not in d1:
                    diff[new_path] = {"type": "added", "new_value": d2[key]}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    nested_diff = dict_diff(d1[key], d2[key], new_path)
                    diff.update(nested_diff)
                elif d1[key] != d2[key]:
                    diff[new_path] = {"type": "changed", "old_value": d1[key], "new_value": d2[key]}

            return diff

        return dict_diff(self.config, other_config)
