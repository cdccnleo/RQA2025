
from typing import Dict, Any, List, Optional
import json
import logging
import os
#!/usr/bin/env python3
"""
环境变量配置加载器

提供智能的环境变量配置加载功能
"""

logger = logging.getLogger(__name__)


class EnvironmentConfigLoader:
    """环境变量配置加载器"""

    def __init__(self, prefixes: Optional[List[str]] = None):
        self.prefixes = prefixes or ['RQA_', 'CONFIG_', 'APP_']

    def load_all(self) -> Dict[str, Any]:
        """加载所有匹配前缀的环境变量"""
        config = {}

        for prefix in self.prefixes:
            prefix_config = self.load_with_prefix(prefix)
            config.update(prefix_config)

        return config

    def load_with_prefix(self, prefix: str) -> Dict[str, Any]:
        """使用指定前缀加载环境变量"""
        config = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = self._transform_key(key, prefix)
                config_value = self._transform_value(value)

                # 支持嵌套配置
                self._set_nested_value(config, config_key, config_value)

        return config

    def _transform_key(self, env_key: str, prefix: str) -> str:
        """转换环境变量键名"""
        # 移除前缀并转换为小写，下划线转点号
        key = env_key[len(prefix):].lower().replace('_', '.')
        return key

    def _transform_value(self, value: str) -> Any:
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

    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """设置嵌套配置值"""
        if '.' not in key:
            config[key] = value
            return

        parts = key.split('.')
        current = config

        # 创建嵌套结构
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # 设置最终值
        current[parts[-1]] = value

    def get_env_summary(self) -> Dict[str, Any]:
        """获取环境变量摘要"""
        summary = {
            'total_env_vars': len(os.environ),
            'matched_vars': {},
            'prefixes': self.prefixes
        }

        for prefix in self.prefixes:
            matched = []
            for key in os.environ.keys():
                if key.startswith(prefix):
                    matched.append(key)
            summary['matched_vars'][prefix] = {
                'count': len(matched),
                'vars': matched[:10]  # 只显示前10个
            }

        return summary


# 兼容性别名
EnvLoader = EnvironmentConfigLoader




