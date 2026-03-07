
from typing import Any, Dict
"""
配置处理器 - 提取自UnifiedConfigManager.set方法的处理逻辑

Phase 6.0复杂方法治理: 将配置设置逻辑分离为专门的处理器
"""


class ConfigValueProcessor:
    """配置值处理器 - 负责配置值的设置和处理逻辑"""

    def __init__(self, config_data: Dict[str, Any]):
        self._data = config_data

    def set_value(self, key: str, value: Any, key_parts: list) -> bool:
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值
            key_parts: 解析后的键部分

        Returns:
            bool: 设置是否成功
        """
        section = key_parts[0]

        if len(key_parts) == 1:
            # 单个部分，设置到default section
            return self._set_single_level_value(section, value)
        else:
            # 多级嵌套
            return self._set_nested_value(section, value, key_parts)

    def _set_single_level_value(self, section: str, value: Any) -> bool:
        """设置单级值"""
        try:
            # 初始化default section
            if 'default' not in self._data:
                self._data['default'] = {}

            # 设置值
            self._data['default'][section] = value

            # 如果这个键之前是一个section，需要删除它
            if section in self._data and isinstance(self._data[section], dict):
                del self._data[section]

            return True
        except Exception:
            return False

    def _set_nested_value(self, section: str, value: Any, key_parts: list) -> bool:
        """设置嵌套值"""
        try:
            # 确保顶层section是字典
            if section in self._data and not isinstance(self._data[section], dict):
                self._data[section] = {}
            elif section not in self._data:
                self._data[section] = {}

            # 递归设置嵌套值
            current = self._data[section]
            for part in key_parts[1:-1]:  # 除了第一个和最后一个部分
                if not isinstance(current, dict):
                    return False
                if part not in current:
                    current[part] = {}
                current = current[part]

            # 设置最终值
            final_key = key_parts[-1]
            if not isinstance(current, dict):
                return False

            current[final_key] = value
            return True
        except Exception:
            return False

    def get_old_value(self, key: str) -> Any:
        """获取旧值用于比较"""
        # 这里可以调用配置管理器的get方法
        # 为了解耦，我们暂时简化处理
        return None




