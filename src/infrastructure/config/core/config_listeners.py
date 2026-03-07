"""
config_listeners 模块

提供 config_listeners 相关功能和接口。
"""

import logging

from typing import Any, Dict, Callable, Optional, List
"""
配置监听器管理器 - 提取自UnifiedConfigManager.set方法的监听器逻辑

Phase 6.0复杂方法治理: 将监听器管理逻辑分离为专门的管理器
"""

logger = logging.getLogger(__name__)


class ConfigListenerManager:
    """配置监听器管理器 - 负责监听器的注册、触发和管理"""

    def __init__(self):
        self._watchers: Dict[str, List[Callable]] = {}

    def add_watcher(self, key: str, callback: Callable) -> None:
        """
        添加监听器

        Args:
            key: 监听的键
            callback: 回调函数
        """
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)

    def remove_watcher(self, key: str, callback: Callable) -> None:
        """
        移除监听器

        Args:
            key: 监听的键
            callback: 回调函数
        """
        if key in self._watchers:
            try:
                self._watchers[key].remove(callback)
                if not self._watchers[key]:
                    del self._watchers[key]
            except ValueError:
                pass  # 回调函数不在列表中

    def trigger_listeners(self, key: str, value: Any, old_value: Any = None) -> None:
        """
        触发监听器

        Args:
            key: 配置键
            value: 新值
            old_value: 旧值（用于比较）
        """
        # 触发精确匹配的监听器
        if key in self._watchers:
            for callback in self._watchers[key]:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.warning(f"监听器回调失败: {e}")

        # 如果值发生变化，触发通配符监听器
        if old_value != value:
            self._trigger_wildcard_listeners(key, value)

    def _trigger_wildcard_listeners(self, key: str, value: Any) -> None:
        """触发通配符监听器"""
        try:
            # 解析section
            parts = key.split('.')
            if not parts:
                return

            section = parts[0]
            wildcard_key = f"{section}.*"

            # 触发通配符监听器
            if wildcard_key in self._watchers:
                for callback in self._watchers[wildcard_key]:
                    try:
                        callback(key, value)
                    except Exception as e:
                        logger.warning(f"通配符监听器回调失败: {e}")

        except Exception as e:
            logger.warning(f"通配符监听器处理失败: {e}")

    def has_watchers(self, key: str) -> bool:
        """
        检查是否有监听器

        Args:
            key: 配置键

        Returns:
            bool: 是否有监听器
        """
        return key in self._watchers

    def get_watchers(self, key: str) -> List[Callable]:
        """
        获取指定键的监听器列表

        Args:
            key: 配置键

        Returns:
            List[Callable]: 监听器列表
        """
        return self._watchers.get(key, []).copy()

    def clear_watchers(self, key: Optional[str] = None) -> None:
        """
        清除监听器

        Args:
            key: 指定键，如果为None则清除所有监听器
        """
        if key is None:
            self._watchers.clear()
        elif key in self._watchers:
            del self._watchers[key]

    def add_listener(self, key: str, listener: Callable) -> None:
        """
        添加监听器 (兼容性方法)

        Args:
            key: 监听的键
            listener: 监听器函数
        """
        self.add_watcher(key, listener)

    def remove_listener(self, key: str, listener: Callable) -> None:
        """
        移除监听器 (兼容性方法)

        Args:
            key: 监听的键
            listener: 监听器函数
        """
        self.remove_watcher(key, listener)




