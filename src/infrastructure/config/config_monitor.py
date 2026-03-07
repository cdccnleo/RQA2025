
from .config_event import ConfigChangeEvent
from typing import Dict, Any, List, Callable
import logging
#!/usr/bin/env python3
"""
配置监控器
监控配置变更和性能指标
"""

logger = logging.getLogger(__name__)


class ConfigMonitor:
    """配置监控器"""

    def __init__(self):
        self.listeners: List[Callable] = []
        self.change_history: List[ConfigChangeEvent] = []
        self.max_history_size = 1000

    def add_listener(self, listener: Callable) -> None:
        """添加变更监听器"""
        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener: Callable) -> None:
        """移除变更监听器"""
        if listener in self.listeners:
            self.listeners.remove(listener)

    def record_config_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """记录配置变更"""
        event = ConfigChangeEvent(key, old_value, new_value)

        # 添加到历史记录
        self.change_history.append(event)

        # 限制历史记录大小
        if len(self.change_history) > self.max_history_size:
            self.change_history.pop(0)

        # 通知监听器
        for listener in self.listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"配置监听器执行失败: {e}")

    def get_recent_changes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的变更记录"""
        if limit == 0:
            # limit为0时返回所有记录
            recent_changes = self.change_history
        elif limit > 0:
            recent_changes = self.change_history[-limit:]
        else:
            # 负数limit返回空列表
            recent_changes = []

        return [
            {
                'key': event.key,
                'old_value': event.old_value,
                'new_value': event.new_value,
                'timestamp': event.timestamp
            }
            for event in recent_changes
        ]

    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        return {
            'listener_count': len(self.listeners),
            'change_history_size': len(self.change_history),
            'max_history_size': self.max_history_size,
            'is_active': True
        }

    def get_change_statistics(self) -> Dict[str, Any]:
        """获取变更统计信息"""
        if not self.change_history:
            return {
                'total_changes': 0,
                'unique_keys': 0,
                'change_types': {},
                'time_range': None
            }

        unique_keys = len(set(event.key for event in self.change_history))
        oldest_change = min(event.timestamp for event in self.change_history)
        newest_change = max(event.timestamp for event in self.change_history)

        # 统计变更类型
        change_types = {}
        for event in self.change_history:
            change_type = event.data.get('change_type', 'unknown')
            change_types[change_type] = change_types.get(change_type, 0) + 1

        return {
            'total_changes': len(self.change_history),
            'unique_keys': unique_keys,
            'change_types': change_types,
            'time_range': {
                'oldest': oldest_change,
                'newest': newest_change,
                'duration': newest_change - oldest_change
            }
        }

    def clear_history(self) -> None:
        """清除变更历史记录"""
        self.change_history.clear()




