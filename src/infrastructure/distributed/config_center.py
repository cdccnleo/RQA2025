"""
config_center 模块

提供 config_center 相关功能和接口。
"""

import json
import logging
from collections import defaultdict
from copy import deepcopy

import hashlib
import time

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分布式配置中心
提供配置的分布式管理和同步功能
"""

logger = logging.getLogger(__name__)


class ConfigEventType(Enum):
    """配置事件类型"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    SYNCED = "synced"


@dataclass
class ConfigEntry:
    """配置条目"""
    key: str
    value: Any
    version: int
    timestamp: float
    checksum: str
    metadata: Dict[str, Any]


@dataclass
class ConfigEvent:
    """配置事件"""
    event_type: ConfigEventType
    key: str
    old_value: Any = None
    new_value: Any = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ConfigCenterManager:
    """分布式配置中心管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化配置中心管理器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 配置存储
        self._configs: Dict[str, ConfigEntry] = {}
        self._listeners: defaultdict[str, List[Callable]] = defaultdict(list)
        self._baseline_timestamps: Dict[str, float] = {}

        # 版本管理
        self._global_version = 0

        # 缓存配置
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5分钟

        self.logger.info("分布式配置中心初始化完成")

    def set_config(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        设置配置项

        Args:
            key: 配置键
            value: 配置值
            metadata: 元数据

        Returns:
            bool: 是否设置成功
        """
        try:
            # 计算校验和
            checksum = self._calculate_checksum(value)

            # 获取旧值用于事件
            old_value = None
            if key in self._configs:
                old_value = self._configs[key].value

            # 创建配置条目
            entry = ConfigEntry(
                key=key,
                value=value,
                version=self._global_version + 1,
                timestamp=time.time(),
                checksum=checksum,
                metadata=metadata or {}
            )
            # 存储配置
            self._configs[key] = entry
            if key not in self._baseline_timestamps:
                self._baseline_timestamps[key] = entry.timestamp
            self._global_version += 1

            # 触发事件
            event = ConfigEvent(
                event_type=ConfigEventType.CREATED if old_value is None else ConfigEventType.UPDATED,
                key=key,
                old_value=old_value,
                new_value=value
            )
            self._notify_listeners(key, event)

            self.logger.info(f"配置设置成功: {key} = {value}")
            return True

        except Exception as e:
            self.logger.error(f"设置配置失败: {key}, 错误: {e}")
            return False

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键
            default: 默认值

        Returns:
            Any: 配置值
        """
        if key in self._configs:
            entry = self._configs[key]
            # 检查配置是否过期（如果启用了TTL）
            if self.cache_enabled and self._is_expired(entry):
                self.logger.warning(f"配置已过期: {key}")
                return default
            return entry.value

        self.logger.debug(f"配置不存在，使用默认值: {key} = {default}")
        return default

    def delete_config(self, key: str) -> bool:
        """
        删除配置项

        Args:
            key: 配置键

        Returns:
            bool: 是否删除成功
        """
        if key in self._configs:
            old_value = self._configs[key].value
            del self._configs[key]
            self._baseline_timestamps.pop(key, None)

            # 触发事件
            event = ConfigEvent(
                event_type=ConfigEventType.DELETED,
                key=key,
                old_value=old_value
            )
            self._notify_listeners(key, event)

            self.logger.info(f"配置删除成功: {key}")
            return True

        self.logger.warning(f"尝试删除不存在的配置: {key}")
        return False

    def list_configs(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        列出配置项

        Args:
            prefix: 配置键前缀过滤

        Returns:
            Dict[str, Any]: 配置字典
        """
        result = {}
        for key, entry in self._configs.items():
            if prefix is None or key.startswith(prefix):
                if not self.cache_enabled or not self._is_expired(entry):
                    result[key] = entry.value

        return result

    def watch_config(self, key: str, callback: Callable[[ConfigEvent], None]):
        """
        监听配置变化

        Args:
            key: 配置键
            callback: 回调函数
        """
        listeners = self._listeners.setdefault(key, [])
        if callback not in listeners:
            listeners.append(callback)
        self.logger.info(f"配置监听器已添加: {key}")

    def unwatch_config(self, key: str, callback: Callable[[ConfigEvent], None]):
        """
        取消监听配置变化

        Args:
            key: 配置键
            callback: 回调函数
        """
        if key in self._listeners:
            listeners = self._listeners[key]
            try:
                listeners.remove(callback)
            except ValueError:
                self.logger.warning(f"尝试移除不存在的监听器: {key}")
            else:
                if not listeners:
                    self._listeners.pop(key, None)
                self.logger.info(f"配置监听器已移除: {key}")

    def sync_configs(self, remote_configs: Dict[str, Any]) -> int:
        """
        同步远程配置

        Args:
            remote_configs: 远程配置字典

        Returns:
            int: 同步的配置数量
        """
        synced_count = 0

        for key, value in remote_configs.items():
            current_entry = self._configs.get(key)
            if current_entry and current_entry.value == value:
                continue

            checksum = self._calculate_checksum(value)
            entry = ConfigEntry(
                key=key,
                value=value,
                version=self._global_version + 1,
                timestamp=time.time(),
                checksum=checksum,
                metadata={"source": "remote_sync"}
            )
            self._configs[key] = entry
            if key not in self._baseline_timestamps:
                self._baseline_timestamps[key] = entry.timestamp
            self._global_version += 1

            event = ConfigEvent(
                event_type=ConfigEventType.SYNCED,
                key=key,
                old_value=current_entry.value if current_entry else None,
                new_value=value
            )
            self._notify_listeners(key, event)
            synced_count += 1

        if synced_count > 0:
            self.logger.info(f"配置同步完成，共同步 {synced_count} 个配置项")

        return synced_count

    def export_configs(self) -> Dict[str, Any]:
        """
        导出所有配置

        Returns:
            Dict[str, Any]: 配置字典
        """
        return {key: entry.value for key, entry in self._configs.items()}

    def get_config_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取配置详细信息

        Args:
            key: 配置键

        Returns:
            Optional[Dict[str, Any]]: 配置信息字典
        """
        if key in self._configs:
            entry = self._configs[key]
            return {
                "key": entry.key,
                "value": entry.value,
                "version": entry.version,
                "timestamp": entry.timestamp,
                "checksum": entry.checksum,
                "metadata": deepcopy(entry.metadata),
                "expired": self._is_expired(entry) if self.cache_enabled else False
            }
        return None

    def _calculate_checksum(self, value: Any) -> str:
        """计算配置值的校验和"""
        value_str = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(value_str.encode()).hexdigest()

    def _is_expired(self, entry: ConfigEntry) -> bool:
        """检查配置是否过期"""
        if not self.cache_enabled:
            return False

        if entry.key in self._configs:
            if time.time() - entry.timestamp > self.cache_ttl:
                return True

        baseline = self._baseline_timestamps.get(entry.key)
        if baseline is None:
            baseline = entry.metadata.get("__baseline_timestamp__", entry.timestamp)
            self._baseline_timestamps[entry.key] = baseline
        return abs(baseline - entry.timestamp) > self.cache_ttl

    def _notify_listeners(self, key: str, event: ConfigEvent):
        """通知监听器"""
        for callback in list(self._listeners.get(key, [])):
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"配置监听器执行失败: {key}, 错误: {e}")

    # ------------------------------------------------------------------ #
    # 兼容旧接口
    # ------------------------------------------------------------------ #

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return self.set_config(key, value, metadata)

    def get(self, key: str, default: Any = None) -> Any:
        return self.get_config(key, default)

    def delete(self, key: str) -> bool:
        return self.delete_config(key)

    def watch(self, key: str, callback: Callable[[ConfigEvent], None]):
        self.watch_config(key, callback)

    def unwatch(self, key: str, callback: Callable[[ConfigEvent], None]):
        self.unwatch_config(key, callback)

    def sync(self, remote_configs: Dict[str, Any]) -> int:
        return self.sync_configs(remote_configs)

    def export(self) -> Dict[str, Any]:
        return self.export_configs()

    def clear_expired_configs(self) -> int:
        """
        清除过期的配置

        Returns:
            int: 清除的配置数量
        """
        if not self.cache_enabled:
            return 0

        expired_keys = []
        current_time = time.time()

        for key, entry in list(self._configs.items()):
            if current_time - entry.timestamp > self.cache_ttl:
                expired_keys.append(key)
                del self._configs[key]
                self._baseline_timestamps.pop(key, None)

        if expired_keys:
            self.logger.info(f"清除过期配置: {len(expired_keys)} 个")

        return len(expired_keys)


class ConfigCenter(ConfigCenterManager):
    """向后兼容的配置中心门面类。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)


__all__ = [
    "ConfigCenterManager",
    "ConfigCenter",
    "ConfigEntry",
    "ConfigEvent",
    "ConfigEventType",
]
