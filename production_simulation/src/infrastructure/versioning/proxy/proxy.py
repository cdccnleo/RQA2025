"""
版本代理模块

提供轻量级的版本访问代理，封装 `VersionManager` 并加入简单缓存，
以满足测试场景对历史接口的期待。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable, Dict, Optional, Union

from ..core.version import Version
from ..manager.manager import VersionManager, _ensure_version


class VersionProxy:
    """
    版本控制代理。

    - 对外暴露 `register_version`、`get_version`、`set_version` 等接口，
      兼容老旧测试用例。
    - 内部使用 `VersionManager` 保存真实数据，并维护一个轻量的 LRU 缓存。
    """

    def __init__(
        self,
        version_manager: Optional[VersionManager] = None,
        *,
        max_cache_size: int = 256,
        cache_ttl: int = 300,
    ) -> None:
        self.version_manager = version_manager or VersionManager()
        self._cache: OrderedDict[str, Version] = OrderedDict()
        self._cache_timestamps: Dict[str, float] = {}
        self._max_cache_size = max_cache_size
        self._cache_ttl = cache_ttl
        self._lock = RLock()
        self._default_key = "__default__"

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def register_version(self, name: str, version: Union[str, Version]) -> Version:
        with self._lock:
            self.version_manager.register_version(name, version)
            self._invalidate_cache(name)
            return self.version_manager.get_version(name)

    def update_version(self, name: str, version: Union[str, Version]) -> Version:
        with self._lock:
            self.version_manager.update_version(name, version)
            self._invalidate_cache(name)
            return self.version_manager.get_version(name)

    def set_version(self, version: Union[str, Version]) -> bool:
        """
        设置默认版本，兼容旧的 `set_version` 调用形式。
        """
        with self._lock:
            result = self.version_manager.set_current_version(self._default_key, version)
            self._invalidate_cache(self._default_key)
            return result

    def get_version(
        self,
        name: Optional[str] = None,
        version: Union[str, Version, None] = None,
    ) -> Optional[Version]:
        """
        获取版本。

        - 未指定 `name` 时返回当前版本。
        - 指定了 `version` 时尝试从历史列表中查找。
        """
        with self._lock:
            if name is None:
                return self.version_manager.get_current_version()

            cache_key = f"{name}:{str(version) if version else 'latest'}"
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            if version is None:
                result = self.version_manager.get_version(name)
            else:
                version_obj = _ensure_version(version)
                result = None
                history = self.version_manager.get_version_history(name)
                for item in reversed(history):
                    if item == version_obj:
                        result = item
                        break
                if result is None:
                    current = self.version_manager.get_version(name)
                    if current and current == version_obj:
                        result = current

            if result is not None:
                self._add_to_cache(cache_key, result)
            return result

    def list_versions(self) -> Any:
        """返回版本列表，保持向后兼容。"""
        return self.version_manager.list_versions()

    def execute(self, action: str, **kwargs: Any) -> Any:
        """
        简单的命令式接口，用于旧代码路径。
        """
        actions: Dict[str, Callable[..., Any]] = {
            "get_version": self.get_version,
            "register_version": self.register_version,
            "update_version": self.update_version,
        }
        handler = actions.get(action)
        if not handler:
            raise ValueError(f"不支持的动作: {action}")
        return handler(**kwargs)

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------
    def _get_from_cache(self, key: str) -> Optional[Version]:
        current = time.time()
        if key in self._cache and key in self._cache_timestamps:
            if current - self._cache_timestamps[key] < self._cache_ttl:
                self._cache.move_to_end(key)
                return self._cache[key]
            self._remove_cache_entry(key)
        return None

    def _add_to_cache(self, key: str, value: Version) -> None:
        if len(self._cache) >= self._max_cache_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._cache_timestamps.pop(oldest_key, None)
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
        self._cache.move_to_end(key)

    def _invalidate_cache(self, name: str) -> None:
        keys = [key for key in self._cache.keys() if key.startswith(f"{name}:")]
        for key in keys:
            self._remove_cache_entry(key)

    def _remove_cache_entry(self, key: str) -> None:
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)


def get_default_version_proxy() -> VersionProxy:
    if not hasattr(get_default_version_proxy, "_instance"):
        get_default_version_proxy._instance = VersionProxy()
    return get_default_version_proxy._instance
