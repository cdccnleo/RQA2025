
import logging
import threading
import time
from collections import defaultdict
import hashlib
from typing import Dict, Any, Optional, List
from .configitem import ConfigItem
from .configscope import ConfigScope
from .iconfigstorage import IConfigStorage
from .storageconfig import StorageConfig

"""配置文件存储相关类"""
logger = logging.getLogger(__name__)


class MemoryConfigStorage(IConfigStorage):
    """内存配置存储实现"""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._data: Dict[ConfigScope, Dict[str, ConfigItem]] = defaultdict(dict)
        self._lock = threading.RLock()

    def get(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> Optional[Any]:
        """获取配置值"""
        with self._lock:
            if scope in self._data and key in self._data[scope]:
                return self._data[scope][key].value
            return None

    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """设置配置值"""
        try:
            with self._lock:
                timestamp = time.time()
                version = hashlib.sha256(f"{key}:{value}:{timestamp}".encode()).hexdigest()[:8]

                item = ConfigItem(
                    key=key,
                    value=value,
                    scope=scope,
                    timestamp=timestamp,
                    version=version
                )

                self._data[scope][key] = item
                return True
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False

    def delete(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """删除配置"""
        with self._lock:
            if scope in self._data and key in self._data[scope]:
                del self._data[scope][key]
                return True
            return False

    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """检查配置是否存在"""
        with self._lock:
            return scope in self._data and key in self._data[scope]

    def list_keys(self, scope: Optional[ConfigScope] = None) -> List[str]:
        """列出配置键"""
        with self._lock:
            if scope:
                return list(self._data.get(scope, {}).keys())
            else:
                all_keys = []
                for scope_data in self._data.values():
                    all_keys.extend(scope_data.keys())
                return all_keys

    def save(self) -> bool:
        """内存存储不需要保存"""
        return True

    def load(self) -> bool:
        """内存存储不需要加载"""
        return True

    def _get_item(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION):
        """获取配置项对象（内部方法）"""
        with self._lock:
            if scope in self._data and key in self._data[scope]:
                return self._data[scope][key]
            return None




