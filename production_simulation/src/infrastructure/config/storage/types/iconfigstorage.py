
"""配置文件存储相关类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import threading
from .configscope import ConfigScope


class IConfigStorage(ABC):
    """配置存储接口"""

    @abstractmethod
    def get(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> Optional[Any]:
        """获取配置值"""

    @abstractmethod
    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """设置配置值"""

    @abstractmethod
    def delete(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """删除配置"""

    @abstractmethod
    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """检查配置是否存在"""

    @abstractmethod
    def list_keys(self, scope: Optional[ConfigScope] = None) -> List[str]:
        """列出配置键"""

    @abstractmethod
    def save(self) -> bool:
        """保存配置"""

    @abstractmethod
    def load(self) -> bool:
        """加载配置"""


class BaseConfigStorage:
    """配置存储基类"""

    def __init__(self):
        self._data: Dict[ConfigScope, Dict[str, Any]] = {}
        self._lock = threading.RLock()

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

    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """检查配置是否存在"""
        with self._lock:
            return scope in self._data and key in self._data[scope]

    def _get_item(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION):
        """获取配置项对象（内部方法）"""
        with self._lock:
            if scope in self._data and key in self._data[scope]:
                return self._data[scope][key]
            return None




