from abc import ABC, abstractmethod
from typing import Any, Optional

class ICacheBackend(ABC):
    """
    缓存后端接口定义，所有缓存后端实现必须遵循此接口
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存的值，如果不存在则返回None
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 缓存时间(秒)，None表示永不过期

        Returns:
            是否设置成功
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除缓存值

        Args:
            key: 要删除的缓存键

        Returns:
            是否删除成功
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            是否清空成功
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在

        Args:
            key: 要检查的缓存键

        Returns:
            是否存在
        """
        pass
