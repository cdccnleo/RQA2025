
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..core.exceptions import LogStorageError as StorageError
"""
基础设施层 - 日志存储基础实现

定义日志存储的基础接口和实现。
"""


class ILogStorage(ABC):
    """日志存储接口"""

    @abstractmethod
    def store(self, record: Dict[str, Any]) -> bool:
        """存储日志记录"""

    @abstractmethod
    def retrieve(self, query: Optional[Dict[str, Any]] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """检索日志记录"""

    @abstractmethod
    def delete(self, query: Dict[str, Any]) -> int:
        """删除日志记录"""

    @abstractmethod
    def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """统计日志记录数量"""

    @abstractmethod
    def clear(self) -> None:
        """清空存储"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取存储状态"""


class BaseStorage(ILogStorage):
    """基础日志存储实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础存储

        Args:
            config: 存储配置
        """
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)
        self.enabled = self.config.get('enabled', True)
        self.max_records = self.config.get('max_records', 0)  # 0表示不限制
        self.compression = self.config.get('compression', False)

    def store(self, record: Dict[str, Any]) -> bool:
        """存储日志记录"""
        if not self.enabled:
            return False

        try:
            return self._store(record)
        except Exception as e:
            raise StorageError(f"Failed to store log record: {e}")

    def retrieve(self, query: Optional[Dict[str, Any]] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """检索日志记录"""
        if not self.enabled:
            return []

        try:
            return self._retrieve(query, limit)
        except Exception as e:
            raise StorageError(f"Failed to retrieve log records: {e}")

    def delete(self, query: Dict[str, Any]) -> int:
        """删除日志记录"""
        if not self.enabled:
            return 0

        try:
            return self._delete(query)
        except Exception as e:
            raise StorageError(f"Failed to delete log records: {e}")

    def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """统计日志记录数量"""
        if not self.enabled:
            return 0

        try:
            return self._count(query)
        except Exception as e:
            raise StorageError(f"Failed to count log records: {e}")

    def clear(self) -> None:
        """清空存储"""
        if not self.enabled:
            return

        try:
            self._clear()
        except Exception as e:
            raise StorageError(f"Failed to clear storage: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取存储状态"""
        try:
            status = self._get_status()
            status.update({
                'name': self.name,
                'enabled': self.enabled,
                'type': self.__class__.__name__,
                'max_records': self.max_records,
                'compression': self.compression
            })
            return status
        except Exception as e:
            return {
                'name': self.name,
                'enabled': self.enabled,
                'type': self.__class__.__name__,
                'error': str(e),
                'status': 'error'
            }

    # 子类需要实现的抽象方法
    @abstractmethod
    def _store(self, record: Dict[str, Any]) -> bool:
        """实际的存储逻辑"""

    @abstractmethod
    def _retrieve(self, query: Optional[Dict[str, Any]] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """实际的检索逻辑"""

    @abstractmethod
    def _delete(self, query: Dict[str, Any]) -> int:
        """实际的删除逻辑"""

    @abstractmethod
    def _count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """实际的统计逻辑"""

    @abstractmethod
    def _clear(self) -> None:
        """实际的清空逻辑"""

    @abstractmethod
    def _get_status(self) -> Dict[str, Any]:
        """实际的状态获取逻辑"""


class MemoryStorage(BaseStorage):
    """内存日志存储实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化内存存储

        Args:
            config: 存储配置
        """
        super().__init__(config)
        self._storage: List[Dict[str, Any]] = []
        self._max_records = self.config.get('max_records', 10000)

    def _store(self, record: Dict[str, Any]) -> bool:
        """
        存储日志记录

        Args:
            record: 日志记录

        Returns:
            存储是否成功
        """
        try:
            self._storage.append(record)

            # 限制存储数量
            if len(self._storage) > self._max_records:
                self._storage.pop(0)

            return True
        except Exception:
            return False

    def _retrieve(self, query: Optional[Dict[str, Any]] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索日志记录

        Args:
            query: 查询条件
            limit: 限制返回数量

        Returns:
            匹配的日志记录列表
        """
        # 获取基础记录集合
        records = self._get_base_records(query)

        # 应用限制
        records = self._apply_limit(records, limit)

        return records

    def _get_base_records(self, query: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取基础记录集合"""
        if query is None:
            return self._storage
        else:
            return self._filter_records_by_query(query)

    def _filter_records_by_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据查询条件过滤记录"""
        filtered_records = []
        for record in self._storage:
            if self._record_matches_query(record, query):
                filtered_records.append(record)
        return filtered_records

    def _record_matches_query(self, record: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """检查记录是否匹配查询条件"""
        for key, value in query.items():
            if record.get(key) != value:
                return False
        return True

    def _apply_limit(self, records: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
        """应用数量限制"""
        if limit is not None:
            return records[-limit:]
        return records

    def _delete(self, query: Dict[str, Any]) -> int:
        """
        删除日志记录

        Args:
            query: 删除条件

        Returns:
            删除的记录数量
        """
        original_length = len(self._storage)
        self._storage = [
            record for record in self._storage
            if not all(record.get(key) == value for key, value in query.items())
        ]
        return original_length - len(self._storage)

    def _count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """
        统计日志记录数量

        Args:
            query: 查询条件

        Returns:
            匹配的记录数量
        """
        if query is None:
            return len(self._storage)

        count = 0
        for record in self._storage:
            if all(record.get(key) == value for key, value in query.items()):
                count += 1
        return count

    def _clear(self) -> None:
        """清空存储"""
        self._storage.clear()

    def _get_status(self) -> Dict[str, Any]:
        """
        获取存储状态

        Returns:
            存储状态信息
        """
        return {
            "storage_type": "memory",
            "record_count": len(self._storage),
            "max_records": self._max_records,
            "usage_percent": (len(self._storage) / self._max_records) * 100
        }
