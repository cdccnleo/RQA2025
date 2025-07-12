from typing import Any, Optional
from .data_version_manager import DataVersionManager
from threading import RLock

class VersionedStorageAdapter:
    """带版本控制的存储适配器"""

    def __init__(self, base_storage: Any):
        """
        初始化适配器
        :param base_storage: 基础存储对象
        """
        self._storage = base_storage
        self._version_manager = DataVersionManager()
        self._lock = RLock()

    def get(self, key: str, version: Optional[float] = None) -> Any:
        """
        获取数据(可选指定版本)
        :param key: 数据键
        :param version: 可选版本时间戳
        :return: 数据值
        """
        with self._lock:
            if version is not None:
                # 获取历史版本
                snapshot = self._version_manager.get_version(version)
                return snapshot.get(key) if snapshot else None
            return self._storage.get(key)

    def set(self, key: str, value: Any) -> float:
        """
        设置数据并创建快照
        :param key: 数据键
        :param value: 数据值
        :return: 快照时间戳
        """
        with self._lock:
            # 先获取当前完整状态
            current_state = self._storage.get_all()
            # 更新基础存储
            self._storage.set(key, value)
            # 记录变更
            self._version_manager.record_change(
                operation="set",
                key=key,
                value=value
            )
            # 创建快照
            return self._version_manager.take_snapshot(current_state)

    def delete(self, key: str) -> float:
        """
        删除数据并创建快照
        :param key: 数据键
        :return: 快照时间戳
        """
        with self._lock:
            current_state = self._storage.get_all()
            self._storage.delete(key)
            self._version_manager.record_change(
                operation="delete",
                key=key
            )
            return self._version_manager.take_snapshot(current_state)

    def get_history(self, key: str, since: float = 0) -> list:
        """
        获取键的变更历史
        :param key: 数据键
        :param since: 起始时间戳
        :return: 变更记录列表
        """
        with self._lock:
            return [
                log for log in self._version_manager.get_changelog(since)
                if log['params'].get('key') == key
            ]

    def rollback(self, version: float) -> bool:
        """
        回滚到指定版本
        :param version: 目标版本时间戳
        :return: 是否回滚成功
        """
        with self._lock:
            snapshot = self._version_manager.get_version(version)
            if not snapshot:
                return False

            # 恢复所有键值
            for key, value in snapshot.items():
                self._storage.set(key, value)

            # 记录回滚操作
            self._version_manager.record_change(
                operation="rollback",
                target_version=version
            )
            return True
