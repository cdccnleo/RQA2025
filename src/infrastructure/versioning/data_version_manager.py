import time
from typing import Dict, List, Any
import threading
from collections import deque

class DataVersionManager:
    """数据版本管理核心类"""

    def __init__(self, max_snapshots: int = 30, max_changelog: int = 1000):
        """
        初始化版本管理器
        :param max_snapshots: 保留的最大快照数量
        :param max_changelog: 保留的最大变更记录数量
        """
        self._snapshots: Dict[float, Any] = {}  # {timestamp: data_snapshot}
        self._changelog: deque = deque(maxlen=max_changelog)
        self._lock = threading.RLock()
        self.max_snapshots = max_snapshots

    def take_snapshot(self, data: Any) -> float:
        """
        创建数据快照
        :param data: 需要快照的数据
        :return: 快照时间戳
        """
        with self._lock:
            ts = time.time()
            self._snapshots[ts] = self._deep_copy(data)
            self._cleanup_snapshots()
            return ts

    def record_change(self, operation: str, **params) -> None:
        """
        记录数据变更
        :param operation: 操作类型
        :param params: 操作参数
        """
        with self._lock:
            self._changelog.append({
                'timestamp': time.time(),
                'operation': operation,
                'params': params
            })

    def get_version(self, timestamp: float) -> Any:
        """
        获取指定版本的数据
        :param timestamp: 快照时间戳
        :return: 该时间点的数据快照
        """
        with self._lock:
            return self._deep_copy(self._snapshots.get(timestamp))

    def get_changelog(self, since: float = 0) -> List[dict]:
        """
        获取变更记录
        :param since: 起始时间戳
        :return: 变更记录列表
        """
        with self._lock:
            return [log for log in self._changelog if log['timestamp'] >= since]

    def _cleanup_snapshots(self) -> None:
        """清理过期的快照"""
        if len(self._snapshots) > self.max_snapshots:
            oldest = sorted(self._snapshots.keys())[0]
            del self._snapshots[oldest]

    @staticmethod
    def _deep_copy(data: Any) -> Any:
        """深拷贝数据"""
        # 实际实现应根据数据类型优化
        try:
            import copy
            return copy.deepcopy(data)
        except:
            return data

    @property
    def latest_snapshot(self) -> float:
        """获取最新快照时间"""
        with self._lock:
            return max(self._snapshots.keys()) if self._snapshots else 0
