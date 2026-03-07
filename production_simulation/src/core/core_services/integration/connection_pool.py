"""
连接池管理

提供服务调用的连接池管理功能。
"""

import logging
import threading
from typing import Dict, Any
from queue import Queue

from src.core.constants import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class ConnectionPool:
    """连接池管理"""

    def __init__(self, pool_size: int = DEFAULT_BATCH_SIZE, service_name: str = ""):
        self.pool_size = pool_size
        self.service_name = service_name
        self._connections = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created_count = 0

        # 预创建连接
        for _ in range(pool_size):
            self._create_connection()

    def _create_connection(self):
        """创建新连接"""
        try:
            # 这里可以实现实际的连接创建逻辑
            connection = f"connection_{self.service_name}_{self._created_count}"
            self._created_count += 1
            self._connections.put(connection, block=False)
        except Exception as e:
            logger.error(f"创建连接失败: {e}")

    def get_connection(self, timeout: float = 5.0):
        """获取连接"""
        try:
            return self._connections.get(timeout=timeout)
        except Exception as e:
            logger.warning(f"获取连接超时: {e}")
            return None

    def return_connection(self, connection):
        """归还连接"""
        try:
            if connection and self._connections.qsize() < self.pool_size:
                self._connections.put(connection, block=False)
        except Exception as e:
            logger.error(f"归还连接失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        return {
            'pool_size': self.pool_size,
            'active_connections': self.pool_size - self._connections.qsize(),
            'idle_connections': self._connections.qsize(),
            'created_count': self._created_count
        }


class ConnectionPoolManager:
    """连接池管理器 - 职责：管理所有服务的连接池"""

    def __init__(self):
        self._connection_pools: Dict[str, ConnectionPool] = {}

    def get_connection_pool(self, service_name: str,
                           pool_size: int = DEFAULT_BATCH_SIZE) -> ConnectionPool:
        """获取或创建连接池"""
        if service_name not in self._connection_pools:
            self._connection_pools[service_name] = ConnectionPool(
                pool_size=pool_size,
                service_name=service_name
            )
        return self._connection_pools[service_name]

    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有连接池统计信息"""
        return {
            name: pool.get_stats()
            for name, pool in self._connection_pools.items()
        }

    def close_all_pools(self) -> None:
        """关闭所有连接池"""
        for pool in self._connection_pools.values():
            # 这里可以添加清理逻辑
            pass
        self._connection_pools.clear()
        logger.info("所有连接池已关闭")

