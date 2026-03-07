#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
MiniQMT连接池管理器
实现连接复用、自动管理和故障恢复
"""

import time
import threading
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class ConnectionType(Enum):

    """连接类型"""
    DATA = "data"      # 数据连接
    business = "business"    # 交易连接


class ConnectionStatus(Enum):

    """连接状态"""
    IDLE = "idle"          # 空闲
    BUSY = "busy"          # 忙碌
    ERROR = "error"         # 错误
    CLOSED = "closed"       # 已关闭


@dataclass
class ConnectionInfo:

    """连接信息"""
    connection_id: str
    connection_type: ConnectionType
    status: ConnectionStatus
    created_time: float
    last_used_time: float
    error_count: int = 0
    max_retries: int = 3


class ConnectionPool:

    """MiniQMT连接池管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化连接池

        Args:
            config: 连接池配置
        """
        self.config = config
        self.max_connections = config.get('max_connections', 10)
        self.min_connections = config.get('min_connections', 2)
        self.connection_timeout = config.get('connection_timeout', 30)
        self.idle_timeout = config.get('idle_timeout', 300)
        self.max_lifetime = config.get('max_lifetime', 3600)

        # 连接池
        self._data_pool: Dict[str, ConnectionInfo] = {}
        self._trade_pool: Dict[str, ConnectionInfo] = {}

        # 连接队列
        self._data_queue = Queue()
        self._trade_queue = Queue()

        # 锁和状态
        self._lock = threading.RLock()
        self._running = False
        self._cleanup_thread = None

        # 统计信息
        self._stats = {
            'total_connections': 0,
            'active_connections': 0,
            'idle_connections': 0,
            'error_connections': 0,
            'connection_requests': 0,
            'connection_timeouts': 0
        }

    def start(self):
        """启动连接池"""
        if self._running:
            return

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="ConnectionPool - Cleanup"
        )
        self._cleanup_thread.start()
        logger.info("MiniQMT连接池已启动")

    def stop(self):
        """停止连接池"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

        # 关闭所有连接
        with self._lock:
            self._close_all_connections()
        logger.info("MiniQMT连接池已停止")

    def get_connection(self, connection_type: ConnectionType, timeout: float = 10.0) -> Optional[str]:
        """
        获取连接

        Args:
            connection_type: 连接类型
            timeout: 超时时间

        Returns:
            连接ID，如果获取失败返回None
        """
        with self._lock:
            self._stats['connection_requests'] += 1

            # 选择对应的连接池
            pool = self._get_pool(connection_type)
            queue = self._get_queue(connection_type)

            # 尝试从队列获取空闲连接
            try:
                connection_id = queue.get(timeout=timeout)
                if connection_id in pool:
                    conn_info = pool[connection_id]
                    if conn_info.status == ConnectionStatus.IDLE:
                        conn_info.status = ConnectionStatus.BUSY
                        conn_info.last_used_time = time.time()
                        self._stats['active_connections'] += 1
                        self._stats['idle_connections'] -= 1
                        logger.debug(f"复用连接: {connection_id}")
                        return connection_id
            except Empty:
                pass

            # 创建新连接
            if self._can_create_connection(connection_type):
                connection_id = self._create_connection(connection_type)
                if connection_id:
                    return connection_id

            # 等待连接释放
            try:
                connection_id = queue.get(timeout=timeout)
                if connection_id in pool:
                    conn_info = pool[connection_id]
                    conn_info.status = ConnectionStatus.BUSY
                    conn_info.last_used_time = time.time()
                    self._stats['active_connections'] += 1
                    return connection_id
            except Empty:
                self._stats['connection_timeouts'] += 1
                logger.warning(f"获取{connection_type.value}连接超时")
                return None

    def release_connection(self, connection_id: str, connection_type: ConnectionType):
        """
        释放连接

        Args:
            connection_id: 连接ID
            connection_type: 连接类型
        """
        with self._lock:
            pool = self._get_pool(connection_type)
            queue = self._get_queue(connection_type)

            if connection_id in pool:
                conn_info = pool[connection_id]
                if conn_info.status == ConnectionStatus.BUSY:
                    conn_info.status = ConnectionStatus.IDLE
                    conn_info.last_used_time = time.time()
                    self._stats['active_connections'] -= 1
                    self._stats['idle_connections'] += 1

                    # 放回队列
                    queue.put(connection_id)
                    logger.debug(f"释放连接: {connection_id}")

    def mark_connection_error(self, connection_id: str, connection_type: ConnectionType):
        """
        标记连接错误

        Args:
            connection_id: 连接ID
            connection_type: 连接类型
        """
        with self._lock:
            pool = self._get_pool(connection_type)

            if connection_id in pool:
                conn_info = pool[connection_id]
                conn_info.status = ConnectionStatus.ERROR
                conn_info.error_count += 1
                self._stats['error_connections'] += 1

                # 如果错误次数超过限制，关闭连接
                if conn_info.error_count >= conn_info.max_retries:
                    self._close_connection(connection_id, connection_type)
                else:
                    # 尝试重新初始化连接
                    self._reinitialize_connection(connection_id, connection_type)

                logger.warning(f"连接错误: {connection_id}, 错误次数: {conn_info.error_count}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self._lock:
            return {
                'total_connections': self._stats['total_connections'],
                'active_connections': self._stats['active_connections'],
                'idle_connections': self._stats['idle_connections'],
                'error_connections': self._stats['error_connections'],
                'connection_requests': self._stats['connection_requests'],
                'connection_timeouts': self._stats['connection_timeouts'],
                'data_connections': len(self._data_pool),
                'trade_connections': len(self._trade_pool),
                'data_queue_size': self._data_queue.qsize(),
                'trade_queue_size': self._trade_queue.qsize()
            }

    def _get_pool(self, connection_type: ConnectionType) -> Dict[str, ConnectionInfo]:
        """获取对应的连接池"""
        if connection_type == ConnectionType.DATA:
            return self._data_pool
        else:
            return self._trade_pool

    def _get_queue(self, connection_type: ConnectionType) -> Queue:
        """获取对应的连接队列"""
        if connection_type == ConnectionType.DATA:
            return self._data_queue
        else:
            return self._trade_queue

    def _can_create_connection(self, connection_type: ConnectionType) -> bool:
        """检查是否可以创建新连接"""
        pool = self._get_pool(connection_type)
        return len(pool) < self.max_connections

    def _create_connection(self, connection_type: ConnectionType) -> Optional[str]:
        """创建新连接"""
        try:
            connection_id = f"{connection_type.value}_{int(time.time() * 1000)}"
            conn_info = ConnectionInfo(
                connection_id=connection_id,
                connection_type=connection_type,
                status=ConnectionStatus.BUSY,
                created_time=time.time(),
                last_used_time=time.time()
            )

            pool = self._get_pool(connection_type)
            pool[connection_id] = conn_info

            self._stats['total_connections'] += 1
            self._stats['active_connections'] += 1

            logger.info(f"创建新连接: {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"创建连接失败: {e}")
            return None

    def _close_connection(self, connection_id: str, connection_type: ConnectionType):
        """关闭连接"""
        pool = self._get_pool(connection_type)
        if connection_id in pool:
            conn_info = pool[connection_id]
            conn_info.status = ConnectionStatus.CLOSED

            if conn_info.status == ConnectionStatus.BUSY:
                self._stats['active_connections'] -= 1
            elif conn_info.status == ConnectionStatus.IDLE:
                self._stats['idle_connections'] -= 1

            del pool[connection_id]
            logger.info(f"关闭连接: {connection_id}")

    def _reinitialize_connection(self, connection_id: str, connection_type: ConnectionType):
        """重新初始化连接"""
        try:
            # 这里应该调用具体的连接初始化逻辑
            pool = self._get_pool(connection_type)
            if connection_id in pool:
                conn_info = pool[connection_id]
                conn_info.status = ConnectionStatus.IDLE
                conn_info.last_used_time = time.time()

                queue = self._get_queue(connection_type)
                queue.put(connection_id)

                logger.info(f"重新初始化连接: {connection_id}")

        except Exception as e:
            logger.error(f"重新初始化连接失败: {connection_id}, {e}")
            self._close_connection(connection_id, connection_type)

    def _close_all_connections(self):
        """关闭所有连接"""
        for pool in [self._data_pool, self._trade_pool]:
            for connection_id in list(pool.keys()):
                conn_info = pool[connection_id]
                conn_info.status = ConnectionStatus.CLOSED

        self._data_pool.clear()
        self._trade_pool.clear()
        self._stats['active_connections'] = 0
        self._stats['idle_connections'] = 0

    def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                self._cleanup_expired_connections()
                time.sleep(30)  # 每30秒清理一次
            except Exception as e:
                logger.error(f"连接池清理异常: {e}")

    def _cleanup_expired_connections(self):
        """清理过期连接"""
        current_time = time.time()

        with self._lock:
            for pool_name, pool in [('data', self._data_pool), ("business", self._trade_pool)]:
                for connection_id in list(pool.keys()):
                    conn_info = pool[connection_id]

                    # 检查连接生命周期
                    if current_time - conn_info.created_time > self.max_lifetime:
                        logger.info(f"清理过期连接: {connection_id}")
                        self._close_connection(connection_id, conn_info.connection_type)
                        continue

                    # 检查空闲超时
                    if (conn_info.status == ConnectionStatus.IDLE
                            and current_time - conn_info.last_used_time > self.idle_timeout):
                        logger.info(f"清理空闲连接: {connection_id}")
                        self._close_connection(connection_id, conn_info.connection_type)
                        continue

                    # 检查错误连接
                    if (conn_info.status == ConnectionStatus.ERROR
                            and conn_info.error_count >= conn_info.max_retries):
                        logger.info(f"清理错误连接: {connection_id}")
                        self._close_connection(connection_id, conn_info.connection_type)

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
