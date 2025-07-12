#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .core import QuoteStorage, StorageAdapter
from .adapters.file_system import (
    FileSystemAdapter,
    AShareFileSystemAdapter
)
from .adapters.database import (
    DatabaseAdapter,
    AShareDatabaseAdapter
)

__all__ = [
    'QuoteStorage',
    'StorageAdapter',
    'FileSystemAdapter',
    'AShareFileSystemAdapter',
    'DatabaseAdapter',
    'AShareDatabaseAdapter'
]

class StorageMonitor:
    """存储模块监控组件"""

    def __init__(self):
        self._metrics = {
            'write_count': 0,
            'read_count': 0,
            'error_count': 0,
            'avg_write_latency': 0.0,
            'avg_read_latency': 0.0
        }
        self._lock = threading.RLock()

    def record_write(self, size: int, latency: float):
        """记录写入操作"""
        with self._lock:
            self._metrics['write_count'] += 1
            # 计算平均延迟(指数移动平均)
            self._metrics['avg_write_latency'] = (
                0.8 * self._metrics['avg_write_latency'] +
                0.2 * latency
            )

    def record_read(self, latency: float):
        """记录读取操作"""
        with self._lock:
            self._metrics['read_count'] += 1
            self._metrics['avg_read_latency'] = (
                0.8 * self._metrics['avg_read_latency'] +
                0.2 * latency
            )

    def record_error(self):
        """记录错误发生"""
        with self._lock:
            self._metrics['error_count'] += 1

    def get_metrics(self) -> dict:
        """获取当前监控指标"""
        return self._metrics.copy()


def create_storage(adapter_type: str = 'file', **kwargs) -> QuoteStorage:
    """
    创建存储实例的工厂方法

    Args:
        adapter_type: 适配器类型(file/db)
        **kwargs: 适配器配置参数

    Returns:
        QuoteStorage实例
    """
    if adapter_type == 'file':
        adapter = AShareFileSystemAdapter(
            base_path=kwargs.get('base_path', 'data/storage')
        )
    elif adapter_type == 'db':
        adapter = AShareDatabaseAdapter(
            connection_pool=kwargs['connection_pool']
        )
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    return QuoteStorage(adapter)
