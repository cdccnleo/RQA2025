"""
IO Optimization Module
IO优化模块

This module provides IO optimization capabilities for quantitative trading systems
此模块为量化交易系统提供IO优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Union
import threading
import time
import os
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import mmap

logger = logging.getLogger(__name__)


class IOOptimizer:

    """
    IO Optimizer Class
    IO优化器类

    Provides IO operation optimization including file I / O, disk I / O, and data transfer
    提供IO操作优化，包括文件IO、磁盘IO和数据传输
    """

    def __init__(self, buffer_size: int = 8192, max_workers: int = 4):
        """
        Initialize IO optimizer
        初始化IO优化器

        Args:
            buffer_size: Default buffer size for IO operations
                        IO操作的默认缓冲区大小
            max_workers: Maximum number of worker threads
                        最大工作线程数
        """
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.io_stats = {
            'total_reads': 0,
            'total_writes': 0,
            'total_bytes_read': 0,
            'total_bytes_written': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_read_time': 0.0,
            'average_write_time': 0.0
        }
        self.is_async_enabled = True

        # IO optimization settings
        self.enable_buffering = True
        self.enable_caching = True
        self.enable_compression = False
        self.max_cache_size = 100 * 1024 * 1024  # 100MB
        self.current_cache_size = 0

        logger.info("IO optimizer initialized")

    def read_file_optimized(self,


                            file_path: str,
                            use_cache: bool = True,
                            use_async: bool = True) -> Union[str, bytes]:
        """
        Optimized file reading
        优化的文件读取

        Args:
            file_path: Path to the file to read
                      要读取的文件路径
            use_cache: Whether to use cache
                      是否使用缓存
            use_async: Whether to use async IO
                      是否使用异步IO

        Returns:
            File content as string or bytes
            文件内容作为字符串或字节
        """
        start_time = time.time()

        try:
            # Check cache first
            if use_cache and self.enable_caching:
                cached_content = self._get_from_cache(file_path)
                if cached_content is not None:
                    self.io_stats['cache_hits'] += 1
                    read_time = time.time() - start_time
                    self._update_read_stats(read_time, len(cached_content))
                    return cached_content

            self.io_stats['cache_misses'] += 1

            if use_async and self.is_async_enabled:
                # Async reading
                content = asyncio.run(self._read_file_async(file_path))
            else:
                # Synchronous reading with optimization
                content = self._read_file_sync_optimized(file_path)

            # Cache the content
            if use_cache and self.enable_caching:
                self._add_to_cache(file_path, content)

            read_time = time.time() - start_time
            self._update_read_stats(read_time, len(content))

            return content

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            raise

    def write_file_optimized(self,


                             file_path: str,
                             content: Union[str, bytes],
                             use_buffering: bool = True,
                             use_async: bool = True) -> bool:
        """
        Optimized file writing
        优化的文件写入

        Args:
            file_path: Path to the file to write
                      要写入的文件路径
            content: Content to write
                    要写入的内容
            use_buffering: Whether to use buffering
                          是否使用缓冲
            use_async: Whether to use async IO
                      是否使用异步IO

        Returns:
            bool: True if successful, False otherwise
                  成功返回True，否则返回False
        """
        start_time = time.time()

        try:
            if use_async and self.is_async_enabled:
                # Async writing
                success = asyncio.run(self._write_file_async(file_path, content))
            else:
                # Synchronous writing with optimization
                success = self._write_file_sync_optimized(file_path, content)

            if success:
                write_time = time.time() - start_time
                self._update_write_stats(write_time, len(content))

                # Invalidate cache if file was cached
                self._invalidate_cache(file_path)

            return success

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {str(e)}")
            return False

    async def _read_file_async(self, file_path: str) -> Union[str, bytes]:
        """
        Asynchronous file reading
        异步文件读取

        Args:
            file_path: Path to the file
                      文件路径

        Returns:
            File content
            文件内容
        """
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()

    async def _write_file_async(self, file_path: str, content: Union[str, bytes]) -> bool:
        """
        Asynchronous file writing
        异步文件写入

        Args:
            file_path: Path to the file
                      文件路径
            content: Content to write
                    要写入的内容

        Returns:
            bool: True if successful
                  成功返回True
        """
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                if isinstance(content, str):
                    content = content.encode('utf-8')
                await f.write(content)
            return True
        except Exception:
            return False

    def _read_file_sync_optimized(self, file_path: str) -> Union[str, bytes]:
        """
        Synchronous optimized file reading
        同步优化的文件读取

        Args:
            file_path: Path to the file
                      文件路径

        Returns:
            File content
            文件内容
        """
        file_size = os.path.getsize(file_path)

        # Use memory mapping for large files
        if file_size > self.buffer_size * 10:  # Larger than 80KB
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.read()
        else:
            # Use buffered reading for smaller files
            with open(file_path, 'rb', buffering=self.buffer_size) as f:
                return f.read()

    def _write_file_sync_optimized(self, file_path: str, content: Union[str, bytes]) -> bool:
        """
        Synchronous optimized file writing
        同步优化的文件写入

        Args:
            file_path: Path to the file
                      文件路径
            content: Content to write
                    要写入的内容

        Returns:
            bool: True if successful
                  成功返回True
        """
        try:
            # Ensure content is bytes
            if isinstance(content, str):
                content = content.encode('utf-8')

            # Use buffered writing
            with open(file_path, 'wb', buffering=self.buffer_size) as f:
                f.write(content)

            return True

        except Exception:
            return False

    def _get_from_cache(self, file_path: str) -> Optional[Union[str, bytes]]:
        """
        Get content from cache
        从缓存中获取内容

        Args:
            file_path: File path
                      文件路径

        Returns:
            Cached content or None
            缓存的内容或None
        """
        with self.cache_lock:
            return self.cache.get(file_path)

    def _add_to_cache(self, file_path: str, content: Union[str, bytes]) -> None:
        """
        Add content to cache
        将内容添加到缓存

        Args:
            file_path: File path
                      文件路径
            content: Content to cache
                    要缓存的内容
        """
        content_size = len(content) if isinstance(content, (str, bytes)) else 0

        with self.cache_lock:
            # Check if adding this content would exceed cache size
            if self.current_cache_size + content_size > self.max_cache_size:
                self._evict_cache_entries(content_size)

            self.cache[file_path] = content
            self.current_cache_size += content_size

    def _invalidate_cache(self, file_path: str) -> None:
        """
        Invalidate cache entry for a file
        使文件的缓存条目无效

        Args:
            file_path: File path
                      文件路径
        """
        with self.cache_lock:
            if file_path in self.cache:
                content_size = len(self.cache[file_path])
                del self.cache[file_path]
                self.current_cache_size -= content_size

    def _evict_cache_entries(self, required_size: int) -> None:
        """
        Evict cache entries to make room for new content
        清除缓存条目为新内容腾出空间

        Args:
            required_size: Size required for new content
                          新内容所需的大小
        """
        # Simple LRU eviction - remove oldest entries
        entries_to_remove = []
        total_freed = 0

        for file_path, content in list(self.cache.items()):
            content_size = len(content)
            entries_to_remove.append((file_path, content_size))
            total_freed += content_size

            if total_freed >= required_size:
                break

        for file_path, _ in entries_to_remove:
            del self.cache[file_path]

        self.current_cache_size -= total_freed

    def _update_read_stats(self, read_time: float, bytes_read: int) -> None:
        """
        Update read statistics
        更新读取统计信息

        Args:
            read_time: Time taken for read operation
                      读取操作所用时间
            bytes_read: Number of bytes read
                       读取的字节数
        """
        self.io_stats['total_reads'] += 1
        self.io_stats['total_bytes_read'] += bytes_read

        # Update average read time
        total_reads = self.io_stats['total_reads']
        current_avg = self.io_stats['average_read_time']
        self.io_stats['average_read_time'] = (
            (current_avg * (total_reads - 1)) + read_time
        ) / total_reads

    def _update_write_stats(self, write_time: float, bytes_written: int) -> None:
        """
        Update write statistics
        更新写入统计信息

        Args:
            write_time: Time taken for write operation
                       写入操作所用时间
            bytes_written: Number of bytes written
                          写入的字节数
        """
        self.io_stats['total_writes'] += 1
        self.io_stats['total_bytes_written'] += bytes_written

        # Update average write time
        total_writes = self.io_stats['total_writes']
        current_avg = self.io_stats['average_write_time']
        self.io_stats['average_write_time'] = (
            (current_avg * (total_writes - 1)) + write_time
        ) / total_writes

    def get_disk_io_stats(self) -> Dict[str, Any]:
        """
        Get disk IO statistics
        获取磁盘IO统计信息

        Returns:
            dict: Disk IO statistics
                  磁盘IO统计信息
        """
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get disk IO stats: {str(e)}")
            return {}

    def optimize_file_operations(self,


                                 operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize a batch of file operations
        优化一批文件操作

        Args:
            operations: List of file operations
                       文件操作列表

        Returns:
            dict: Optimization results
                  优化结果
        """
        try:
            # Group operations by type
            reads = [op for op in operations if op.get('type') == 'read']
            writes = [op for op in operations if op.get('type') == 'write']

            results = {
                'total_operations': len(operations),
                'read_operations': len(reads),
                'write_operations': len(writes),
                'optimized_operations': []
            }

            # Optimize read operations
            for read_op in reads:
                file_path = read_op.get('file_path')
                if file_path:
                    content = self.read_file_optimized(
                        file_path,
                        use_cache=read_op.get('use_cache', True),
                        use_async=read_op.get('use_async', True)
                    )
                    results['optimized_operations'].append({
                        'operation': read_op,
                        'result': content,
                        'status': 'success'
                    })

            # Optimize write operations
            for write_op in writes:
                file_path = write_op.get('file_path')
                content = write_op.get('content')
                if file_path and content is not None:
                    success = self.write_file_optimized(
                        file_path,
                        content,
                        use_buffering=write_op.get('use_buffering', True),
                        use_async=write_op.get('use_async', True)
                    )
                    results['optimized_operations'].append({
                        'operation': write_op,
                        'result': success,
                        'status': 'success' if success else 'failed'
                    })

            return results

        except Exception as e:
            logger.error(f"Failed to optimize file operations: {str(e)}")
            return {'error': str(e)}

    def clear_cache(self) -> None:
        """
        Clear the IO cache
        清除IO缓存
        """
        with self.cache_lock:
            self.cache.clear()
            self.current_cache_size = 0
        logger.info("IO cache cleared")

    def get_io_optimizer_status(self) -> Dict[str, Any]:
        """
        Get IO optimizer status
        获取IO优化器状态

        Returns:
            dict: Optimizer status information
                  优化器状态信息
        """
        return {
            'buffer_size': self.buffer_size,
            'max_workers': self.max_workers,
            'enable_buffering': self.enable_buffering,
            'enable_caching': self.enable_caching,
            'enable_compression': self.enable_compression,
            'max_cache_size': self.max_cache_size,
            'current_cache_size': self.current_cache_size,
            'cache_size_mb': self.current_cache_size / (1024 * 1024),
            'is_async_enabled': self.is_async_enabled,
            'io_stats': self.io_stats,
            'disk_io_stats': self.get_disk_io_stats(),
            'cache_hit_rate': (
                self.io_stats['cache_hits'] /
                max(self.io_stats['cache_hits'] + self.io_stats['cache_misses'], 1) * 100
            )
        }


# Global IO optimizer instance
# 全局IO优化器实例
io_optimizer = IOOptimizer()

__all__ = ['IOOptimizer', 'io_optimizer']
