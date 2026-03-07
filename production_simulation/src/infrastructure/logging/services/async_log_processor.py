"""
async_log_processor 模块

提供 async_log_processor 相关功能和接口。
"""

import os
import json
import os

import glob
import secrets
import asyncio
import gzip
import queue
import threading
import time
import weakref

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步日志处理器
提供高性能的异步日志处理能力
"""


@dataclass
class AsyncLogEntry:
    """异步日志条目"""
    timestamp: float
    level: str
    logger_name: str
    message: str
    module: str = ""
    function: str = ""
    line: int = 0
    thread_id: int = 0
    process_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    formatted_message: str = ""


class AsyncLogBatch:
    """异步日志批次"""

    def __init__(self, entries: List[AsyncLogEntry], batch_id: str):
        self.entries = entries.copy()  # 复制列表以避免外部修改
        self.batch_id = batch_id
        self.created_at = time.time()
        self.processed_at: Optional[float] = None
        self.size_bytes = sum(len(str(entry).encode('utf-8')) for entry in entries)

    def add_entry(self, entry: AsyncLogEntry):
        """添加日志条目"""
        self.entries.append(entry)
        self.size_bytes += len(str(entry).encode('utf-8'))

    def mark_processed(self, processing_time: Optional[float] = None):
        """标记为已处理"""
        self.processed_at = time.time()
        if processing_time is not None:
            # 如果提供了处理时间，直接设置
            self.processed_at = self.created_at + processing_time

    def get_size(self) -> int:
        """获取批次大小（条目数量）"""
        return len(self.entries)

    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self.entries) == 0

    @property
    def processed(self) -> bool:
        """是否已处理"""
        return self.processed_at is not None

    @property
    def processing_time(self) -> float:
        """获取处理时间"""
        if self.processed_at:
            return self.processed_at - self.created_at
        return 0.0


class AsyncLogQueue:
    """异步日志队列"""

    def __init__(self, max_size: int = 10000, batch_size: int = 100,
                 flush_interval: float = 1.0):
        self.max_size = max_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # 队列和批次管理
        self.queue: queue.Queue = queue.Queue(maxsize=max_size)
        self.current_batch: List[AsyncLogEntry] = []
        self.batch_counter = 0

        # 同步原语
        self._lock = threading.RLock()
        self._flush_event = threading.Event()
        self._shutdown_event = threading.Event()

        # 工作线程
        self._worker_thread: Optional[threading.Thread] = None
        self._flush_timer: Optional[threading.Timer] = None

        # 统计信息
        self.stats = {
            'entries_processed': 0,
            'batches_processed': 0,
            'queue_full_drops': 0,
            'flush_operations': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0
        }

        # 处理器列表
        self.processors: List[Callable] = []

    def start(self):
        """启动异步处理"""
        if self._worker_thread is None:
            self._worker_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="AsyncLogProcessor"
            )
            self._worker_thread.start()

            # 启动定时刷新
            self._schedule_flush()

    def stop(self):
        """停止异步处理"""
        self._shutdown_event.set()
        self._flush_event.set()  # 唤醒处理线程

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

        if self._flush_timer:
            self._flush_timer.cancel()

    def add_processor(self, processor: Callable):
        """添加日志处理器"""
        with self._lock:
            self.processors.append(processor)

    def remove_processor(self, processor: Callable):
        """移除日志处理器"""
        with self._lock:
            if processor in self.processors:
                self.processors.remove(processor)

    def put(self, entry: AsyncLogEntry, block: bool = True,
            timeout: float = 1.0) -> bool:
        """添加日志条目"""
        try:
            self.queue.put(entry, block=block, timeout=timeout)

            # 触发可能的刷新 - 简化逻辑避免死锁
            self._flush_event.set()

            return True

        except queue.Full:
            with self._lock:
                self.stats['queue_full_drops'] += 1
            return False

    def enqueue(self, entry: AsyncLogEntry, block: bool = True,
                timeout: float = 1.0) -> bool:
        """入队日志条目（put的别名）"""
        return self.put(entry, block, timeout)

    def flush(self):
        """立即刷新当前批次"""
        with self._lock:
            if self.current_batch:
                batch_to_process = self.current_batch.copy()
                self.current_batch.clear()
                # Collect more entries from queue if available
                while len(self.current_batch) < self.batch_size:
                    try:
                        entry = self.queue.get_nowait()
                        self.current_batch.append(entry)
                    except queue.Empty:
                        break
                
                # Add any newly collected entries to the batch to process
                if self.current_batch:
                    batch_to_process.extend(self.current_batch)
                    self.current_batch.clear()
                
                self.stats['flush_operations'] += 1
                
                # Process outside the lock to avoid deadlock
                if batch_to_process:
                    self._process_batch_now(batch_to_process)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'queue_size': self.queue.qsize(),
                'current_batch_size': len(self.current_batch),
                'active_processors': len(self.processors)
            })
            return stats

    def _flush_batch(self):
        """刷新当前批次（内部方法）"""
        with self._lock:
            # Collect entries from queue to current_batch
            while len(self.current_batch) < self.batch_size:
                try:
                    # 非阻塞获取
                    entry = self.queue.get_nowait()
                    self.current_batch.append(entry)
                except queue.Empty:
                    break
            
            # Process the current batch if it has entries
            if self.current_batch:
                batch_to_process = self.current_batch.copy()
                self.current_batch.clear()
                self.stats['flush_operations'] += 1
                
                # Process outside the lock to avoid deadlock
                if batch_to_process:
                    self._process_batch_now(batch_to_process)

    def _processing_loop(self):
        """处理循环"""
        while not self._shutdown_event.is_set():
            try:
                # 收集并处理批次，避免死锁
                batch_to_process = None
                with self._lock:
                    # 从队列收集日志条目
                    while len(self.current_batch) < self.batch_size:
                        try:
                            entry = self.queue.get_nowait()
                            self.current_batch.append(entry)
                        except queue.Empty:
                            break
                    
                    # 准备要处理的批次
                    if self.current_batch:
                        batch_to_process = self.current_batch.copy()
                        self.current_batch.clear()

                # 在锁外处理批次，避免死锁
                if batch_to_process:
                    self._process_batch_now(batch_to_process)

                # 等待下一次处理
                self._flush_event.wait(timeout=self.flush_interval)
                self._flush_event.clear()

            except Exception as e:
                print(f"异步日志处理循环错误: {e}")
                time.sleep(1)

    def _collect_batch(self):
        """收集一批日志条目"""
        with self._lock:
            while len(self.current_batch) < self.batch_size:
                try:
                    # 非阻塞获取
                    entry = self.queue.get_nowait()
                    self.current_batch.append(entry)
                except queue.Empty:
                    break

    def _process_batch_now(self, batch: List[AsyncLogEntry]):
        """立即处理一批日志"""
        if not batch:
            return

        # 创建批次对象
        batch_id = f"batch_{self.batch_counter}"
        self.batch_counter += 1

        log_batch = AsyncLogBatch(batch, batch_id)

        try:
            # 并行处理批次
            for processor in self.processors:
                try:
                    processor(log_batch)
                except Exception as e:
                    print(f"日志处理器执行失败: {e}")

            # 标记为已处理
            log_batch.mark_processed()

            # 更新统计信息
            self.stats['entries_processed'] += len(batch)
            self.stats['batches_processed'] += 1
            self.stats['avg_batch_size'] = (
                (self.stats['avg_batch_size'] * (self.stats['batches_processed'] - 1))
                + len(batch)
            ) / self.stats['batches_processed']
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['batches_processed'] - 1))
                + log_batch.processing_time
            ) / self.stats['batches_processed']

        except Exception as e:
            print(f"批次处理失败: {e}")

    def _schedule_flush(self):
        """调度下一次刷新"""
        if self._flush_timer:
            self._flush_timer.cancel()

        self._flush_timer = threading.Timer(self.flush_interval, self._trigger_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _trigger_flush(self):
        """触发刷新"""
        self._flush_event.set()
        if not self._shutdown_event.is_set():
            self._schedule_flush()


class FileLogProcessor:
    """文件日志处理器"""

    def __init__(self, log_file: str, max_size: int = 100 * 1024 * 1024,
                 backup_count: int = 5, compress: bool = True):
        self.log_file = Path(log_file)
        self.max_size = max_size
        self.backup_count = backup_count
        self.compress = compress

        self.log_dir = self.log_file.parent
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._file_handle: Optional[object] = None
        self._current_size = 0

    def __call__(self, batch: AsyncLogBatch):
        """处理日志批次"""
        # 确保文件打开
        if self._file_handle is None:
            self._open_file()

        # 检查是否需要轮转
        if self._should_rotate(batch.size_bytes):
            self._rotate_file()

        # 写入批次
        for entry in batch.entries:
            line = self._format_entry(entry) + '\n'
            self._file_handle.write(line)
            self._current_size += len(line.encode('utf-8'))

        # 刷新到磁盘
        self._file_handle.flush()

    def process_batch(self, batch: AsyncLogBatch):
        """处理日志批次（兼容性方法）"""
        self(batch)

    def _open_file(self):
        """打开日志文件"""
        self._file_handle = open(self.log_file, 'a', encoding='utf-8')
        try:
            self._current_size = self.log_file.stat().st_size
        except OSError:
            self._current_size = 0

    def _should_rotate(self, new_data_size: int) -> bool:
        """检查是否需要轮转"""
        return self._current_size + new_data_size > self.max_size

    def _rotate_file(self):
        """轮转日志文件"""
        if self._file_handle:
            self._file_handle.close()

        # 重命名现有文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = self.log_file.with_suffix(f".{timestamp}{self.log_file.suffix}")

        if self.log_file.exists():
            self.log_file.rename(backup_file)

            # 压缩备份文件
            if self.compress:
                self._compress_file(backup_file)

        # 删除旧的备份
        self._cleanup_backups()

        # 重新打开文件
        self._open_file()

    def _compress_file(self, file_path: Path):
        """压缩文件"""
        compressed_file = file_path.with_suffix(f"{file_path.suffix}.gz")

        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)

            file_path.unlink()  # 删除原始文件
        except Exception as e:
            print(f"文件压缩失败: {e}")

    def _cleanup_backups(self):
        """清理旧备份"""
        pattern = str(self.log_file.with_suffix(".*.gz"))
        backup_files = glob.glob(pattern)

        # 按修改时间排序
        backup_files.sort(key=os.path.getmtime, reverse=True)

        # 删除多余的备份
        for old_file in backup_files[self.backup_count:]:
            try:
                os.remove(old_file)
            except OSError:
                pass

    def _format_entry(self, entry: AsyncLogEntry) -> str:
        """格式化日志条目"""
        if entry.formatted_message:
            return entry.formatted_message

        # JSON格式
        data = {
            "timestamp": entry.timestamp,
            "level": entry.level,
            "logger": entry.logger_name,
            "message": entry.message,
            "module": entry.module,
            "function": entry.function,
            "line": entry.line,
            "thread_id": entry.thread_id,
            "process_id": entry.process_id
        }

        if entry.metadata:
            data["metadata"] = entry.metadata

        return json.dumps(data, ensure_ascii=False)

    def close(self):
        """关闭处理器"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


class ConsoleLogProcessor:
    """控制台日志处理器"""

    def __init__(self, level_filter: Optional[str] = None):
        self.level_filter = level_filter

    def __call__(self, batch: AsyncLogBatch):
        """处理日志批次"""
        for entry in batch.entries:
            if self.level_filter and entry.level != self.level_filter:
                continue

            print(self._format_entry(entry))

    def process_batch(self, batch: AsyncLogBatch):
        """处理日志批次（兼容性方法）"""
        self(batch)

    def _format_entry(self, entry: AsyncLogEntry) -> str:
        """格式化日志条目"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(entry.timestamp))

        return f"[{timestamp}] {entry.level} {entry.logger_name}: {entry.message}"


# 全局异步日志队列实例
_global_async_queue = None
_queue_lock = threading.Lock()


def get_async_log_queue() -> AsyncLogQueue:
    """获取全局异步日志队列"""
    global _global_async_queue

    if _global_async_queue is None:
        with _queue_lock:
            if _global_async_queue is None:
                _global_async_queue = AsyncLogQueue()

    return _global_async_queue


def setup_async_logging(log_file: str = "logs/async.log"):
    """设置异步日志记录"""
    # 获取全局队列
    queue = get_async_log_queue()

    # 添加处理器
    file_processor = FileLogProcessor(log_file)
    console_processor = ConsoleLogProcessor()

    queue.add_processor(file_processor)
    queue.add_processor(console_processor)

    # 启动异步处理
    queue.start()

    return queue


def async_log(level: str, message: str, logger_name: str = "async",
              **kwargs):
    """异步记录日志"""
    queue = get_async_log_queue()

    # 创建日志条目
    entry = AsyncLogEntry(
        timestamp=time.time(),
        level=level,
        logger_name=logger_name,
        message=message,
        **kwargs
    )

    # 添加到队列
    queue.put(entry)


if __name__ == "__main__":
    # 测试异步日志处理器
    print("设置异步日志记录...")
    queue = setup_async_logging()

    print("开始异步日志测试...")

    # 生成测试日志
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

    for i in range(1000):
        level = secrets.choice(levels)
        async_log(
            level=level,
            message=f"测试日志消息 {i}",
            logger_name="test_logger",
            module="test_module",
            function="test_function",
            line=secrets.randint(1, 100)
        )

    print("等待日志处理...")
    time.sleep(5)

    # 获取统计信息
    stats = queue.get_stats()
    print(f"异步日志统计: {stats}")

    # 停止异步处理
    queue.stop()
    print("异步日志处理已停止")
