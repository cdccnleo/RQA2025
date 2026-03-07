#!/usr/bin/env python3
"""
异步处理队列

提供异步任务处理能力，提升系统并发性能
"""

import threading
import queue
import time
from typing import Callable
from concurrent.futures import ThreadPoolExecutor


class AsyncProcessor:
    """异步处理器"""

    def __init__(self, max_workers=10, queue_size=1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.worker_thread = None

        # 性能统计
        self.processed_tasks = 0
        self.failed_tasks = 0
        self.avg_processing_time = 0

    def start(self):
        """启动异步处理器"""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        print(f"🚀 异步处理器已启动，工作者数量: {self.max_workers}")

    def stop(self):
        """停止异步处理器"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        print("🛑 异步处理器已停止")

    def submit_task(self, func: Callable, *args, **kwargs):
        """提交任务到异步队列"""
        try:
            task = {
                'func': func,
                'args': args,
                'kwargs': kwargs,
                'submitted_at': time.time()
            }
            self.task_queue.put(task, timeout=1)
            return True
        except queue.Full:
            print("⚠️ 任务队列已满，拒绝新任务")
            return False

    def _process_queue(self):
        """处理任务队列"""
        while self.is_running:
            try:
                # 获取任务
                task = self.task_queue.get(timeout=1)

                # 提交到线程池执行
                future = self.executor.submit(self._execute_task, task)
                future.add_done_callback(self._task_completed)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 队列处理错误: {e}")

    def _execute_task(self, task):
        """执行单个任务"""
        start_time = time.time()

        try:
            result = task['func'](*task['args'], **task['kwargs'])
            processing_time = time.time() - start_time

            return {
                'success': True,
                'result': result,
                'processing_time': processing_time
            }

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }

    def _task_completed(self, future):
        """任务完成回调"""
        try:
            result = future.result()

            if result['success']:
                self.processed_tasks += 1
            else:
                self.failed_tasks += 1

            # 更新平均处理时间
            total_tasks = self.processed_tasks + self.failed_tasks
            self.avg_processing_time = (
                (self.avg_processing_time * (total_tasks - 1)) + result['processing_time']
            ) / total_tasks

        except Exception as e:
            print(f"❌ 任务完成处理错误: {e}")

    def get_stats(self):
        """获取处理器统计信息"""
        return {
            'is_running': self.is_running,
            'queue_size': self.task_queue.qsize(),
            'max_queue_size': self.queue_size,
            'processed_tasks': self.processed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.processed_tasks / max(self.processed_tasks + self.failed_tasks, 1),
            'avg_processing_time': self.avg_processing_time,
            'active_threads': threading.active_count()
        }


# 全局异步处理器实例
_async_processor = None


def get_async_processor():
    """获取全局异步处理器实例"""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncProcessor()
    return _async_processor


def submit_async_task(func: Callable, *args, **kwargs):
    """提交异步任务的便捷函数"""
    processor = get_async_processor()
    if not processor.is_running:
        processor.start()
    return processor.submit_task(func, *args, **kwargs)
