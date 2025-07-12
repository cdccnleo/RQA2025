import os
import threading
import time
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
import psutil

@dataclass
class ThreadPoolConfig:
    """线程池配置项"""
    core_pool_size: int = os.cpu_count() or 2
    max_pool_size: int = 50
    queue_capacity: int = 1000
    keep_alive: int = 60  # 秒

    def adjust_based_on_load(self, current_load: float) -> None:
        """根据系统负载动态调整配置"""
        if current_load > 0.7:  # 高负载
            self.core_pool_size = max(2, int(os.cpu_count() * 0.5))
            self.queue_capacity = 500
        elif current_load < 0.3:  # 低负载
            self.core_pool_size = min(
                self.max_pool_size,
                int(os.cpu_count() * 2)
            )
            self.queue_capacity = 2000

class DynamicThreadPool:
    """动态线程池管理器"""

    def __init__(self, config: ThreadPoolConfig):
        self.config = config
        self.executor: Optional[ThreadPoolExecutor] = None
        self.task_queue = Queue(maxsize=config.queue_capacity)
        self.monitor_thread = threading.Thread(
            target=self._monitor_load,
            daemon=True
        )
        self.running = False
        self._init_executor()

    def _init_executor(self) -> None:
        """初始化线程池执行器"""
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_pool_size,
            thread_name_prefix='DataLoader_'
        )
        self.running = True
        self.monitor_thread.start()

    def _monitor_load(self) -> None:
        """监控系统负载并动态调整"""
        while self.running:
            # 获取系统负载 (1分钟平均)
            load = psutil.getloadavg()[0] / os.cpu_count()

            # 动态调整配置
            old_size = self.config.core_pool_size
            self.config.adjust_based_on_load(load)

            # 如果核心线程数变化，调整线程池
            if old_size != self.config.core_pool_size:
                self.executor._max_workers = self.config.max_pool_size
                self.executor._core_threads = self.config.core_pool_size

            time.sleep(5)  # 5秒监控间隔

    def submit(self, fn, *args, **kwargs) -> Future:
        """提交任务到线程池"""
        if not self.running or not self.executor:
            raise RuntimeError("Thread pool not running")

        # 队列满时等待
        while self.task_queue.full():
            time.sleep(0.1)

        future = self.executor.submit(fn, *args, **kwargs)
        self.task_queue.put(future)
        return future

    def shutdown(self, wait=True) -> None:
        """关闭线程池"""
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=wait)

    def get_stats(self) -> dict:
        """获取线程池统计信息"""
        if not self.executor:
            return {}

        return {
            'active_threads': self.executor._num_threads,
            'pending_tasks': self.task_queue.qsize(),
            'core_pool_size': self.config.core_pool_size,
            'max_pool_size': self.config.max_pool_size,
            'queue_capacity': self.config.queue_capacity
        }

def create_default_pool() -> DynamicThreadPool:
    """创建默认配置的线程池"""
    config = ThreadPoolConfig(
        core_pool_size=os.cpu_count() * 2,
        max_pool_size=50,
        queue_capacity=1000,
        keep_alive=60
    )
    return DynamicThreadPool(config)
