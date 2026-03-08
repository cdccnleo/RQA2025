import os
import threading
import time
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Full
import psutil


@dataclass
class ThreadPoolConfig:

    """线程池配置项"""
    core_pool_size: int = max(2, os.cpu_count() or 2)
    max_pool_size: int = 50
    queue_capacity: int = 1000
    keep_alive: int = 60  # 秒
    submit_timeout: float = 30.0  # 提交超时时间
    enable_auto_adjust: bool = True  # 是否允许自动调整

    def __post_init__(self):
        """配置验证"""
        if self.core_pool_size <= 0:
            raise ValueError("core_pool_size must be positive")
        if self.max_pool_size <= 0:
            raise ValueError("max_pool_size must be positive")
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if self.core_pool_size > self.max_pool_size:
            raise ValueError("core_pool_size cannot be greater than max_pool_size")

    def adjust_based_on_load(self, current_load: float) -> None:
        """根据系统负载动态调整配置"""
        if not self.enable_auto_adjust:
            return
        cpu_count = os.cpu_count() or 2
        if current_load > 0.7:  # 高负载
            self.core_pool_size = max(2, int(cpu_count * 0.5))
            self.queue_capacity = 500
        elif current_load < 0.3:  # 低负载
            self.core_pool_size = min(
                self.max_pool_size,
                int(cpu_count * 2)
            )
            self.queue_capacity = 2000


class DynamicThreadPool:

    """动态线程池管理器"""

    def __init__(self, config: ThreadPoolConfig):

        self.config = config
        self.executor: Optional[ThreadPoolExecutor] = None
        self.task_queue = Queue(maxsize=config.queue_capacity)
        self.running = False
        if self.config.enable_auto_adjust:
            self.monitor_thread = threading.Thread(target=self._monitor_load, daemon=True)
        else:
            self.monitor_thread = None
        self._init_executor()

    def processor(self) -> None:
        """初始化线程池执行器"""
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_pool_size,
            thread_name_prefix='DataLoader_'
        )
        self.running = True
        if self.monitor_thread:
            self.monitor_thread.start()

    def _monitor_load(self) -> None:
        """监控系统负载并动态调整"""
        while self.running:
            try:
                # 获取系统负载 (1分钟平均)
                cpu_count = os.cpu_count() or 2
                load = psutil.getloadavg()[0] / cpu_count

                # 动态调整配置
                old_size = self.config.core_pool_size
                self.config.adjust_based_on_load(load)

                # 如果核心线程数变化，调整线程池
                if old_size != self.config.core_pool_size and self.executor:
                    # 注意：ThreadPoolExecutor不支持运行时调整max_workers
                    # 这里只是记录配置变化，实际调整需要重新创建executor
                    pass

                time.sleep(5)  # 5秒监控间隔
            except Exception:
                # 监控线程异常不应该影响主程序
                time.sleep(5)

    def submit(self, fn, *args, **kwargs) -> Future:
        """提交任务到线程池 - 修复死锁问题"""
        if not self.running or not self.executor:
            raise RuntimeError("Thread pool not running")

        # 使用超时机制避免死锁
        start_time = time.time()
        timeout = self.config.submit_timeout

        while self.task_queue.full():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"提交任务超时: 队列已满 {timeout} 秒")
            time.sleep(0.01)  # 减少等待间隔

        try:
            future = self.executor.submit(fn, *args, **kwargs)
            # 使用非阻塞方式添加到队列
            try:
                self.task_queue.put_nowait(future)
            except Full:
                # 队列满时，直接返回future，不等待
                pass
            return future
        except Exception as e:
            raise RuntimeError(f"提交任务失败: {e}")

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
            'active_threads': getattr(self.executor, '_num_threads', 0),
            'pending_tasks': self.task_queue.qsize(),
            'core_pool_size': self.config.core_pool_size,
            'max_pool_size': self.config.max_pool_size,
            'queue_capacity': self.config.queue_capacity
        }


def create_default_pool() -> DynamicThreadPool:
    """创建默认配置的线程池"""
    cpu_count = os.cpu_count() or 2
    # core_pool_size不大于max_pool_size
    config = ThreadPoolConfig(
        core_pool_size=min(cpu_count, 50),
        max_pool_size=50,
        queue_capacity=1000,
        keep_alive=60,
        enable_auto_adjust=True
    )
    return DynamicThreadPool(config)
