# src/data/parallel/dynamic_executor.py
import concurrent.futures
import time
import numpy as np
from typing import Callable, List, Any


class DynamicExecutor:
    def __init__(self, initial_workers=4, max_workers=16):
        self.workers = initial_workers
        self.max_workers = max_workers
        self.performance_metrics = []
        self.last_adjust_time = time.time()

    def execute_batch(self, tasks: List[Callable], timeout: float = 30.0) -> List[Any]:
        """执行批量任务"""
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(task): task for task in tasks}
            results = []

            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(None)
                    # 错误处理...

        # 记录性能指标
        duration = time.time() - start_time
        self._record_metrics(len(tasks), duration)

        # 动态调整worker数量
        if time.time() - self.last_adjust_time > 60:  # 每分钟调整一次
            self._adjust_workers()

        return results

    def _record_metrics(self, task_count: int, duration: float):
        """记录性能指标"""
        self.performance_metrics.append({
            'timestamp': time.time(),
            'tasks': task_count,
            'workers': self.workers,
            'duration': duration,
            'throughput': task_count / duration if duration > 0 else 0
        })

        # 保留最近10条记录
        if len(self.performance_metrics) > 10:
            self.performance_metrics.pop(0)

    def _adjust_workers(self):
        """动态调整worker数量"""
        if len(self.performance_metrics) < 3:
            return

        # 计算最近平均吞吐量
        avg_throughput = np.mean([m['throughput'] for m in self.performance_metrics[-3:]])

        # 调整策略
        if avg_throughput > self.workers * 10:  # 每个worker处理10个任务/秒
            self.workers = min(self.workers + 2, self.max_workers)
        elif avg_throughput < self.workers * 5:  # 利用率不足
            self.workers = max(self.workers - 1, 1)

        self.last_adjust_time = time.time()