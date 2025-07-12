from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import time
from dataclasses import dataclass

@dataclass
class LoadResult:
    success: bool
    data: Any
    error: str = None
    duration: float = 0.0

class ParallelLoader:
    """并行数据加载引擎"""

    def __init__(self, max_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stats = {
            'total_tasks': 0,
            'success_tasks': 0,
            'total_time': 0.0
        }

    def batch_load(self, tasks: List[Tuple[str, Dict]]) -> Dict[str, LoadResult]:
        """批量并行加载数据"""
        results = {}
        futures = {}

        # 提交所有任务
        for task_id, config in tasks:
            future = self.executor.submit(self._load_single, task_id, config)
            futures[future] = task_id

        # 收集结果
        for future in as_completed(futures, timeout=30):
            task_id = futures[future]
            try:
                result = future.result()
                results[task_id] = result

                # 更新统计
                self.stats['total_tasks'] += 1
                if result.success:
                    self.stats['success_tasks'] += 1
                self.stats['total_time'] += result.duration
            except Exception as e:
                results[task_id] = LoadResult(
                    success=False,
                    data=None,
                    error=str(e),
                    duration=0.0
                )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_time = (self.stats['total_time'] / self.stats['total_tasks']) if self.stats['total_tasks'] > 0 else 0
        success_rate = (self.stats['success_tasks'] / self.stats['total_tasks']) * 100 if self.stats['total_tasks'] > 0 else 0

        return {
            **self.stats,
            'avg_time_ms': avg_time * 1000,
            'success_rate': f"{success_rate:.2f}%"
        }

    def _load_single(self, task_id: str, config: Dict) -> LoadResult:
        """单个数据加载任务"""
        start_time = time.time()
        try:
            # 这里应该调用实际的适配器加载数据
            # 模拟加载过程
            time.sleep(0.05)  # 模拟50ms加载时间
            data = {"id": task_id, **config}
            return LoadResult(
                success=True,
                data=data,
                duration=time.time() - start_time
            )
        except Exception as e:
            return LoadResult(
                success=False,
                data=None,
                error=str(e),
                duration=time.time() - start_time
            )
