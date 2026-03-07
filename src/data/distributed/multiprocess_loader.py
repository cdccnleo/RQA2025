from typing import List, Dict, Any, Callable
from multiprocessing import Pool, cpu_count
from ...interfaces import IDistributedDataLoader


class MultiprocessDataLoader(IDistributedDataLoader):

    """
    基于多进程的分布式数据加载器
    """

    def __init__(self, worker_fn: Callable[[Dict[str, Any]], Any], num_workers: int = None):

        self.worker_fn = worker_fn
        self.num_workers = num_workers or cpu_count()

    def distribute_load(self, tasks: List[Dict[str, Any]], **kwargs) -> List[Any]:

        with Pool(self.num_workers) as pool:
            results = pool.map(self.worker_fn, tasks)
        return results

    def aggregate_results(self, results: List[Any], aggregate_fn: Callable = None, **kwargs) -> Any:

        # 支持自定义聚合函数，默认返回原列表
        if aggregate_fn:
            return aggregate_fn(results)
        return results

    def load_distributed(self, start_date: str, end_date: str, frequency: str, **kwargs) -> List[Any]:
        """分布式数据加载接口"""
        # 简化的分布式加载实现
        # 在实际应用中，这里应该分发到多个节点
        tasks = [{'start_date': start_date, 'end_date': end_date, 'frequency': frequency}]
        return self.distribute_load(tasks, **kwargs)

    def get_node_info(self) -> Dict[str, Any]:
        """获取节点信息"""
        return {
            'node_type': 'multiprocess',
            'num_workers': self.num_workers,
            'worker_function': self.worker_fn.__name__ if hasattr(self.worker_fn, '__name__') else str(self.worker_fn)
        }

    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        return {
            'status': 'active',
            'node_count': 1,  # 多进程模式只有一个主节点
            'worker_count': self.num_workers,
            'total_tasks_processed': 0  # 可以添加任务计数器
        }
