# src/data/loader/batch_loader.py
from typing import List, Callable, Dict, Any

from src.infrastructure import config
from .base_loader import BaseDataLoader
from ..parallel.dynamic_executor import DynamicExecutor


class BatchDataLoader(BaseDataLoader):
    def __init__(self):
        self.executor = DynamicExecutor(
            initial_workers=4,
            max_workers=16 if config.get('high_perf_mode', False) else 8
        )

    def load_batch(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """批量加载数据"""
        tasks = [self._create_load_task(s, start_date, end_date) for s in symbols]
        results = self.executor.execute_batch(tasks)
        return {s: r for s, r in zip(symbols, results) if r is not None}

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据
        
        返回:
            包含加载器元数据的字典
        """
        return {
            "loader_type": "BatchDataLoader",
            "initial_workers": self.executor.initial_workers,
            "max_workers": self.executor.max_workers,
            "supports_batch": True
        }

    def load(self, *args, **kwargs) -> Any:
        """实现BaseDataLoader的抽象方法，包装load_batch"""
        return self.load_batch(*args, **kwargs)

    def validate(self, data: Any) -> bool:
        """验证加载的数据是否符合预期
        
        参数:
            data: 要验证的数据
            
        返回:
            bool: 数据是否有效
        """
        if not isinstance(data, dict):
            return False
        return all(isinstance(v, dict) for v in data.values())

    def _create_load_task(self, symbol: str, start_date: str, end_date: str) -> Callable:
        """创建加载任务"""

        def task():
            # 实际数据加载逻辑
            return self._load_single(symbol, start_date, end_date)

        return task