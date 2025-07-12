# RQA2025 数据层功能增强完整报告（续）

## 2. 功能分析（续）

### 2.1 性能优化（续）

#### 2.1.3 数据预加载机制

**现状分析**：
当前无法预先加载可能使用的数据，每次请求都需要重新加载，导致首次加载时间较长，影响用户体验。

**实现建议**：
实现一个 `DataPreloader` 类，用于预先加载可能使用的数据。该类将提供以下功能：

- 后台线程：使用守护线程在后台执行预加载任务
- 预加载队列：管理预加载任务队列
- 任务管理：支持添加、执行和取消预加载任务

在 `DataManager` 中集成预加载功能，提供 `preload_data` 方法，支持预先加载数据。

**核心代码示例**：
```python
import threading
import queue
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreloader:
    """数据预加载器"""
    
    def __init__(
        self,
        max_queue_size: int = 100,
        worker_count: int = 1
    ):
        """
        初始化数据预加载器
        
        Args:
            max_queue_size: 最大队列大小
            worker_count: 工作线程数
        """
        self.queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.worker_count = worker_count
        self.workers = []
        self.running = False
        self.lock = threading.Lock()
        
        # 任务计数器（用于生成唯一任务ID）
        self.task_counter = 0
        
        # 任务结果存储
        self.results = {}
        self.result_lock = threading.Lock()
    
    def start(self):
        """启动预加载器"""
        with self.lock:
            if self.running:
                return
            
            self.running = True
            
            # 创建工作线程
            for i in range(self.worker_count):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"PreloaderWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
            
            logger.info(f"Started {self.worker_count} preloader workers")
    
    def stop(self):
        """停止预加载器"""
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            # 清空队列
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break
            
            # 等待工作线程结束
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)
            
            self.workers = []
            logger.info("Stopped preloader workers")
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                # 从队列获取任务
                priority, task_id, func, args, kwargs = self.queue.get(timeout=1.0)
                
                try:
                    # 执行任务
                    logger.debug(f"Executing preload task {task_id}: {func.__name__}")
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # 存储结果
                    with self.result_lock:
                        self.results[task_id] = {
                            'result': result,
                            'timestamp': datetime.now().isoformat(),
                            'execution_time': end_time - start_time
                        }
                    
                    logger.debug(f"Preload task {task_id} completed in {end_time - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Preload task {task_id} failed: {e}")
                    
                    # 存储错误
                    with self.result_lock:
                        self.results[task_id] = {
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                
                # 标记任务完成
                self.queue.task_done()
            
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"Preloader worker error: {e}")
    
    def add_task(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = {},
        priority: int = 0
    ) -> int:
        """
        添加预加载任务
        
        Args:
            func: 要执行的函数
            args: 函数参数
            kwargs: 函数关键字参数
            priority: 优先级（数字越小优先级越高）
            
        Returns:
            int: 任务ID
        """
        with self.lock:
            if not self.running:
                self.start()
            
            # 生成任务ID
            task_id = self.task_counter
            self.task_counter += 1
            
            # 添加任务到队列
            self.queue.put((priority, task_id, func, args, kwargs))
            logger.debug(f"Added preload task {task_id}: {func.__name__}")
            
            return task_id
    
    def get_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 任务结果，如果不存在则返回None
        """
        with self.result_lock:
            return self.results.get(task_id)
    
    def clear_results(self, older_than: Optional[str] = None) -> int:
        """
        清除任务结果
        
        Args:
            older_than: 清除早于指定时间的结果，格式为ISO 8601，默认为None（清除所有）
            
        Returns:
            int: 清除的结果数量
        """
        with self.result_lock:
            if older_than is None:
                count = len(self.results)
                self.results.clear()
                return count
            
            # 清除早于指定时间的结果
            to_remove = []
            for task_id, result in self.results.items():
                if result.get('timestamp', '') < older_than:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.results[task_id]
            
            return len(to_remove)
    
    def get_queue_size(self) -> int:
        """
        获取队列大小
        
        Returns:
            int: 队列中的任务数量
        """
        return self.queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'queue_size': self.queue.qsize(),
            'worker_count': len([w for w in self.workers if w.is_alive()]),
            'result_count': len(self.results),
            'running': self.running
        }
```

### 2.2 功能扩展

#### 2.2.1 数据质量监控

**现状分析**：
缺乏对数据质量的系统化监控机制，无法及时发现和处理数据质量问题，影响模型训练和预测的准确性。

**实现建议**：
实现一个 `DataQualityMonitor` 类，用于监控数据质量。该类将提供以下功能：

- 缺失值检查：检查数据中的缺失值比例
- 重复值检查：检查数据中的重复值比例
- 异常值检查：使用 IQR 或 Z-Score 方法检测异常值
- 数据类型检查：检查数据类型是否符合预期
- 日期范围检查：检查数据的日期范围是否完整
- 股票代码覆盖率检查：检查股票代码的覆盖率

在 `DataManager` 中集成数据质量监控功能，提供 `check_data_quality` 方法，支持全面检查数据质量。

**核心代码示例**：
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Set
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self):
        """初始化数据质量监控器"""
        pass
    
    def check_quality(
        self,
        data: pd.DataFrame,
        date_column: Optional[str] = None,
        symbol_column: Optional[str] = None,
        expected_symbols: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            data: 数据
            date_column: 日期列名
            symbol_column: 股票代码列名
            expected_symbols: 预期的股票代码列表
            numeric_columns: 数值列列表
            
        Returns:
            Dict[str, Any]: 数据质量报告
        """
        if data is None or data.empty:
            return {
                'status': 'error',
                'message': 'Empty data',
                'timestamp': datetime.now().isoformat()
            }
        
        # 基本信息
        quality_report = {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
        }
        
        # 检查缺失值
        quality_report['missing_values'] = self._check_missing_values(data)
        
        # 检查重复值
        quality_report['duplicates'] = self._check_duplicates(data)
        
        # 检查数据类型
        quality_report['data_types'] = self._check_data_types(data)
        
        # 检查异常值
        if numeric_columns:
            quality_report['outliers'] = self._check_outliers(data, numeric_columns)
        
        # 检查日期范围
        if date_column and date_column in data.columns:
            quality_report['date_range'] = self._check_date_range(data, date_column)
        
        # 检查股票代码覆盖率
        if symbol_column and symbol_column in data.columns:
            quality_report['symbol_coverage'] = self._check_symbol_coverage(
                data, symbol_column, date_column, expected_symbols
            )
        
        return quality_report
    
    def _check_missing_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        检查缺失值
        
        Args:
            data: 数据
            
        Returns:
            Dict[str, float]: 每列的缺失值比例
        """
        missing_values = {}
        for column in data.columns:
            missing_ratio = data[column].isna().mean()
            if missing_ratio > 0:
                missing_values[column] = missing_ratio
        
        return missing_values
    
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查重复值
        
        Args:
            data: 数据
            
        Returns:
            Dict[str, Any]: 重复值信息
        """
        duplicate_count = data.duplicated().sum()
        duplicate_ratio = duplicate_count / len(data) if len(data) > 0 else 0
        
        return {
            'duplicate_count': int(duplicate_count),
            'duplicate_ratio': float(duplicate_ratio)
        }
    
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        检查数据类型
        
        Args:
            data: 数据
            
        Returns:
            Dict[str, str]: 每列的数据类型
        """
        data_types = {}
        for column in data.columns:
            data_types[column] = str(data[column].dtype)
        
        return data_types
    
    def _check_outliers(
        self,
        data: pd.DataFrame,
        numeric_columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, Dict[str, Any]]:
        """
        检查异常值
        
        Args:
            data: 数据
            numeric_columns: 数值列列表
            method: 异常值检测方法，'iqr'或'zscore'
            threshold: 阈值
            
        Returns:
            Dict[str, Dict[str, Any]]: 每列的异常值信息
        """
        outliers = {}
        
        for column in numeric_columns:
            if column not in data.columns:
                continue
            
            # 跳过非数值列
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue
            
            # 跳过全是缺失值的列
            if data[column].isna().all():
                continue
            
            # 计算异常值
            values = data[column].dropna()
            
            if method == 'iqr':
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask = (values < lower_bound) | (values > upper_bound)
            elif method == 'zscore':
                mean = values.mean()
                std = values.std()
                z_scores = (values - mean) / std
                outlier_mask = z_scores.abs() > threshold
            else:
                raise ValueError