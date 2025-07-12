"""调度器优化器模块"""

from typing import Dict, List, Any, Optional
from src.infrastructure.monitoring import MetricsCollector
from src.data.market_data import MarketData
import pandas as pd
import numpy as np
import logging
import time
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class DispatcherOptimizer:
    """调度器优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.market_data = MarketData()
        
        # 调度配置
        self.max_workers = config.get('max_workers', 4)
        self.queue_size = config.get('queue_size', 1000)
        self.batch_size = config.get('batch_size', 100)
        self.timeout_seconds = config.get('timeout_seconds', 30)
        
        # 性能监控
        self.performance_metrics = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'avg_processing_time_ms': 0,
            'queue_size_current': 0,
            'worker_utilization': 0.0
        }
        
        # 初始化调度队列
        self.task_queue = deque(maxlen=self.queue_size)
        self.worker_pool = []
        self.task_results = {}
        self.processing_times = []
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 启动工作线程
        self._start_workers()
        
    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"DispatcherWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_pool.append(worker)
        
        logger.info(f"Started {self.max_workers} dispatcher workers")
    
    def _worker_loop(self):
        """工作线程循环"""
        while True:
            try:
                # 从队列获取任务
                task = self._get_next_task()
                if task is None:
                    time.sleep(0.001)  # 短暂休眠
                    continue
                
                # 处理任务
                start_time = time.time()
                result = self._process_task(task)
                processing_time = (time.time() - start_time) * 1000
                
                # 更新性能指标
                with self._lock:
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > 100:
                        self.processing_times.pop(0)
                    
                    self.performance_metrics['tasks_processed'] += 1
                    self.performance_metrics['avg_processing_time_ms'] = np.mean(self.processing_times)
                    
                    # 存储结果
                    self.task_results[task['id']] = {
                        'result': result,
                        'processing_time_ms': processing_time,
                        'status': 'completed'
                    }
                
                # 记录指标
                self.metrics_collector.record_dispatcher_task(
                    task_type=task.get('type', 'unknown'),
                    processing_time_ms=processing_time,
                    success=True
                )
                
                logger.debug(f"Processed task {task['id']} in {processing_time:.2f}ms")
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                with self._lock:
                    self.performance_metrics['tasks_failed'] += 1
    
    def _get_next_task(self) -> Optional[Dict[str, Any]]:
        """获取下一个任务"""
        with self._lock:
            if self.task_queue:
                return self.task_queue.popleft()
        return None
    
    def _process_task(self, task: Dict[str, Any]) -> Any:
        """处理任务"""
        task_type = task.get('type', 'unknown')
        data = task.get('data', {})
        
        if task_type == 'market_data_update':
            return self._process_market_data_update(data)
        elif task_type == 'order_execution':
            return self._process_order_execution(data)
        elif task_type == 'risk_check':
            return self._process_risk_check(data)
        elif task_type == 'strategy_signal':
            return self._process_strategy_signal(data)
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return None
    
    def _process_market_data_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理市场数据更新任务"""
        symbol = data.get('symbol')
        market_data = data.get('market_data')
        
        if not symbol or not market_data:
            return {'status': 'error', 'message': 'Invalid market data'}
        
        # 更新市场数据
        self.market_data.update_data(symbol, market_data)
        
        return {
            'status': 'success',
            'symbol': symbol,
            'timestamp': market_data.get('timestamp')
        }
    
    def _process_order_execution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订单执行任务"""
        order = data.get('order')
        
        if not order:
            return {'status': 'error', 'message': 'Invalid order'}
        
        # 模拟订单执行
        execution_result = {
            'order_id': order.get('order_id'),
            'symbol': order.get('symbol'),
            'side': order.get('side'),
            'quantity': order.get('quantity'),
            'price': order.get('price'),
            'status': 'filled',
            'execution_time': time.time()
        }
        
        return {
            'status': 'success',
            'execution': execution_result
        }
    
    def _process_risk_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理风险检查任务"""
        order = data.get('order')
        account = data.get('account')
        
        if not order or not account:
            return {'status': 'error', 'message': 'Invalid risk check data'}
        
        # 模拟风险检查
        risk_result = {
            'order_id': order.get('order_id'),
            'risk_level': 'low',
            'approved': True,
            'checks_passed': ['position_limit', 'exposure_limit', 'volatility_check']
        }
        
        return {
            'status': 'success',
            'risk_check': risk_result
        }
    
    def _process_strategy_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理策略信号任务"""
        signal = data.get('signal')
        
        if not signal:
            return {'status': 'error', 'message': 'Invalid signal'}
        
        # 模拟信号处理
        signal_result = {
            'signal_id': signal.get('signal_id'),
            'symbol': signal.get('symbol'),
            'action': signal.get('action'),
            'strength': signal.get('strength', 0.5),
            'processed': True
        }
        
        return {
            'status': 'success',
            'signal_processed': signal_result
        }
    
    def submit_task(self, task: Dict[str, Any]) -> str:
        """提交任务"""
        task_id = f"task_{int(time.time() * 1000)}_{len(self.task_queue)}"
        task['id'] = task_id
        task['submit_time'] = time.time()
        
        with self._lock:
            if len(self.task_queue) >= self.queue_size:
                logger.warning("Task queue is full, dropping oldest task")
                self.task_queue.popleft()
            
            self.task_queue.append(task)
            self.performance_metrics['queue_size_current'] = len(self.task_queue)
        
        logger.debug(f"Submitted task {task_id}")
        return task_id
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        if timeout is None:
            timeout = self.timeout_seconds
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if task_id in self.task_results:
                    return self.task_results[task_id]
            
            time.sleep(0.01)  # 短暂休眠
        
        logger.warning(f"Task {task_id} timeout")
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            metrics = self.performance_metrics.copy()
            
            # 计算工作线程利用率
            active_workers = sum(1 for worker in self.worker_pool if worker.is_alive())
            metrics['worker_utilization'] = active_workers / self.max_workers
            
            # 计算内存使用
            total_memory = len(self.task_queue) * 1024  # 粗略估算
            metrics['memory_usage_mb'] = total_memory / (1024 * 1024)
            
            return metrics
    
    def submit_batch_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """批量提交任务"""
        task_ids = []
        
        for task in tasks:
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(tasks)} tasks")
        return task_ids
    
    def wait_for_batch_completion(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        """等待批量任务完成"""
        if timeout is None:
            timeout = self.timeout_seconds
        
        start_time = time.time()
        completed_tasks = {}
        failed_tasks = []
        
        while time.time() - start_time < timeout:
            with self._lock:
                for task_id in task_ids:
                    if task_id in self.task_results:
                        result = self.task_results[task_id]
                        if result['status'] == 'completed':
                            completed_tasks[task_id] = result
                        else:
                            failed_tasks.append(task_id)
                
                # 检查是否所有任务都完成
                if len(completed_tasks) + len(failed_tasks) == len(task_ids):
                    break
            
            time.sleep(0.01)
        
        return {
            'completed': completed_tasks,
            'failed': failed_tasks,
            'pending': [tid for tid in task_ids if tid not in completed_tasks and tid not in failed_tasks]
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        with self._lock:
            return {
                'queue_size': len(self.task_queue),
                'max_queue_size': self.queue_size,
                'active_workers': sum(1 for worker in self.worker_pool if worker.is_alive()),
                'total_workers': self.max_workers,
                'pending_tasks': len(self.task_queue),
                'completed_tasks': len(self.task_results)
            }
    
    def clear_completed_tasks(self, max_age_seconds: int = 3600):
        """清理已完成的任务"""
        current_time = time.time()
        tasks_to_remove = []
        
        with self._lock:
            for task_id, result in self.task_results.items():
                if 'processing_time_ms' in result:
                    # 估算任务完成时间
                    completion_time = current_time - (result['processing_time_ms'] / 1000)
                    if completion_time > max_age_seconds:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.task_results[task_id]
        
        logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
    
    def shutdown(self):
        """关闭调度器"""
        logger.info("Shutting down dispatcher optimizer")
        
        # 等待队列清空
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if len(self.task_queue) == 0:
                    break
            time.sleep(0.1)
        
        logger.info("Dispatcher optimizer shutdown complete")
