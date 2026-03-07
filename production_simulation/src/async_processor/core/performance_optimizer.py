"""
性能优化器

异步处理的性能监控和优化。

从async_processing_optimizer.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import threading
import time
from collections import deque
from typing import Dict, Any

from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    性能优化器
    
    负责:
    1. 性能指标收集
    2. 吞吐量计算
    3. 自动优化决策
    """
    
    def __init__(self, optimization_interval: float = 30.0):
        self.optimization_interval = optimization_interval
        
        # 性能指标
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_processing_time': 0.0,
            'average_queue_time': 0.0,
            'throughput': 0.0,
            'resource_utilization': 0.0
        }
        
        # 历史数据
        self.throughput_history = deque(maxlen=100)
        self.response_time_history = deque(maxlen=1000)
        
        # 资源管理器
        self.resource_manager = ResourceManager()
        
        # 优化控制
        self._running = False
        self._optimization_thread: threading.Thread = None
        
        logger.info("性能优化器初始化完成")
    
    def start_optimization(self):
        """启动优化循环"""
        if self._running:
            return
        
        self._running = True
        self._optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self._optimization_thread.start()
        
        logger.info("性能优化器已启动")
    
    def stop_optimization(self):
        """停止优化循环"""
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5)
        
        logger.info("性能优化器已停止")
    
    def _optimization_loop(self):
        """优化循环"""
        while self._running:
            try:
                self._perform_optimization()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"优化循环异常: {e}")
                time.sleep(5)
    
    def _perform_optimization(self):
        """执行优化"""
        try:
            # 计算当前指标
            throughput = self._calculate_throughput()
            utilization = self._calculate_resource_utilization()
            
            # 记录历史
            self.throughput_history.append(throughput)
            
            # 更新性能指标
            self.performance_metrics['throughput'] = throughput
            self.performance_metrics['resource_utilization'] = utilization
            
            # 调整资源限制
            self.resource_manager.adjust_resource_limits(throughput, utilization)
            
            logger.debug(f"优化完成 - 吞吐量: {throughput:.2f}, 利用率: {utilization:.2%}")
            
        except Exception as e:
            logger.error(f"执行优化失败: {e}")
    
    def _calculate_throughput(self) -> float:
        """计算吞吐量"""
        if not self.response_time_history:
            return 0.0
        
        # 简化计算：最近1000个任务的平均处理速度
        total_time = sum(self.response_time_history)
        if total_time > 0:
            return len(self.response_time_history) / total_time
        return 0.0
    
    def _calculate_resource_utilization(self) -> float:
        """计算资源利用率"""
        usage = self.resource_manager.get_resource_usage()
        
        if not usage:
            return 0.0
        
        # 综合CPU和内存利用率
        cpu_util = usage.get('cpu_percent', 0) / 100.0
        mem_util = usage.get('memory_percent', 0) / 100.0
        
        return (cpu_util + mem_util) / 2.0
    
    def record_task_completion(self, processing_time: float, success: bool):
        """记录任务完成"""
        self.performance_metrics['total_tasks'] += 1
        
        if success:
            self.performance_metrics['completed_tasks'] += 1
        else:
            self.performance_metrics['failed_tasks'] += 1
        
        self.response_time_history.append(processing_time)
        
        # 更新平均处理时间
        if self.performance_metrics['total_tasks'] > 0:
            self.performance_metrics['average_processing_time'] = (
                sum(self.response_time_history) / len(self.response_time_history)
            )
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """获取优化器状态"""
        return {
            'is_running': self._running,
            'metrics': self.performance_metrics.copy(),
            'resource_usage': self.resource_manager.get_resource_usage(),
            'resource_limits': self.resource_manager.resource_limits.copy()
        }


__all__ = ['PerformanceOptimizer']

