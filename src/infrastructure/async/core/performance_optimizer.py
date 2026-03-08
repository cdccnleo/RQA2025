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
    
    def __init__(self, optimization_interval: float = 30.0, resource_manager=None):
        self.optimization_interval = optimization_interval

        # 资源管理器
        self.resource_manager = resource_manager or self._create_default_resource_manager()

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

    def _create_default_resource_manager(self):
        """创建默认资源管理器"""
        # 创建一个简单的Mock资源管理器
        class DefaultResourceManager:
            def get_resource_usage(self):
                return {
                    'cpu_percent': 50.0,
                    'memory_percent': 60.0
                }

        return DefaultResourceManager()

    def _create_default_resource_manager(self):
        """创建默认的资源管理器"""
        class DefaultResourceManager:
            def get_resource_usage(self):
                return {
                    'cpu_percent': 50.0,  # 默认50% CPU使用率
                    'memory_percent': 60.0  # 默认60% 内存使用率
                }
        return DefaultResourceManager()

    def update_performance_metrics(self, processing_time: float, queue_time: float, success: bool = True):
        """
        更新性能指标

        Args:
            processing_time: 处理时间
            queue_time: 队列等待时间
            success: 是否成功
        """
        with threading.Lock():
            self.performance_metrics['total_tasks'] += 1

            if success:
                self.performance_metrics['completed_tasks'] += 1

                # 更新平均处理时间
                total_completed = self.performance_metrics['completed_tasks']
                current_avg = self.performance_metrics['average_processing_time']
                if total_completed == 1:
                    self.performance_metrics['average_processing_time'] = processing_time
                else:
                    self.performance_metrics['average_processing_time'] = (
                        (current_avg * (total_completed - 1)) + processing_time
                    ) / total_completed

                # 更新平均队列时间
                current_avg_queue = self.performance_metrics['average_queue_time']
                if total_completed == 1:
                    self.performance_metrics['average_queue_time'] = queue_time
                else:
                    self.performance_metrics['average_queue_time'] = (
                        (current_avg_queue * (total_completed - 1)) + queue_time
                    ) / total_completed

                # 记录响应时间历史
                self.response_time_history.append(processing_time + queue_time)

            else:
                self.performance_metrics['failed_tasks'] += 1

            # 更新吞吐量（如果有足够的历史数据）
            self._update_throughput()

    def record_task_failure(self):
        """记录任务失败"""
        with threading.Lock():
            self.performance_metrics['total_tasks'] += 1
            self.performance_metrics['failed_tasks'] += 1

    def calculate_throughput(self) -> float:
        """
        Calculate current throughput (tasks per second)
        计算当前吞吐量（每秒任务数）

        Returns:
            float: Current throughput
                   当前吞吐量
        """
        try:
            if len(self.throughput_history) > 0:
                # 计算历史吞吐量的平均值
                return sum(self.throughput_history) / len(self.throughput_history)
            else:
                # 如果没有历史数据，基于当前任务计算简单吞吐量
                total_tasks = self.performance_metrics['total_tasks']
                if total_tasks > 0:
                    # 假设运行时间为1秒作为简单估算
                    return total_tasks / 1.0
                return 0.0
        except Exception as e:
            logger.warning(f"计算吞吐量失败: {e}")
            return 0.0

    def _update_throughput(self):
        """更新吞吐量指标"""
        try:
            if len(self.throughput_history) >= 2:
                # 计算基于历史数据的吞吐量
                recent_throughput = sum(self.throughput_history) / len(self.throughput_history)
                self.performance_metrics['throughput'] = recent_throughput
            else:
                # 如果没有足够历史数据，基于当前任务计算简单吞吐量
                total_tasks = self.performance_metrics['total_tasks']
                if total_tasks > 0:
                    # 假设运行时间为1秒作为简单估算
                    self.performance_metrics['throughput'] = total_tasks / 1.0
        except Exception as e:
            logger.warning(f"更新吞吐量失败: {e}")
            self.performance_metrics['throughput'] = 0.0
        
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
    
    def calculate_resource_utilization(self) -> float:
        """
        Calculate current resource utilization
        计算当前资源利用率

        Returns:
            float: Resource utilization percentage (0.0 to 1.0)
                   资源利用率百分比（0.0到1.0）
        """
        return self._calculate_resource_utilization()

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
    
    def update_resource_utilization(self, utilization: float):
        """
        更新资源利用率指标

        Args:
            utilization: 新的资源利用率值 (0.0 到 1.0)
        """
        if 0.0 <= utilization <= 1.0:
            self.performance_metrics['resource_utilization'] = utilization
            logger.debug(f"资源利用率更新为: {utilization:.2%}")
        else:
            logger.warning(f"无效的资源利用率值: {utilization}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标

        Returns:
            Dict[str, Any]: 性能指标字典
        """
        return self.performance_metrics.copy()

    def get_average_response_time(self) -> float:
        """
        获取平均响应时间

        Returns:
            float: 平均响应时间
        """
        if self.response_time_history:
            return sum(self.response_time_history) / len(self.response_time_history)
        return 0.0

    def get_optimizer_status(self) -> Dict[str, Any]:
        """获取优化器状态"""
        return {
            'is_running': self._running,
            'metrics': self.performance_metrics.copy(),
            'resource_usage': self.resource_manager.get_resource_usage(),
            'resource_limits': self.resource_manager.resource_limits.copy()
        }


__all__ = ['PerformanceOptimizer']

