"""
资源管理器

异步处理的资源限制和监控管理。

从async_processing_optimizer.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import psutil
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    资源管理器
    
    负责:
    1. CPU和内存监控
    2. 资源限制管理
    3. 资源使用统计
    """
    
    def __init__(self, max_concurrent_tasks: int = 100):
        self.resource_limits = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'max_threads': min(32, max_concurrent_tasks),
            'max_processes': min(8, max_concurrent_tasks // 4)
        }
        
        logger.info(f"资源管理器初始化: CPU限制{self.resource_limits['cpu_percent']}%, 内存限制{self.resource_limits['memory_percent']}%")
    
    def check_resource_limits(self) -> bool:
        """检查资源是否在限制范围内"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > self.resource_limits['cpu_percent']:
                logger.warning(f"CPU使用率超限: {cpu_percent}% > {self.resource_limits['cpu_percent']}%")
                return False
            
            if memory_percent > self.resource_limits['memory_percent']:
                logger.warning(f"内存使用率超限: {memory_percent}% > {self.resource_limits['memory_percent']}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"资源检查失败: {e}")
            return True  # 检查失败时允许继续
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 ** 2),
                'cpu_count': psutil.cpu_count(),
                'within_limits': cpu_percent <= self.resource_limits['cpu_percent'] and 
                                memory.percent <= self.resource_limits['memory_percent']
            }
            
        except Exception as e:
            logger.error(f"获取资源使用失败: {e}")
            return {}
    
    def adjust_resource_limits(self, throughput: float, utilization: float):
        """根据吞吐量和利用率调整资源限制"""
        try:
            # 高利用率时增加资源限制
            if utilization > 0.8:
                new_workers = min(self.resource_limits['max_threads'] + 2, 64)
                self.resource_limits['max_threads'] = new_workers
                logger.info(f"提高线程限制至 {new_workers}")
            
            # 低利用率时减少资源限制
            elif utilization < 0.3:
                new_workers = max(self.resource_limits['max_threads'] - 1, 4)
                self.resource_limits['max_threads'] = new_workers
                logger.info(f"降低线程限制至 {new_workers}")
                
        except Exception as e:
            logger.error(f"调整资源限制失败: {e}")


__all__ = ['ResourceManager']

