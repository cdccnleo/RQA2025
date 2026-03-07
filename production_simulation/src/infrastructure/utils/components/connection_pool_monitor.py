"""
连接池监控器组件

负责连接池的性能监控、泄漏检测和统计信息收集。
"""

import logging
import time
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionPoolMonitor:
    """连接池监控器"""
    
    def __init__(self, leak_detection_threshold: float = 60.0):
        """
        初始化监控器
        
        Args:
            leak_detection_threshold: 泄漏检测阈值（秒）
        """
        self.leak_detection_threshold = leak_detection_threshold
        self.leak_detection_enabled = True
        
        # 统计信息
        self.total_created = 0
        self.total_destroyed = 0
        self.total_acquired = 0
        self.total_released = 0
        self.total_timeouts = 0
        self.total_errors = 0
        
        # 性能统计
        self.avg_acquire_time = 0.0
        self.max_acquire_time = 0.0
        self.acquire_times: List[float] = []
        self.max_acquire_times_size = 1000
    
    def record_connection_created(self) -> None:
        """记录连接创建"""
        self.total_created += 1
    
    def record_connection_destroyed(self) -> None:
        """记录连接销毁"""
        self.total_destroyed += 1
    
    def record_connection_acquired(self, acquire_time: float) -> None:
        """记录连接获取"""
        self.total_acquired += 1
        
        # 更新性能统计
        self.acquire_times.append(acquire_time)
        if len(self.acquire_times) > self.max_acquire_times_size:
            self.acquire_times.pop(0)
        
        # 更新平均和最大获取时间
        self.avg_acquire_time = sum(self.acquire_times) / len(self.acquire_times)
        self.max_acquire_time = max(self.max_acquire_time, acquire_time)
    
    def record_connection_released(self) -> None:
        """记录连接释放"""
        self.total_released += 1
    
    def record_timeout(self) -> None:
        """记录超时"""
        self.total_timeouts += 1
    
    def record_error(self) -> None:
        """记录错误"""
        self.total_errors += 1
    
    def detect_connection_leaks(self, active_connections: Dict) -> List[str]:
        """
        检测连接泄漏
        
        Args:
            active_connections: 活跃连接字典
            
        Returns:
            泄漏连接ID列表
        """
        if not self.leak_detection_enabled:
            return []
        
        leaked_connections = []
        current_time = time.time()
        
        for conn_id, (conn_info, acquire_time) in active_connections.items():
            # 检查连接是否长时间未释放
            holding_time = current_time - acquire_time
            if holding_time > self.leak_detection_threshold:
                leaked_connections.append(conn_id)
                logger.warning(
                    f"检测到可能的连接泄漏: {conn_id}, "
                    f"持有时间: {holding_time:.1f}秒"
                )
        
        return leaked_connections
    
    def get_statistics(self, connections: List, available_count: int, 
                      active_count: int) -> Dict[str, Any]:
        """
        获取监控统计信息
        
        Args:
            connections: 所有连接列表
            available_count: 可用连接数
            active_count: 活跃连接数
            
        Returns:
            统计信息字典
        """
        return {
            'total_connections': len(connections),
            'available_connections': available_count,
            'active_connections': active_count,
            'total_created': self.total_created,
            'total_destroyed': self.total_destroyed,
            'total_acquired': self.total_acquired,
            'total_released': self.total_released,
            'total_timeouts': self.total_timeouts,
            'total_errors': self.total_errors,
            'avg_acquire_time': self.avg_acquire_time,
            'max_acquire_time': self.max_acquire_time,
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.total_created = 0
        self.total_destroyed = 0
        self.total_acquired = 0
        self.total_released = 0
        self.total_timeouts = 0
        self.total_errors = 0
        self.avg_acquire_time = 0.0
        self.max_acquire_time = 0.0
        self.acquire_times.clear()

