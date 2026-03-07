"""
Logger池监控循环管理器组件

负责管理Logger池的监控循环、数据收集和状态更新。
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from infrastructure.logging.core.interfaces import get_logger_pool
    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False


@dataclass
class LoggerPoolStats:
    """Logger池统计数据"""
    pool_size: int
    max_size: int
    created_count: int
    hit_count: int
    hit_rate: float
    logger_count: int
    total_access_count: int
    avg_access_time: float
    memory_usage_mb: float
    timestamp: float


class LoggerPoolMonitoringLoop:
    """Logger池监控循环管理器"""
    
    def __init__(self, pool_name: str = "default", collection_interval: int = 60):
        """初始化监控循环管理器"""
        self.pool_name = pool_name
        self.collection_interval = collection_interval
        self.max_history_size = 1000
        self.max_access_times_size = 1000
        
        # 状态管理
        self.current_stats: Optional[LoggerPoolStats] = None
        self.history_stats: List[LoggerPoolStats] = []
        self.access_times: List[float] = []
        self._lock = threading.RLock()
        
        # 获取Logger池实例
        if POOL_AVAILABLE:
            self.logger_pool = get_logger_pool()
        else:
            self.logger_pool = None
    
    def collect_initial_stats(self) -> None:
        """收集初始统计信息"""
        try:
            self.collect_current_stats()
        except Exception as e:
            print(f"收集初始Logger池统计失败: {e}")
    
    def collect_current_stats(self) -> Optional[LoggerPoolStats]:
        """收集当前统计信息"""
        if not self.logger_pool:
            return None
            
        try:
            pool_stats = self.logger_pool.get_stats()
            usage_stats = pool_stats.get('usage_stats', {})

            # 计算平均访问时间
            avg_access_time = self._calculate_avg_access_time()

            # 估算内存使用
            memory_usage = self._estimate_memory_usage(pool_stats)

            # 创建统计对象
            stats = LoggerPoolStats(
                pool_size=pool_stats.get('pool_size', 0),
                max_size=pool_stats.get('max_size', 0),
                created_count=pool_stats.get('created_count', 0),
                hit_count=pool_stats.get('hit_count', 0),
                hit_rate=pool_stats.get('hit_rate', 0.0),
                logger_count=len(pool_stats.get('loggers', [])),
                total_access_count=self._calculate_total_access_count(usage_stats),
                avg_access_time=avg_access_time,
                memory_usage_mb=memory_usage,
                timestamp=time.time()
            )

            with self._lock:
                self.current_stats = stats
                self._add_to_history(stats)

            return stats

        except Exception as e:
            print(f"收集Logger池统计失败: {e}")
            return None
    
    def update_access_time(self, access_time: float) -> None:
        """更新访问时间记录"""
        with self._lock:
            self.access_times.append(access_time)
            
            # 限制访问时间记录大小
            if len(self.access_times) > self.max_access_times_size:
                self.access_times.pop(0)
    
    def get_current_stats(self) -> Optional[LoggerPoolStats]:
        """获取当前统计信息"""
        with self._lock:
            return self.current_stats
    
    def get_history_stats(self, limit: int = 100) -> List[LoggerPoolStats]:
        """获取历史统计信息"""
        with self._lock:
            return self.history_stats[-limit:] if limit > 0 else self.history_stats.copy()
    
    def get_current_access_times(self) -> List[float]:
        """获取当前访问时间列表"""
        with self._lock:
            return self.access_times.copy()
    
    def _calculate_avg_access_time(self) -> float:
        """计算平均访问时间"""
        with self._lock:
            if not self.access_times:
                return 0.0
            return sum(self.access_times) / len(self.access_times)
    
    def _estimate_memory_usage(self, pool_stats: Dict[str, Any]) -> float:
        """估算内存使用量 (MB)"""
        try:
            pool_size = pool_stats.get('pool_size', 0)

            # 每个Logger实例的估算内存占用
            memory_per_logger = 2.0  # MB (估算值)
            total_memory = pool_size * memory_per_logger

            # 添加历史数据的内存占用
            with self._lock:
                history_memory = len(self.history_stats) * 0.1  # 每条历史记录0.1MB

            return total_memory + history_memory

        except Exception:
            return 0.0
    
    def _calculate_total_access_count(self, usage_stats: Dict[str, Any]) -> int:
        """计算总访问次数"""
        try:
            return sum(
                stat.get('access_count', 0)
                for stat in usage_stats.values()
            )
        except Exception:
            return 0
    
    def _add_to_history(self, stats: LoggerPoolStats) -> None:
        """添加统计数据到历史记录"""
        self.history_stats.append(stats)
        
        # 限制历史数据大小
        if len(self.history_stats) > self.max_history_size:
            self.history_stats.pop(0)
