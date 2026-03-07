"""
Logger池统计收集器组件

负责收集Logger池的统计数据和性能指标。
"""

import time
import threading
from typing import Dict, Any, Optional, List

# 导入Logger池相关模块
try:
    from infrastructure.logging.core.interfaces import get_logger_pool
    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False

# 导入统计数据类
try:
    from ...monitoring.logger_pool_monitor import LoggerPoolStats
except ImportError:
    # 如果没有导入成功，定义基础的数据类
    from dataclasses import dataclass
    from typing import Dict, Any
    
    @dataclass
    class LoggerPoolStats:
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


class LoggerPoolStatsCollector:
    """Logger池统计收集器"""
    
    def __init__(self, pool_name: str = "default"):
        """初始化统计收集器"""
        self.pool_name = pool_name
        self.access_times: List[float] = []
        self.max_access_times_size = 1000
        self.history_stats: List[LoggerPoolStats] = []
        self.max_history_size = 1000
        self._lock = threading.RLock()
        
        # 获取Logger池实例
        if POOL_AVAILABLE:
            self.logger_pool = get_logger_pool()
        else:
            self.logger_pool = None
    
    def collect_current_stats(self) -> Optional[LoggerPoolStats]:
        """收集当前统计信息"""
        try:
            if not self.logger_pool:
                stats = self._create_mock_stats()
                # 保存到历史记录
                self._add_to_history(stats)
                return stats
            
            pool_stats = self.logger_pool.get_stats()
            usage_stats = pool_stats.get('usage_stats', {})

            # 计算平均访问时间
            avg_access_time = self._calculate_avg_access_time()

            # 估算内存使用
            memory_usage = self._estimate_memory_usage(pool_stats)

            # 创建统计对象（包含新的优化指标）
            stats = LoggerPoolStats(
                pool_size=pool_stats.get('pool_size', 0),
                max_size=pool_stats.get('max_size', 0),
                created_count=pool_stats.get('created_count', 0),
                hit_count=pool_stats.get('hit_count', 0),
                hit_rate=pool_stats.get('hit_rate', 0.0),
                logger_count=len(pool_stats.get('logger_stats', {})),
                total_access_count=sum(
                    stat.get('use_count', 0)
                    for stat in pool_stats.get('logger_stats', {}).values()
                ),
                avg_access_time=avg_access_time,
                memory_usage_mb=memory_usage,
                timestamp=time.time()
            )
            
            # 添加优化相关的额外指标
            if hasattr(stats, '__dict__'):
                stats.__dict__['warmed_up'] = pool_stats.get('warmed_up', False)
                stats.__dict__['preloaded_count'] = pool_stats.get('preloaded_count', 0)
                stats.__dict__['lru_cache_size'] = pool_stats.get('lru_cache_size', 0)

            # 保存到历史记录
            self._add_to_history(stats)
            return stats

        except Exception as e:
            print(f"收集Logger池统计失败: {e}")
            return None
    
    def record_access_time(self, access_time: float) -> None:
        """记录访问时间（用于性能监控）"""
        with self._lock:
            self.access_times.append(access_time)
            if len(self.access_times) > self.max_access_times_size:
                self.access_times.pop(0)
    
    def _calculate_avg_access_time(self) -> float:
        """计算平均访问时间"""
        if not self.access_times:
            return 0.0
        return sum(self.access_times) / len(self.access_times)
    
    def _estimate_memory_usage(self, pool_stats: Dict[str, Any]) -> float:
        """估算内存使用量 (MB)"""
        try:
            pool_size = pool_stats.get('pool_size', 0)
            
            # 每个Logger实例的估算内存占用
            # BaseLogger实例 + 处理器 + 格式化器 + 缓存
            memory_per_logger = 2.0  # MB (估算值)
            
            total_memory = pool_size * memory_per_logger
            
            # 添加历史数据的内存占用
            history_memory = len(self.history_stats) * 0.1  # 每条历史记录0.1MB
            
            return total_memory + history_memory
            
        except Exception:
            return 0.0
    
    def _add_to_history(self, stats: LoggerPoolStats) -> None:
        """添加统计到历史记录"""
        with self._lock:
            self.history_stats.append(stats)
            
            # 限制历史数据大小
            if len(self.history_stats) > self.max_history_size:
                self.history_stats.pop(0)
    
    def _create_mock_stats(self) -> LoggerPoolStats:
        """创建模拟统计数据（当Logger池不可用时）"""
        return LoggerPoolStats(
            pool_size=10,
            max_size=100,
            created_count=50,
            hit_count=40,
            hit_rate=0.8,
            logger_count=10,
            total_access_count=200,
            avg_access_time=0.001,
            memory_usage_mb=20.0,
            timestamp=time.time()
        )
    
    def get_history_stats(self, limit: Optional[int] = None) -> List[LoggerPoolStats]:
        """获取历史统计数据"""
        with self._lock:
            if limit is None:
                return self.history_stats.copy()
            return self.history_stats[-limit:] if limit > 0 else []
    
    def get_current_access_times(self) -> List[float]:
        """获取当前访问时间列表"""
        with self._lock:
            return self.access_times.copy()
    
    def clear_history(self) -> None:
        """清空历史记录"""
        with self._lock:
            self.history_stats.clear()
            self.access_times.clear()

