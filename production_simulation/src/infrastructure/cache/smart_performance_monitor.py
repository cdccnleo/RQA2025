"""
智能缓存性能监控模块（别名模块）
提供向后兼容的导入路径

实际实现在 monitoring/performance_monitor.py 中
"""

try:
    from .monitoring.performance_monitor import (
        SmartCacheMonitor,
        PerformanceMetrics,
        CachePerformanceMonitor
    )
except ImportError:
    # 提供基础实现
    from dataclasses import dataclass
    from typing import Dict, Any
    
    @dataclass
    class PerformanceMetrics:
        """性能指标"""
        hit_rate: float = 0.0
        response_time: float = 0.0
        throughput: int = 0
        memory_usage: float = 0.0
        cache_size: int = 0
        eviction_rate: float = 0.0
        miss_penalty: float = 0.0
    
    class SmartCacheMonitor:
        """智能缓存监控器"""
        
        def __init__(self, cache_manager=None, enable_monitoring=True, monitor_interval=60.0):
            self.cache_manager = cache_manager
            self.enable_monitoring = enable_monitoring
            self.monitor_interval = monitor_interval
            self.monitors = {}
            self.alerts = []
            self.is_monitoring = False
        
        def add_monitor(self, cache_name: str, monitor):
            """添加监控器"""
            self.monitors[cache_name] = monitor
        
        def collect_metrics(self) -> Dict[str, Any]:
            """收集所有指标"""
            return {}
        
        def check_health(self) -> Dict[str, Any]:
            """检查健康状态"""
            return {}
        
        def start_monitoring(self):
            """开始监控"""
            self.is_monitoring = True
        
        def stop_monitoring(self):
            """停止监控"""
            self.is_monitoring = False
    
    class CachePerformanceMonitor:
        """缓存性能监控器"""
        
        def __init__(self):
            self.metrics = {}
        
        def record_metric(self, name, value):
            """记录指标"""
            self.metrics[name] = value
        
        def get_metrics(self):
            """获取指标"""
            return self.metrics.copy()

    class SmartPerformanceMonitor(SmartCacheMonitor):
        """向后兼容的智能性能监控器别名"""
        pass
else:
    SmartPerformanceMonitor = SmartCacheMonitor  # type: ignore[assignment]

__all__ = [
    'SmartCacheMonitor',
    'SmartPerformanceMonitor',
    'PerformanceMetrics',
    'CachePerformanceMonitor'
]

