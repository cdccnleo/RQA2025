import time
import psutil
from typing import Dict, Any, Optional

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        
    def stop_monitoring(self):
        """停止监控"""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics['duration'] = duration
            self.start_time = None
            
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return self.metrics.copy()

class LoggingMetrics:
    def __init__(self):
        self.log_count = 0
        self.error_count = 0
        self.warning_count = 0
    
    def increment_log_count(self):
        self.log_count += 1
    
    def increment_error_count(self):
        self.error_count += 1
    
    def increment_warning_count(self):
        self.warning_count += 1
    
    def get_metrics(self):
        return {
            'log_count': self.log_count,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }
